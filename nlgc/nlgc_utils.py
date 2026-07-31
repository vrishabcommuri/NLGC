from multiprocessing import cpu_count, current_process
from functools import cached_property
import copy
import numpy as np
import pickle
import warnings
from functools import reduce
from joblib import Parallel, delayed, parallel_backend
from mne.utils import logger
from nlgc.opt import (NeuraLVAR, NeuraLVARCV, create_shared_mem, 
                      link_share_memory)
from nlgc.stat import fdr_control
from nlgc.bias_utils import debias_deviances
from nlgc.parallel_gc import batched_test_links
from nlgc.opt.proximal import instantiate_proximal_solvers
from nlgc.opt.em import final_log_likelihood, _copycast_em_state_numpy
import dataclasses
from nlgc.utils.restriction import (roi_to_link_restriction, 
                                    _expand_roi_indices_as_tup)
from nlgc.bias_utils import compute_bias
from nlgc.config import ModelMultiprocessConfig

class NLGC:
    """NLGC object

    Provides a an object including captured connectivity map via NLGC and its related parameters.

    Parameters
    ----------
    subject: str
        subject_id
    nx: int
        n_sources
    ny: int
        n_sensors
    t: int
        n_samples
    p: int
        VAR model order
    n_eigenmodes: int
        number of eigenmodes
    n_segments: int
        number of chunks used for non-centrality parameter estimation
    d_raw: numpy array (n_sources * n_sources)
        *biased* deviance matrix
    bias_f: float
        full model bias (scalar)
    biar_r: numpy array (n_sources * n_sources)
        reduced model bias matrix, [.]_{i,j} corresponds to link j->i
    """
    def __init__(self, subject, nx, ny, t, p, n_eigenmodes, n_orients, 
                 n_segments, d_raw, bias_f, bias_r, model_f, nonconv_flag, 
                 label_names, label_vertidx, forward_orig, whitener, 
                 eig_src_weights, debug=None):

        self.subject = subject
        self.nx = nx
        self.ny = ny
        self.t = t
        self.p = p
        self.n_eigenmodes = n_eigenmodes
        self.n_orients = n_orients
        self.n_segments = n_segments
        self.d_raw = d_raw
        self.bias_f = bias_f
        self.bias_r = bias_r
        self._model_f = model_f
        self._nonconv_flag = nonconv_flag
        self._labels = label_names
        self._label_vertidx = label_vertidx
        self.forward_orig = forward_orig
        self.whitener = whitener
        self.eig_src_weights = eig_src_weights
        self._debug = debug
    

    @cached_property
    def avg_debiased_dev(self):
        """averaging the calculted deviances across chunks (n_segments)

            """
        debiased_deviances = [debias_deviances(*args) 
                              for args in zip(self.d_raw, 
                                              self.bias_f, 
                                              self.bias_r)]
        if self.n_segments > 1:
            return reduce(lambda x, y: x + y, debiased_deviances) / self.n_segments
        else:
            return debiased_deviances[0]
        

    def get_J_statistics(self, alpha=0.1):
        """calculating J-stat (connectivity map) from deviance matrix

        Parameters
        ----------
        alpha : float
            individual-level confidence interval
            """
        
        eff_eigenmodes = self.n_orients * self.n_eigenmodes

        return fdr_control(self.avg_debiased_dev, self.p * (eff_eigenmodes**2), alpha)


    def pickle_as(self, filename):
        """saving the object as a pickle

        Parameters
        ----------
        filename : str
            file name (including directory address)
            """
        if filename.endswith('.pkl') or filename.endswith('.pickled') or filename.endswith('.pickle'):
            pass
        else:
            filename += '.pkl'

        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)


def gc_extraction(y, F, R, ROIs, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    
    instantiate_proximal_solvers(config, em_state.N_sources_upper)

    lambda_range = config.sparsity.lambda_range

    if lambda_range is None:
        raise ValueError("lambda range must be a float or list of floats")
    
    if config.numerical.verbose:
        print("running full model fit")

    if len(lambda_range) > 1:
        # pick best lambda from list
        model_f = NeuraLVARCV.from_config(config)
        em_state, smoother_result = model_f.fit(y, F, R, 
                                                copy.deepcopy(em_state))
    else:
        model_f = NeuraLVAR.from_config(config)
        lambda_ = lambda_range[0]
        em_state, smoother_result = model_f.fit(y, F, R, lambda_, 
                                               copy.deepcopy(em_state))
        
    lambda_ = model_f.lambda_
        
    if config.numerical.verbose:
        print(f"finished full model fit in {em_state.em_iter} EM iterations")
    
    bias_f = compute_bias(em_state, smoother_result, config)

    sparsity, ROIs = sparsity_mask(em_state, smoother_result, ROIs, config)
    links_to_check = roi_to_link_restriction(ROIs, sparsity, 
                                             eff_eigenmodes, config)

    if config.numerical.verbose:
        print(f"Checking {len(links_to_check)} links...")
    
    if isinstance(config.parallel, ModelMultiprocessConfig):
        dev_raw, bias_r, nonconv_flag = multiprocess_test_links(links_to_check, 
                                                                y, F, R, 
                                                                lambda_,
                                                                em_state, 
                                                                config)
    else:
        dev_raw, bias_r, nonconv_flag = batched_test_links(links_to_check, 
                                                           y, F, R, lambda_,
                                                           em_state, config)

    return dev_raw, bias_r, bias_f, model_f, nonconv_flag


def sparsity_mask(em_state, smoother_result, ROIs, config):
    m = em_state.N_sources_upper
    A = em_state.A[:m]
    smoothed_state = smoother_result.smoothed_state
    order = config.latent.order
    n_orients = config.latent.n_orients
    n_eigenmodes = config.latent.n_eigenmodes
    eff_eigenmodes = n_orients * n_eigenmodes
    energy_thresh = config.sparsity.negligible_candidate_link_energy_thr

    N = m // eff_eigenmodes

    A_blocks = A.reshape(N, eff_eigenmodes, order, N, eff_eigenmodes)

    block_strength = np.sqrt(np.sum(A_blocks**2, axis=(1,2,4)))

    # don't include self-links in strength calculation
    block_strength = block_strength * (~np.eye(N).astype(bool)).astype(float)

    if config.sparsity.negligible_candidate_link_energy_thr < 1 and \
            block_strength.sum() > 0:
        link_power = block_strength.ravel() ** 2

        sorted_idx = np.argsort(link_power)[::-1]

        cumul_power = np.cumsum(link_power[sorted_idx])
        cumul_power /= cumul_power[-1]

        idx = np.searchsorted(cumul_power, energy_thresh)

        keep_idx = sorted_idx[:idx + 1]

        sparsity_mask = np.zeros_like(link_power, dtype=bool)
        sparsity_mask[keep_idx] = 1.0

        sparsity_mask = sparsity_mask.reshape(N, N)
        if config.numerical.verbose:
            print(f"retained {np.count_nonzero(sparsity_mask)}/"
                  f"{np.count_nonzero(block_strength)} candidate "
                  "links (dropped lowest  "
                  f"{(1-energy_thresh)*100:.5f}% of total off diag energy)")
            
        sparsity = block_strength * sparsity_mask

        np.count_nonzero(sparsity), np.count_nonzero(sparsity_mask)

        assert np.count_nonzero(sparsity) == \
               np.count_nonzero(sparsity_mask)
    else:
        sparsity = block_strength

    if config.sparsity.var_thr < 1:
        x = smoothed_state[:, :em_state.N_sources_upper]

        # energy of each latent state
        state_power = np.sum(x**2, axis=0)

        # group all eigenmodes/orientations belonging to one ROI
        N_roi = em_state.N_sources_upper // eff_eigenmodes

        roi_power = state_power.reshape(N_roi, eff_eigenmodes).sum(axis=1)

        sorted_idx = np.argsort(roi_power)[::-1]

        cumul_power = np.cumsum(roi_power[sorted_idx])
        cumul_power /= cumul_power[-1]

        idx = np.searchsorted(cumul_power, config.sparsity.var_thr)

        ROIs = sorted_idx[:idx + 1]

    return sparsity, ROIs


def multiprocess_test_links(links_to_check, y, F, R, lambda_, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    # F is the companion gain, (n_sensors, m*p) -- deriving nx from F.shape[1]
    # gave an ROI count p times too large. batched_test_links uses
    # N_sources_upper for exactly this reason.
    m = em_state.N_sources_upper
    nx = m // eff_eigenmodes
    fullmodel_log_likelihood = final_log_likelihood(em_state)

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    nonconv_flag = np.zeros((nx, nx), dtype=np.bool_)

    if len(links_to_check) == 0:
        return dev_raw, bias_r, nonconv_flag

    # em_state arrives JAX-backed (model_f.fit ends in em_jax), and jax.Array has
    # no `.flags` -- align_cast inside forward_filter_blas would raise on the
    # first em_blas iteration. em_blas does call _copycast_em_state_numpy, but
    # only AFTER proximal_param_update, which is too late. Casting here also
    # means each task pickles plain numpy instead of ArrayImpl.
    em_state = _copycast_em_state_numpy(em_state)

    # Memory management for Parallel implementation
    _, info_y, shm_y = create_shared_mem(y)
    _, info_f, shm_f = create_shared_mem(F)
    shared_bias_r, info_bias_r, shm_bias_r = create_shared_mem(bias_r)
    shared_ll_r, info_ll_r, shm_ll_r = create_shared_mem(dev_raw)
    # allocate from nonconv_flag, not dev_raw -- otherwise the flags come back
    # as float64
    shared_nonconv_flag, info_nonconv_flag, shm_nonconv_flag = \
        create_shared_mem(nonconv_flag)

    shared_args = (info_y, info_f, info_bias_r, info_ll_r, info_nonconv_flag)
    args = (R, lambda_, em_state, config)

    # config.parallel.n_workers used to be ignored entirely (n_jobs was always
    # min(cpu_count(), n_links)). Each worker holds its own JAX runtime and
    # recompiles em_jax, so oversubscribing is expensive; inner_max_num_threads=1
    # stops every worker from also spinning up its own BLAS pool.
    n_jobs = min(config.parallel.n_workers, len(links_to_check))
    with parallel_backend('loky', inner_max_num_threads=1):
        Parallel(n_jobs=n_jobs,
                 verbose=10 if config.numerical.verbose else 0)(
            delayed(_learn_reduced_model_parallel)(
                link, *(shared_args + args)
            ) for link in links_to_check
        )

    ll_r = np.reshape(shared_ll_r, dev_raw.shape).copy()
    bias_r = np.reshape(shared_bias_r, dev_raw.shape).copy()
    nonconv_flag = np.reshape(shared_nonconv_flag,
                              dev_raw.shape).astype(np.bool_)

    for shm in (shm_nonconv_flag, shm_bias_r, shm_f, shm_ll_r, shm_y):
        shm.close()
        try:
            shm.unlink()
        except Exception as exc:
            print(f"\nUnlink shared-memory issue: {exc}")

    indices = tuple(z for z in zip(*links_to_check))
    dev_raw[indices] = 2 * fullmodel_log_likelihood
    dev_raw[indices] -= 2 * ll_r[indices]

    return dev_raw, bias_r, nonconv_flag


def _learn_reduced_model(i, j, y, F, R, lambda_f, em_state, config):
    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    eff_eigenmodes = n_eigenmodes * n_orients
    
    target = _expand_roi_indices_as_tup(j, eff_eigenmodes)
    source = _expand_roi_indices_as_tup(i, eff_eigenmodes)
    if config.numerical.verbose:
        print(f'target: {target}')
        print(f'source: {source}')
    link = '->'.join(map(lambda x: ','.join(map(str, x)), (source, target)))
    
    model_r = NeuraLVAR.from_config(config)
    em_state, smoother_result = model_r.fit(y, F, R, lambda_f, em_state, 
                                            restriction=link)
    bias = compute_bias(em_state, smoother_result, config)
    ll = final_log_likelihood(em_state)
    nonconv_flag = em_state.em_iter == config.optimizer.max_iter
    return ll, bias, nonconv_flag


def _reset_reduced_model_state(em_state):
    """Clear the covariance seeds a reduced-model fit must not inherit.

    P0/N0 warm-start the steady-state DARE solve and were computed against the
    FULL model's A, so they are stale once a link is masked out.
    parallel_gc.batch_em_state zeroes them for the same reason.

    em_iter and log_likelihood are per-fit and are reset by solve_params, which
    this path reaches via NeuraLVAR.fit -- so they are deliberately not touched
    here. The parameter warm start (A, Q from the full model) is intentional.
    """
    return dataclasses.replace(
        em_state,
        P0=np.zeros_like(np.asarray(em_state.P0)),
        N0=np.zeros_like(np.asarray(em_state.N0)),
    )


def _learn_reduced_model_parallel(link_index, info_y, info_f, info_bias_r,
                                  info_ll_r, info_nonconv_flag, R, lambda_f,
                                  em_state, config):

    # loky spawns workers, so nlgc.opt.proximal is re-imported fresh here and its
    # module-level `solver`/`solve_for_Q` are None -- gc_extraction only ever
    # instantiated them in the parent. proximal_param_update reads those globals,
    # so without this every worker dies with 'NoneType' has no attribute 'run'.
    # Idempotent: a reused worker rebuilds nothing on its second task.
    instantiate_proximal_solvers(config, em_state.N_sources_upper)

    # each process will mutate this after this point
    em_state = _reset_reduced_model_state(copy.deepcopy(em_state))

    try:
        y, shm_y = link_share_memory(info_y)
        F, shm_f = link_share_memory(info_f)
        bias_r, shm_bias_r = link_share_memory(info_bias_r)
        ll_r, shm_ll_r = link_share_memory(info_ll_r)
        nonconv_flag, shm_nonconv_flag = link_share_memory(info_nonconv_flag)
    except BaseException as e:
        logger.error("Could not link to memory")
        raise e

    j, i = link_index
    logger.debug(f"{current_process().name} working on {i, j}th link")
    ll, bias, flag = _learn_reduced_model(i, j, y, F, R, lambda_f, 
                                          em_state, config)
    ll_r[j, i] = ll
    bias_r[j, i] = bias
    nonconv_flag[j, i] = flag
    for shm in (shm_y, shm_f, shm_bias_r, shm_ll_r, shm_nonconv_flag):
        shm.close()







