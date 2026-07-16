from multiprocessing import cpu_count, current_process
from functools import cached_property
import copy
import numpy as np
import pickle
import warnings
from functools import reduce
from joblib import Parallel, delayed
from scipy import optimize, linalg
from mne.utils import logger
from nlgc.opt import (NeuraLVAR, NeuraLVARCV, create_shared_mem, 
                      link_share_memory)
from nlgc.stat import fdr_control
from nlgc.bias_utils import debias_deviances
from nlgc.vmap_gc import batched_test_links
from nlgc.opt.proximal import instantiate_proximal_solvers
from nlgc.utils.restriction import (roi_to_link_restriction, 
                                    _expand_roi_indices_as_tup)


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


def gc_extraction(y, F, r, ROIs, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    n, m = F.shape
    R = r * np.eye(n)

    instantiate_proximal_solvers(config, m)

    lambda_range = config.validation.lambda_range
    if lambda_range is None:
        raise ValueError("lambda range must be a float or list of floats")
    
    em_state.Q = _init_Q(y, F)

    print('Start creating models')

    if len(lambda_range) > 1:
        # pick best lambda from list
        model_f = NeuraLVARCV.from_config(config)
        em_state, smoother_state = model_f.fit(y, F, R, copy.deepcopy(em_state))
    else:
        model_f = NeuraLVAR.from_config(config)
        lambda_ = lambda_range[0]
        em_state, smoother_state = model_f.fit(y, F, R, lambda_, 
                                               copy.deepcopy(em_state))
    
    print('Finished fitting models')

    bias_f = model_f.compute_bias(em_state)

    warnings.filterwarnings('ignore')


    sparsity = np.linalg.norm(model_f._parameters[0], axis=0, ord=1) * \
               np.diag(model_f._parameters[2])[None, :]

    if config.sparsity.var_thr < 1:
        x_ = np.sum(model_f._parameters[4][:, :m] ** 2, axis=0)
        total_power = np.zeros(m // eff_eigenmodes)
        for n in range(eff_eigenmodes):
            total_power += x_[n::eff_eigenmodes]
        sorted_idx = np.argsort(total_power)[::-1]
        sorted_pow_ratio = np.cumsum(total_power[sorted_idx])
        sorted_pow_ratio /= sorted_pow_ratio[-1]
        idx = ((sorted_pow_ratio > config.sparsity.var_thr) != 0).argmax()
        ROIs = sorted_idx[:idx + 1]

    print('Checking links')

    links_to_check = roi_to_link_restriction(ROIs, sparsity, 
                                             eff_eigenmodes, config)

    if config.numerical.verbose:
        print(f"Checking {len(links_to_check)} links...")
    
    if config.optimizer.vmap_gc:
        dev_raw, bias_r, nonconv_flag = batched_test_links(links_to_check, 
                                                           model_f, y, F, R, 
                                                           em_state, config)
    else:
        dev_raw, bias_r, nonconv_flag = multiprocess_test_links(links_to_check, 
                                                                model_f, y, F, 
                                                                R, em_state, 
                                                                config)

    return dev_raw, bias_r, bias_f, model_f, nonconv_flag



def multiprocess_test_links(links_to_check, model_f, y, F, R, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    n, m = F.shape
    nx = m // (eff_eigenmodes)
    lambda_ = model_f.lambda_

    dev_raw = np.zeros((nx, nx))
    bias_r = np.zeros((nx, nx))
    nonconv_flag = np.zeros((nx, nx), dtype=np.bool_)

    if len(links_to_check) == 0:
        return dev_raw, bias_r, nonconv_flag

    # Memory management for Parallel implementation
    _, info_y, shm_y = create_shared_mem(y)
    _, info_f, shm_f = create_shared_mem(F)
    shared_bias_r, info_bias_r, shm_bias_r = create_shared_mem(bias_r)
    shared_ll_r, info_ll_r, shm_ll_r = create_shared_mem(dev_raw)
    shared_nonconv_flag, info_nonconv_flag, shm_nonconv_flag = \
        create_shared_mem(dev_raw)
    
    shared_args = (info_y, info_f, info_bias_r, info_ll_r, info_nonconv_flag) 
    args = (R, lambda_, em_state, config)  

   
    n_jobs = min(cpu_count(), len(links_to_check))
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_learn_reduced_model_parallel)(
            link, *(shared_args + args)
        ) for link in links_to_check
    )

    ll_r = np.reshape(shared_ll_r, dev_raw.shape).copy()
    bias_r = np.reshape(shared_bias_r, dev_raw.shape).copy()
    nonconv_flag = np.reshape(shared_nonconv_flag, dev_raw.shape).copy()

    for shm in (shm_nonconv_flag, shm_bias_r, shm_f, shm_ll_r, shm_y):
        shm.close()
        try:
            shm.unlink()
        except Exception as exc:
            print(f"\nUnlink shared-memory issue: {exc}")

    indices = tuple(z for z in zip(*links_to_check))
    dev_raw[indices] = 2 * model_f.ll
    dev_raw[indices] -= 2 * ll_r[indices]

    return dev_raw, bias_r, nonconv_flag


def _init_Q(y, f):
    n, m = f.shape
    e, u = linalg.eigh(f.dot(f.T))
    temp = u.T.dot(y)
    c = (temp ** 2).sum(axis=1)

    def fun(x):
        return (c / (1 + x * e) ** 2).sum() - 1.2 * n * y.shape[1]

    if fun(0) > 0:
        q_val = optimize.newton(fun, 1)
    else:
        q_val = 0.0001

    return q_val * np.eye(m)


def _learn_reduced_model(i, j, y, F, R, lambda_f, em_state, config):
    n_eigenmodes = config.latent.n_eigenmodes
    n_orients = config.latent.n_orients
    eff_eigenmodes = n_eigenmodes * n_orients
    
    target = _expand_roi_indices_as_tup(j, eff_eigenmodes)
    source = _expand_roi_indices_as_tup(i, eff_eigenmodes)
    print(f'target: {target}')
    print(f'source {source}')
    link = '->'.join(map(lambda x: ','.join(map(str, x)), (source, target)))
    
    model_r = NeuraLVAR.from_config(config)
    em_state, _ = model_r.fit(y, F, R, lambda_f, em_state, 
                                            restriction=link)
    bias = model_r.compute_bias(em_state)
    ll = model_r.ll
    nonconv_flag = em_state.em_iter == config.optimizer.max_iter
    return ll, bias, nonconv_flag


def _learn_reduced_model_parallel(link_index, info_y, info_f, info_bias_r, 
                                  info_ll_r, info_nonconv_flag, R, lambda_f, 
                                  em_state, config):
    
    # each process will mutate this after this point
    em_state = copy.deepcopy(em_state)

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







