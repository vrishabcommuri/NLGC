from functools import cached_property
import copy
import pickle
from functools import reduce
from nlgc.opt import NeuraLVAR, NeuraLVARCV
from nlgc.stat import fdr_control
from nlgc.bias_utils import debias_deviances
from nlgc.parallel_gc import batched_test_links, multiprocess_test_links
from nlgc.utils.restriction import roi_to_link_restriction
from nlgc.bias_utils import compute_bias
from nlgc.utils.screen import sparsity_screen, wald_screen
from nlgc.config import ModelMultiprocessConfig
from nlgc.test.profile import pretty_print_elapsed
import time

class NLGC:
    """NLGC object

    Provides a an object including captured connectivity map via NLGC and its
    related parameters.

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
            return reduce(lambda x, y: x + y, debiased_deviances) / \
                self.n_segments
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

        return fdr_control(self.avg_debiased_dev, self.p * (eff_eigenmodes**2), 
                           alpha)


    def pickle_as(self, filename):
        """saving the object as a pickle

        Parameters
        ----------
        filename : str
            file name (including directory address)
            """
        if filename.endswith('.pkl') or filename.endswith('.pickled') or \
            filename.endswith('.pickle'):
            pass
        else:
            filename += '.pkl'

        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)


def gc_extraction(y, F, R, ROIs, em_state, config):
    eff_eigenmodes = config.latent.n_eigenmodes * config.latent.n_orients
    
    lambda_range = config.sparsity.lambda_range

    if lambda_range is None:
        raise ValueError("lambda range must be a float or list of floats")
    
    if config.numerical.verbose:
        start = time.time()
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
        end = time.time()
        elapsed = end - start

        print(f"finished full model fit in {em_state.em_iter} EM iterations.")
        pretty_print_elapsed(elapsed)

    if config.debug.plotlevel > 0:
        print("plotting transition matrices")
        from nlgc.test.viz import plot_transition_blurred
        import matplotlib.pyplot as plt
        plot_transition_blurred(model_f._ravel_a(model_f._parameters[0]), 
                                em_state.N_sources_upper, config.latent.order)
        plt.show()

        plot_transition_blurred(model_f._ravel_a(model_f._parameters[0]) \
                                > 0.0001, 
                                em_state.N_sources_upper, config.latent.order)
        plt.show()
        
    if config.numerical.verbose:
        print("link screening")

    sparsity, ROIs = sparsity_screen(em_state, smoother_result, ROIs, config)
    links_to_check = roi_to_link_restriction(ROIs, sparsity, 
                                             eff_eigenmodes, config)    
    links_to_check = wald_screen(links_to_check, em_state, smoother_result, 
                                 config)

    if config.numerical.verbose:
        print(f"Checking {len(links_to_check)} links...")
    
    if isinstance(config.parallel, ModelMultiprocessConfig):
        dev_raw, bias_r, bias_f, nonconv_flag = multiprocess_test_links(links_to_check, 
                                                                y, F, R, 
                                                                lambda_,
                                                                em_state, 
                                                                config)
    else:
        dev_raw, bias_r, bias_f, nonconv_flag = batched_test_links(links_to_check, 
                                                           y, F, R, lambda_,
                                                           em_state, config)
        
    if config.numerical.verbose:
        print("GC testing finished. Total fit+link testing time:")
        end = time.time()
        pretty_print_elapsed(end-start)

    return dev_raw, bias_r, bias_f, model_f, nonconv_flag









