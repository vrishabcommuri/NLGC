from functools import cached_property
import copy
import dataclasses
import pickle
import warnings
import numpy as np
from functools import reduce
from nlgc.opt import NeuraLVAR, NeuraLVARCV
from nlgc.opt.em import EMState
from nlgc.stat import fdr_control
from nlgc.bias_utils import debias_deviances
from nlgc.parallel_gc import (batched_test_links, 
                              multiprocess_test_links,
                              multiprocess_test_links_ggc)
from nlgc.utils.restriction import roi_to_link_restriction
from nlgc.utils.screen import sparsity_screen, wald_screen
from nlgc.config import ModelMultiprocessConfig
from nlgc.test.profile import pretty_print_elapsed
import time

# Bump on any breaking change to the 'meta'/'stats'/'models' keys and add a
# matching _from_dict_vN. Additive keys need no bump; the loaders use .get().
FORMAT_VERSION = 2


def _encode_model_entry(entry):
    """Tag one _model_f entry so _decode_model_entry can rebuild it.

    _model_f alternates, per segment, the fitted NeuraLVAR(CV) and its
    {lambda: EMState} dict of last-split CV states.
    """
    if entry is None:
        return None
    if isinstance(entry, NeuraLVAR):
        return {'kind': 'neuralvar', **entry.to_dict()}
    if isinstance(entry, EMState):
        # asarray because these may be jax
        return {'kind': 'em_state',
                **{f.name: None if getattr(entry, f.name) is None
                   else np.asarray(getattr(entry, f.name))
                   for f in dataclasses.fields(entry)}}
    if isinstance(entry, dict):
        # keys as a list: _flatten would stringify float dict keys
        return {'kind': 'cv_em_states',
                'lambdas': [float(k) for k in entry],
                'states': [_encode_model_entry(v) for v in entry.values()]}
    raise TypeError(f"Don't know how to save a {type(entry).__name__!r} "
                    f"entry of NLGC._model_f")


def _decode_model_entry(d):
    """Inverse of _encode_model_entry."""
    if d is None:
        return None
    kind = d.get('kind')
    if kind == 'neuralvar':
        return NeuraLVAR.from_dict(d)
    if kind == 'em_state':
        state = {f.name: d.get(f.name) for f in dataclasses.fields(EMState)}
        # static field, so jax needs a hashable int, not a numpy scalar
        if state['N_sources_upper'] is not None:
            state['N_sources_upper'] = int(state['N_sources_upper'])
        if state['em_iter'] is not None:
            state['em_iter'] = int(state['em_iter'])
        return EMState(**state)
    if kind == 'cv_em_states':
        return {lam: _decode_model_entry(state)
                for lam, state in zip(d['lambdas'], d['states'])}
    raise ValueError(f"Unrecognized _model_f entry kind: {kind!r}")


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


    def to_dict(self):
        """Plain-data state, versioned, for nlgc.io.save_model.

        Listed explicitly, not off __dict__, which would capture the cached
        avg_debiased_dev. forward_orig goes to a .fif sidecar.
        """
        return {
            'format_version': FORMAT_VERSION,
            'meta': {
                'subject': self.subject,
                'nx': self.nx,
                'ny': self.ny,
                't': self.t,
                'p': self.p,
                'n_eigenmodes': self.n_eigenmodes,
                'n_orients': self.n_orients,
                'n_segments': self.n_segments,
            },
            'stats': {
                'd_raw': self.d_raw,
                'bias_f': self.bias_f,
                'bias_r': self.bias_r,
                'nonconv_flag': self._nonconv_flag,
            },
            # every entry, in order, so _model_f[i] indexes the same object
            'models': [_encode_model_entry(m) for m in self._model_f],
            'source_space': {
                'labels': self._labels,
                'label_vertidx': self._label_vertidx,
                'eig_src_weights': self.eig_src_weights,
                'whitener': self.whitener,
            },
            'debug': self._debug,
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild from to_dict output, dispatching on format_version.

        A new version adds a branch beside the existing one; it must not edit
        one, or files already written stop loading. Split the bodies back out
        into methods if this grows past two or three of them.
        """
        version = d.get('format_version')
        if version == 1:
            # v1's to_dict filtered _model_f on isinstance(m, NLGC), which
            # nothing in it ever is, so v1 files carry no models at all
            models = [NeuraLVAR.from_dict(m) for m in d.get('models', [])]
            if not models:
                warnings.warn(
                    "This file was written with format_version 1, which did "
                    "not save _model_f; it loads as []. Re-save from the "
                    "pickle or refit to recover it.")
            return cls._from_dict_common(d, models, debug=None)
        if version == 2:
            models = [_decode_model_entry(m) for m in d.get('models', [])]
            return cls._from_dict_common(d, models, debug=d.get('debug'))

        raise ValueError(
            f"Unrecognized NLGC format_version: {version!r} (this build of "
            f"nlgc writes and understands version {FORMAT_VERSION})")

    @classmethod
    def _from_dict_common(cls, d, models, debug):
        meta = d.get('meta', {})
        stats = d.get('stats', {})
        src = d.get('source_space', {})

        # JSON hands the vol path's int keys back as strings
        label_vertidx = src.get('label_vertidx')
        if isinstance(label_vertidx, dict):
            label_vertidx = {int(k): v for k, v in label_vertidx.items()}

        # keyword form: 18 positional params is too many to reshuffle safely
        return cls(
            subject=meta.get('subject'),
            nx=meta.get('nx'),
            ny=meta.get('ny'),
            t=meta.get('t'),
            p=meta.get('p'),
            n_eigenmodes=meta.get('n_eigenmodes'),
            n_orients=meta.get('n_orients'),
            n_segments=meta.get('n_segments'),
            d_raw=stats.get('d_raw'),
            bias_f=stats.get('bias_f'),
            bias_r=stats.get('bias_r'),
            model_f=models,
            nonconv_flag=stats.get('nonconv_flag'),
            label_names=src.get('labels'),
            label_vertidx=label_vertidx,
            forward_orig=None,  # reattached by nlgc.io.load_model
            whitener=src.get('whitener'),
            eig_src_weights=src.get('eig_src_weights'),
            debug=debug,
        )

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
        em_state, smoother_result, cv_em_states = model_f.fit(y, F, R, 
                                                copy.deepcopy(em_state))
    else:
        model_f = NeuraLVAR.from_config(config)
        lambda_ = lambda_range[0]
        em_state, smoother_result, cv_em_states = model_f.fit(y, F, R, lambda_, 
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
        plt.close()

        plot_transition_blurred(model_f._ravel_a(model_f._parameters[0]) \
                                > 0.0001, 
                                em_state.N_sources_upper, config.latent.order)
        plt.show()
        plt.close()
        
    if config.numerical.verbose:
        print("link screening")

    sparsity, ROIs = sparsity_screen(em_state, smoother_result, ROIs, config)
    links_to_check = roi_to_link_restriction(ROIs, sparsity, 
                                             eff_eigenmodes, config)    
    links_to_check = wald_screen(links_to_check, em_state, smoother_result, 
                                 config)
    
    if config.numerical.verbose:
        print(f"Checking {len(links_to_check)} links...")
    
    if config.gctest.gc_test_method == 'likelihood ratio':
        if isinstance(config.parallel, ModelMultiprocessConfig):
            dev_raw, bias_r, bias_f, nonconv_flag = multiprocess_test_links(
                                                    links_to_check, y, F, R, 
                                                    lambda_, em_state, config)
        else:
            dev_raw, bias_r, bias_f, nonconv_flag = batched_test_links(
                                                    links_to_check, y, F, R, 
                                                    lambda_, em_state, config)
    elif config.gctest.gc_test_method == 'wald':
        dev_raw, bias_r, bias_f, nonconv_flag = multiprocess_test_links_ggc(
                                                    links_to_check, y, F, R, 
                                                    lambda_, em_state, config)
    else:
        raise Exception(f"{config.gctest.gc_test_method=} not supported")

    if config.numerical.verbose:
        print("GC testing finished. Total fit+link testing time:")
        end = time.time()
        pretty_print_elapsed(end-start)

    return dev_raw, bias_r, bias_f, model_f, nonconv_flag, cv_em_states









