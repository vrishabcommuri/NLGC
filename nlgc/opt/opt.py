from itertools import product
from multiprocessing import shared_memory, current_process

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import warnings
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

import mne
from mne.utils import logger

from .e_step import sskf, sskfcv, align_cast, sskf_prediction
from .m_step import (calculate_ss, solve_for_a, solve_for_q, compute_ll,
                     compute_cross_ll, compute_Q)

# filename = os.path.realpath(os.path.join(__file__, '..', '..', "debug.log"))
# logging.basicConfig(filename=filename, level=logging.DEBUG)


class NeuraLVAR:
    """Neural Latent Vector Auto-Regressive model

    Provides a vector auto-regressive model for the unobserved (latent) source activities
    that gives rise to m/eeg data.

    Parameters
    ----------
    order : int
    copy : bool, default=True
    standardize : bool, default=False
    normalize : bool, default=False
    use_lapack : bool, default=True
    """
    _dims = None
    _parameters = None
    _lls = None
    ll = None
    lambda_ = None
    _zeroed_index = None
    restriction = None

    def __init__(self, order, self_history=None, n_eigenmodes=None, copy=True, standardize=False, normalize=False,
            use_lapack=True):
        if standardize is not False and normalize is not False:
            raise ValueError(f"both standardize={standardize} and normalize={normalize} cannot be specified")
        elif standardize:
            _preprocessing = preprocessing.StandardScaler(copy)
        elif normalize is not False:
            if isinstance(normalize, bool):
                normalize = 'l2'
            _preprocessing = preprocessing.Normalizer(normalize, copy)
        else:
            _preprocessing = None
        self._preprocessing = _preprocessing
        self._copy = copy
        self._order = order
        self._self_histoty = order if self_history is None else self_history
        self._use_lapack = use_lapack
        self._n_eigenmodes = 1 if n_eigenmodes is None else n_eigenmodes

    def _fit(self, y, f, r, lambda2=None, lb=None, la=None, max_iter=500, max_cyclic_iter=3, a_init=None, q_init=None,
            rel_tol=0.01, xs=None, alpha=0.0, beta=0.0, fixed_a=False, fixed_q=False, verbose=False):
        """Internal function that fits the model from given data

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda2 : float, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        xs : tuple of two ndarrays of shape (n_samples, n_sources), default=None
        alpha: float, default = 0.5
        beta : float, default = 1

        Returns
        -------
        a : ndarray of shape (order, n_sources, n_sources)
        q : ndarray of shape (n_sources, n_sources),
        lls : list
            list containing the likelihood values along the training path
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        xs : tuple of two ndarrays of shape (n_samples, n_sources)
            Used for mostly reusing the allocated memories
        x_ : ndarray of shape (n_samples, n_sources)

        Notes
        -----
        To learn restricted model for i --> j, ([j] * p, list(range(i, m*p, m))) are set
        as zeroed index.
        non-zero alpha, beta values imposes Gamma(alpha*n/2 - 1, beta*n) prior on q's.
        This equivalent to alpha*n - 2 additional observations that sum to beta*n.
        """
        warnings.filterwarnings('always')
        y, a_, a_upper, f_, q_, q_upper, non_zero_indices, r, xs, m, n, p, use_lapack = \
            self._prep_for_sskf(y, a_init, f, q_init, r, xs)
        p1 = self._self_histoty

        if self.restriction is not None:
            i_s, j_s = re.split(r'->', self.restriction)
            i_s = [int(i) for i in re.split(r',', i_s)]
            j_s = [int(j) for j in re.split(r',', j_s)]
            # check for i, j's proper range
            if any(i >= m for i in i_s) or any(j >= m for j in j_s):
                raise ValueError(f"restriction {self.restriction}: i or j needs to be in range of neural sources, {m}")
            x_index = []
            y_index = []
            for i, j in product(i_s, j_s):
                x_index.extend([j] * p)
                y_index.extend(list(range(i, m * p, m)))
            zeroed_index = (x_index, y_index)
        else:
            zeroed_index = None

        lls = []
        ll_s = []
        Qvals = []
        source_fits = []
        for i in range(max_iter):
            a_[:m] = a_upper
            q_[non_zero_indices] = q_upper[non_zero_indices]

            x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=xs, use_lapack=use_lapack)
            ll_s.append(ll_)
            ll = compute_ll(y, x_, s, s_, s_hat, a_upper, f, q_upper, r, m, n, p)
            lls.append(ll)
            Qvals.append(compute_Q(y, x_, s_, b, a_upper, f, q_upper, r, m, p))
            source_fits.append(compute_cross_ll(x_, a_upper, q_upper, m, p))
            # stopping cond
            if i > 0:
                rel_change = (lls[i - 1] - lls[i]) / lls[i - 1]
                if np.abs(rel_change) < rel_tol and rel_q_change < rel_tol:
                    break

            s1, s2, s3, t = calculate_ss(x_, s_, b, m, p)
            beta = 2 * beta / t
            alpha = 2 * (alpha + 1) / t if alpha else alpha

            for _ in range(max_cyclic_iter):
                if not fixed_a:
                    if lb is not None and la is not None:
                        a_upper, changes = solve_for_a_indepdiag(q_upper, s1, s2, a_upper, p1, lb=lb, la=la, max_iter=max_iter,
                                                   tol=min(1e-4, rel_tol), zeroed_index=zeroed_index,
                                                   update_only_target=False, n_eigenmodes=self._n_eigenmodes)
                    else:
                        a_upper, changes = solve_for_a(q_upper, s1, s2, a_upper, p1, lambda2=lambda2, max_iter=max_iter,
                                                   tol=min(1e-4, rel_tol), zeroed_index=zeroed_index,
                                                   update_only_target=False, n_eigenmodes=self._n_eigenmodes)
                if not fixed_q:
                    q_upper, rel_q_change = solve_for_q(q_upper, s3, s1, s2, a_upper, lambda2=lambda2, alpha=alpha,
                                                        beta=beta)
                if rel_q_change < rel_tol:
                    break

                if q_upper.min() < 0:
                    warnings.warn(f'Q possibly contains negative value {q_upper.min()}', RuntimeWarning)

        a = self._unravel_a(a_upper)
        return a, q_upper, (lls, ll_s, Qvals, source_fits), f, r, zeroed_index, xs, x_

    def compute_ll(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        ll = compute_ll(y, x_, s, s_, s_hat, a_upper, f, q_upper, r, m, n, p)
        return ll

    def compute_ll_(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        return ll_

    def compute_Q(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        ll = compute_Q(y, x_, s_, b, a_upper, f, q_upper, r, m, p)
        return ll

    def compute_logsum_q(self, y, max_iter, max_cyclic_iter, rel_tol, alpha, beta, args=None):
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        _, q_upper, *rest = self._fit(y, f, r, None, max_iter, max_cyclic_iter, a.copy(), q.copy(), rel_tol, None,
                                      alpha, beta, fixed_a=True)
        return - np.log(np.diag(q_upper)).sum()

    def compute_norm_one(self, a_):
        return np.sum(np.absolute(a_))

    def compute_crossvalidation_metric(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        # x_, s, s_, b, s_hat = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        # ll = compute_cross_ll(x_, a_upper, q_upper, m, p)
        # return ll
        return sskfcv(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)

    def get_prediction(self, y, args=None):
        """Returns log(p(y|args=(a, f, q, r))).

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        args : tuple of ndarrays, default=None
            Expect a tuple with the model parameters: (a, f, q, r).
            if None, self._parameters is used.

        Returns
        -------
        log_likelihood : float
            Returns log(p(y|(a, f, q, r))).
        """
        if args is None:
            args = self._parameters
        a, f, q, r, *rest = args
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        return sskf_prediction(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)

    def compute_bias(self, y):
        from .._bias_utils import sample_path_bias
        a, f, q, r, *rest = self._parameters
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        bias = sample_path_bias(q_upper, a_upper, x_[:, :m], self._zeroed_index, self._n_eigenmodes)
        return bias

    def compute_bias_idx(self, y, source):
        from .._bias_utils import bias_by_idx
        a, f, q, r, *rest = self._parameters
        y, a_, a_upper, f_, q_, q_upper, _, r, (_x, x_), m, n, p, use_lapack = self._prep_for_sskf(y, a, f, q, r)
        x_, s, s_, b, s_hat, ll_ = sskf(y, a_, f_, q_, r, xs=(_x, x_), use_lapack=use_lapack)
        if isinstance(source, int):
            source = tuple(source)
        bias = sum([bias_by_idx(i, q_upper, a_, x_, s_, b, m, p, self._zeroed_index) for i in source])
        return bias

    def fit(self, y, f, r, lambda2=None, lb=None, la=None, max_iter=500, max_cyclic_iter=3, a_init=None, q_init=None, rel_tol=0.0001,
            restriction=None, alpha=0.0, beta=0.0, use_es=None, verbose=False):
        """Fits the model from given m/eeg data, forward gain and noise covariance

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda2 : float, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        restriction : regular expression like 'i->j' or 'i1,i2->j1,j2', default = None
            i and j should be integers.
        alpha: float, default = 0.5
        beta : float, default = 1

        Notes
        -----
        To learn restricted model for i --> j, pass ([j] * p, list(range(i, m*p, m)))
        as zeroed index.
        non-zero alpha, beta values imposes Gamma(alpha*n/2 - 1, beta*n) prior on q's.
        This equivalent to alpha*n - 2 additional observations that sum to beta*n.
        """
        if (restriction is None or re.search('->', restriction)) is False:
            raise ValueError(f"restriction:{restriction} should be None or should have format 'i->j'!")
        self.restriction = restriction
        a, q_upper, lls, f, r, zeroed_index, _, x_ = self._fit(y, f, r, lambda2=lambda2, lb=None, la=None, max_iter=max_iter,
                                                               max_cyclic_iter=max_cyclic_iter, a_init=a_init,
                                                               q_init=q_init, rel_tol=rel_tol, alpha=alpha, beta=beta, verbose=verbose)
        self._parameters = (a, f, q_upper, r, x_)
        self._zeroed_index = zeroed_index
        self._lls = lls
        self.ll = lls[0][-1]
        self.lambda_ = lambda2

    def information_criterion(self, type='akike'):
        if type not in ['akike', 'bayesian']:
            raise ValueError("type needs to be either 'akike' or 'bayesian'")
        df = (abs(self._parameters[0]) > 1e-15).sum()
        t = self._parameters[4].shape[0]
        mul = 2 if type == 'akike' else np.log(t)
        return (mul * df - 2*self.ll) / t

    @staticmethod
    def _ravel_a(a):
        p, m, m_ = a.shape
        assert m == m_
        return np.reshape(np.swapaxes(a, 0, 1), (m, m * p))

    @staticmethod
    def _unravel_a(a):
        m, mp = a.shape
        p = mp // m
        return np.swapaxes(np.reshape(a, (m, p, m)), 0, 1)

    def _prep_for_sskf(self, y, a, f, q, r, xs=None):
        """Prepares (mostly memory allocation and type-casting) arrays for sskf()

        Parameters
        ----------
        y : ndarray of shape (n_channels, n_samples)
        a : ndarray of shape (order, n_sources, n_sources), None
        f : ndarray of shape (n_channels, n_sources)
        q : ndarray of shape (n_sources, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        xs : tuple of two ndarrays of shape (n_sources*order, n_samples), default=None

        Returns
        -------
        y : ndarray of shape (n_samples, n_channels)
        a_ : ndarray of shape (n_sources*order, n_sources*order)
        a_upper : ndarray of shape (n_sources, n_sources*order)
        f_ : ndarray of shape (n_channels, n_channels*order)
        q_ : ndarray of shape (n_sources*order, n_sources*order)
        q_upper : ndarray of shape (n_sources, n_sources)
        non_zero_indices : tuple of ndarrays of shape (n_sources, )
            (non-zero) diag indices of (q_) q_upper
        r : ndarray of shape (n_samples, samples)
        xs : tuple of two ndarrays of shape (n_sources*order, n_samples)
        m : int
            n_sources
        n : int
            n_channles
        p : int
            order
        use_lapack: bool
        """
        n, m = f.shape
        _, t = y.shape
        p = self._order
        use_lapack = self._use_lapack

        if y.flags['F_CONTIGUOUS']:
            y = y.T
        else:
            y = y.copy(order='F').T  # to make y contiguous in 'F'

        assert q.shape[0] == m
        q_upper = q

        a_upper = np.zeros((m, m * p), dtype=np.float64)
        a_lower = np.hstack((np.eye(m * (p - 1)), np.zeros((m * (p - 1), m))))
        a_ = np.vstack((a_upper, a_lower))
        if a is not None:
            a_upper[:] = self._ravel_a(a)
            a_[:m] = a_upper

        q_ = np.zeros((m * p, m * p))
        non_zero_indices = np.diag_indices_from(q_upper)
        q_[non_zero_indices] = q_upper[non_zero_indices]

        f_ = np.hstack((f, np.zeros((n, m * (p - 1)))))

        if xs is None:
            _x = np.empty((t, m * p), dtype=np.float64)
            x_ = np.empty_like(_x)
            xs = (_x, x_)
        else:
            _x, x_ = xs
            assert _x.shape[0] == y.shape[0]
            assert _x.flags['C_CONTIGUOUS']
            assert x_.flags['C_CONTIGUOUS']
        return y, a_, a_upper, f_, q_, q_upper, non_zero_indices, r, xs, m, n, p, use_lapack

    def _plot_ll_curve(self):
        fig, ax = plt.subplots()
        ax.plot(self._lls[0])
        return fig, ax


class NeuraLVARCV(NeuraLVAR):
    """Neural Latent Vector Auto-Regressive model (supports cross-validation)

    Provides a vector auto-regressive model for the unobserved (latent) source activities
    that gives rise to m/eeg data.

    Parameters
    ----------
    order : int
        VAR order
    max_n_mus : int
        keep it 5
    cv : int | sklearn.model_selection split objects
    n_jobs : int
    copy : bool, default=True
    standardize : bool, default=False
    normalize : bool, default=False
    use_lapack : bool, default=True
    """
    cv_lambdas = None
    mse_path = None
    es_path = None

    def __init__(self, order, self_history, n_eigenmodes, max_n_mus, cv, n_jobs, copy=True, standardize=False,
            normalize=False, use_lapack=True):
        self.max_n_mus = max_n_mus
        self.cv = cv
        self.n_jobs = n_jobs
        NeuraLVAR.__init__(self, order, self_history, n_eigenmodes, copy, standardize, normalize, use_lapack)

    def _cvfit(self, split, info_y, info_f, info_r, info_cv, info_pred, splits, lambda_range, max_iter=500,
            max_cyclic_iter=3, a_init=None, q_init=None, rel_tol=1e-5, alpha=0.0, beta=0.0, verbose=False):
        
        mne.set_log_level(verbose)

        """Utility function to be used by self.fit()

        Parameters
        ----------
        split :
        info_y :
        info_f :
        info_r :
        info_cv :
        splits :
        lambda_range :
        max_iter :
        max_cyclic_iter :
        a_init :
        q_init :
        rel_tol :

        Returns
        -------
        val

        """
        logger.info(f"{current_process().name} working on {split}th split")
        logger.debug(f"{current_process().name} working on {split}th split")
        try:
            y, shm_y = link_share_memory(info_y)
            f, shm_f = link_share_memory(info_f)
            r, shm_r = link_share_memory(info_r)
            cv, shm_c = link_share_memory(info_cv)
            pred, shm_pred = link_share_memory(info_pred)
        except BaseException as e:
            logger.error("Could not link to memory")
            raise e

        logger.debug(f"{current_process().name} successfully read the shared memory")
        train, test = splits[split]
        y_train, y_test = y[:, train], y[:, test]
        xs = None
        logger.debug(f"{current_process().name} successfully split the data")
        for i, lambda2 in enumerate(lambda_range * np.sqrt(y.shape[-1])):
            lambda2 = lambda2 / np.sqrt(y_train.shape[-1])
            logger.info(f"{current_process().name} {split} doing {lambda2}")
            logger.debug(f"{current_process().name} {split} doing {lambda2}")
            if i > 0:
                a_init = a_.copy()
                # q_init = q_upper.copy() * lambda_range[i-1] / lambda_range[i]
            a_, q_upper, lls, _, _, _, xs, _ = \
                self._fit(y_train, f, r, lambda2=lambda2, max_iter=max_iter,
                          max_cyclic_iter=max_cyclic_iter,
                          a_init=a_init, q_init=q_init.copy(), rel_tol=rel_tol, xs=xs, alpha=alpha, beta=beta)
            # # different criteria for cross-validation
            cv[0, split, i] = self.compute_ll_(y_test, (a_, f, q_upper, r))
            cv[1, split, i] = lambda2 * self.compute_norm_one(a_)

            # cv[2, split, i] = self.compute_ll(y_test, (a_, f, q_upper, r))
            # cv[3, split, i] = self.compute_crossvalidation_metric(y_test, (a_, f, q_upper, r))
            # cv[4, split, i] = self.compute_Q(y_test, (a_, f, q_upper, r))
            # cv[5, split, i] = self.compute_logsum_q(y_test, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter,
            #                                  rel_tol=rel_tol, alpha=alpha, beta=beta, args=(a_, f, q_upper, r))
            pred[split, i][:] = self.get_prediction(y, (a_, f, q_upper, r)).T

        for shm in (shm_y, shm_f, shm_r, shm_c):
            shm.close()
        return None

    def fit(self, y, f, r, lambda_range=None, max_iter=500, max_cyclic_iter=3, a_init=None, q_init=None,
            rel_tol=1e-5, restriction=None, alpha=0.0, beta=0.0, use_es=True, verbose=False):
        logger.info(f"fit max iter = {max_iter}")
        """Fits the model from given m/eeg data, forward gain and noise covariance

        y : ndarray of shape (n_channels, n_samples)
        f : ndarray of shape (n_channels, n_sources)
        r : ndarray of shape (n_channels, n_channels)
        lambda_range : ndarray of floats, default=None
        max_iter : int, default=100
        max_cyclic_iter : int, default=2
        a_init : ndarray of shape (order, n_sources, n_sources), default=None
        q_init : ndarray of shape (n_sources, n_sources), default=None
        rel_tol : float, default=0.0001
        restriction : regular expression like 'i->j', default = None
            i and j should be integers.
        alpha: float, default = 0.5
        beta : float, default = 1

        Notes
        -----
        non-zero alpha, beta values imposes Gamma(alpha*n/2 - 1, beta*n) prior on q's.
        This equivalent to alpha*n - 2 additional observations that sum to beta*n.
        """
        if (restriction is None or re.search('->', restriction)) is False:
            raise ValueError(f"restriction:{restriction} should be None or should have format 'i->j'!")
        self.restriction = restriction

        y, f = align_cast((y, f), self._use_lapack)  # to make y, f contiguous in 'F'

        if lambda_range is None or lambda_range == 'auto':
            raise NotImplementedError("Try specifying a pre-determined range")
        else:
            if not isinstance(lambda_range, np.ndarray):
                lambda_range = np.asarray(lambda_range)

        # do cvsplits
        if isinstance(self.cv, int):
            # kf = TimeSeriesSplit(n_splits=2*self.cv)
            # cvsplits = [split for split in kf.split(y.T)][-self.cv:]
            kf = TimeSeriesSplit(n_splits=self.cv)
            cvsplits = [split for split in kf.split(y.T)]
        else:
            kf = self.cv
            cvsplits = [split for split in kf.split(y.T)]

        cv_mat = np.zeros((2, len(cvsplits), len(lambda_range)), dtype=y.dtype)
        pred_mat = np.zeros((len(cvsplits), len(lambda_range)) + y.shape, dtype=y.dtype)

        # Use parallel processing
        # A, b, mu_range, cv_mat needs to shared across processes
        shared_y, info_y, shm_y = create_shared_mem(y)
        shared_f, info_f, shm_f = create_shared_mem(f)
        shared_r, info_r, shm_r = create_shared_mem(r)
        shared_cv_mat, info_cv, shm_c = create_shared_mem(cv_mat)
        shared_pred_mat, info_pred, shm_p = create_shared_mem(pred_mat)
        initargs = (info_y, info_f, info_r, info_cv, info_pred, cvsplits, lambda_range,
                    max_iter, max_cyclic_iter, a_init, q_init, rel_tol, alpha, beta, verbose)

        logger.info('Starting cross-validation')
        # Serial implementation
        # out = [self._cvfit(i, *initargs) for i in range(len(cvsplits))]
        # Parallel implementation
        Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(self._cvfit)(i, *initargs) for i in range(len(cvsplits)))
        logger.info('Done cross-validation')

        self.cv_lambdas = lambda_range
        cv_mat[:] = np.reshape(shared_cv_mat, cv_mat.shape)
        pred_mat[:] = np.reshape(shared_pred_mat, pred_mat.shape)
        self.mse_path = cv_mat
        self.es_path = compute_es_criterion(pred_mat)

        for shm in (shm_y, shm_f, shm_r, shm_c, shm_p):
            shm.close()
            try:
                shm.unlink()
            except:
                logger.info(f"Unlink shared-memory issue!")

        # Find best mu
        # If Estimation stability criterion is used we need cv_mat[0] and pred_mat
        # else we just use $\lambda * ||A||_1$ as the metric.
        if use_es:
            index = self.mse_path[0].mean(axis=0).argmax()
            try:
                best_lambda = lambda_range[np.nanargmin(self.es_path[:index])]
            except ValueError:
                best_lambda = lambda_range[index]
            logger.info(f'\nbest_regularizing parameter: {best_lambda} using es')
        else:
            index = self.mse_path[1].mean(axis=0).argmax()
            best_lambda = lambda_range[index]
            logger.info(f'\nbest_regularizing parameter: {best_lambda}')

        a, q_upper, lls, f, r, zeroed_index, _, x_ = self._fit(y, f, r, lambda2=best_lambda, max_iter=max_iter,
                                                               max_cyclic_iter=max_cyclic_iter, a_init=a_init,
                                                               q_init=q_init, rel_tol=rel_tol, alpha=alpha, beta=beta, verbose=verbose)
        self._parameters = (a, f, q_upper, r, x_)
        self._zeroed_index = zeroed_index
        self._lls = lls
        self.ll = lls[0][-1]
        self.lambda_ = best_lambda

        _, t = y.shape
        df = (abs(a) > 1e-15).sum()
        self.aic = (2*df - 2*self.ll) / t
        self.bic = (np.log(t)*df - 2*self.ll) / t

        return self


def create_shared_mem(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    info = (arr.shape, arr.dtype, shm.name)
    shared_arr = np.ndarray((arr.size,), dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr.ravel()[:]
    return shared_arr, info, shm


def link_share_memory(info):
    shape, dtype, name = info
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr, shm


def initialize_q(y, f, r):
    q = np.eye(f.shape[1], dtype=np.float64)
    return q


def compute_es_criterion(pred):
    cv_split_repeats = np.arange(pred.shape[0]) + 1
    cv_split_repeats[:] = 1
    shape = pred.shape[:-2] + (-1,)
    pred.shape = shape
    es = np.empty(pred.shape[1], pred.dtype)
    for j in range(pred.shape[1]):
        this_pred = pred[:, j, :]
        this_pred_mean = (this_pred * cv_split_repeats[:, None]).mean(axis=0)
        fluctuation = (this_pred - this_pred_mean[None, :]) * np.sqrt(cv_split_repeats[:, None])
        es[j] = (fluctuation ** 2).sum() / (this_pred_mean ** 2).sum()
    return es 
