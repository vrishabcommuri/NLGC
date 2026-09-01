from multiprocessing import shared_memory, current_process
import numpy as np
import warnings
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import cpu_count
from nlgc.opt.em import solve_params
from nlgc.opt.kalman.filter import (forward_filter_jax, rts_smoother_jax, 
                                    measurement_smoother_jax)
import copy
from kneed import KneeLocator


class NeuraLVAR:
    """Neural Latent Vector Auto-Regressive model

    Provides a vector auto-regressive model for the unobserved (latent) source
    activities that gives rise to m/eeg data.
    """
    _dims = None
    _parameters = None
    _lls = None
    ll = None
    lambda_ = None

    def __init__(self, order, self_history=None, n_eigenmodes=None, 
                 n_orients=None, copy=True, standardize=False, normalize=False, 
                 use_lapack=True, config=None):
                 
        if standardize is not False and normalize is not False:
            raise ValueError(f"both {standardize=} and {normalize=} cannot be specified")
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
        self._self_history = order if self_history is None else self_history
        self._use_lapack = use_lapack
        self._n_eigenmodes = 1 if n_eigenmodes is None else n_eigenmodes
        self._n_orients = 1 if n_orients is None else n_orients
        self.config = config


    @classmethod
    def from_config(cls, config):
        return cls(
            order = config.latent.order,
            self_history = config.sparsity.self_history,
            n_eigenmodes = config.latent.n_eigenmodes,
            n_orients = config.latent.n_orients,
            copy = True,
            standardize = False,
            normalize = False,
            use_lapack = config.numerical.use_lapack,
            config = config,
        )
    

    def _fit(self, y, F, R, lambda_, em_state):
        warnings.filterwarnings('always')

        em_state, smoother_result = solve_params(
                y, F, R,
                em_state=em_state,
                config=self.config,
                lambda_=lambda_,
        )

        return em_state, smoother_result

    def compute_norm_one(self, a):
        return np.sum(np.absolute(a))

    @staticmethod
    def GCV(D, N, df):
        """
        Generalized CV metric
        see 10.1016/j.automatica.2017.12.054
        """
        return D/(N * (1 - df/N)**2)

    def compute_two_one_norm(self, a):
        print(f'a_ shape is {a.shape}')
        p = a.shape[0]
        m = a.shape[1]
        N = m // 3
        B = a.reshape(N, 3, p, N, 3)
        return (np.sqrt(np.sum(B * B, axis=(1, 4), keepdims=True)).sum())
    

    def fit(self, y, F, R, lambda_, em_state):        
        em_state, smoother_result = self._fit(y, F, R, lambda_, em_state)
        
        m = em_state.N_sources_upper
        self._parameters = (
            self._unravel_a(em_state.A[:m]), 
            F[:, :m], 
            em_state.Q[:em_state.N_sources_upper, :em_state.N_sources_upper], 
            R, 
            smoother_result.smoothed_state
        )
        self.ll = -smoother_result.negative_log_likelihood
        self.lambda_ = lambda_

        return em_state, smoother_result


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
    

class NeuraLVARCV(NeuraLVAR):
    """Neural Latent Vector Auto-Regressive model (supports cross-validation)

    Provides a vector auto-regressive model for the unobserved (latent) source 
    activities that gives rise to m/eeg data.
    """
    cv_lambdas = None
    mse_path = None
    es_path = None

    def __init__(self, order, self_history, n_eigenmodes, n_orients, cv, n_jobs, 
                 copy=True, standardize=False, normalize=False, use_lapack=True,
                 config=None):
        self.cv = cv
        self.n_jobs = n_jobs
        # config must be forwarded: NeuraLVAR.__init__ ends with
        # `self.config = config`, so omitting it here wipes the assignment
        NeuraLVAR.__init__(self, order, self_history, n_eigenmodes, n_orients,
                           copy, standardize, normalize, use_lapack, config)

    @classmethod
    def from_config(cls, config):
        return cls(
            order = config.latent.order,
            self_history = config.sparsity.self_history,
            n_eigenmodes = config.latent.n_eigenmodes,
            n_orients = config.latent.n_orients,
            cv = config.validation.cv,
            n_jobs = config.parallel.n_workers,
            copy = True,
            standardize = False,
            normalize = False,
            use_lapack = config.numerical.use_lapack,
            config=config,
        )

    def _cvfit(self, lambda_t, info_y, info_f, info_r, info_cv, info_pred, 
               splits, em_state, config):
        lam_idx, lambda_ = lambda_t

        if config.numerical.verbose:
            print(f"{current_process().name} CV fitting with {lambda_}")

        try:
            y, shm_y = link_share_memory(info_y)
            F, shm_f = link_share_memory(info_f)
            R, shm_r = link_share_memory(info_r)
            cv, shm_c = link_share_memory(info_cv)
            pred, shm_pred = link_share_memory(info_pred)
        except BaseException as exc:
            print(f"Could not link to memory: {exc}")
            raise exc


        for split_idx in range(len(splits)):
            train, test = splits[split_idx]
            y_train, y_test = y[train], y[test]

            # full data/train data scale factor since the ll depends to the
            # number of observations, and the training set has a reduced number 
            lambda_scale = np.sqrt(y.shape[0] / y_train.shape[0])
            
            curr_lambda = lambda_ * lambda_scale

            if config.numerical.verbose:
                print(f"{current_process().name} {split_idx=} {curr_lambda=}")

            # deepcopy: each lambda must start from the same initial state,
            # otherwise it warm-starts from the previous (larger) lambda's fit
            fit_state, _ = self._fit(y_train, F, R, curr_lambda,
                                     copy.deepcopy(em_state))

            # test set prediction. forward_filter_blas returns
            # (em_state, filter_result) -- binding the whole tuple made the
            # attribute lookups below fail.
            _, filter_result_test = forward_filter_jax(y_test, F, R,
                                                        em_state=fit_state)
            
            # full-data prediction for ES
            _, filter_result_full = forward_filter_jax(y, F, R,
                                                        em_state=fit_state)

            # full data prediction for GCV
            _, smoother_result = measurement_smoother_jax(y, F, R, 
                                                          em_state=fit_state)
            
            # get GCV params
            A = fit_state.A[:fit_state.N_sources_upper]
            df = np.sum(np.abs(A) > 1e-12)  # nonzero support
            N = y.shape[0]                  # num samples
            D = -smoother_result.model_fit  # positive fit criterion

            # nll metric 
            cv[0, split_idx, lam_idx] = \
                filter_result_test.negative_log_likelihood
            
            # GCV metric
            cv[1, split_idx, lam_idx] = self.GCV(D, N, df)
            
            # raw disturbances (negative fit criterion)
            cv[2, split_idx, lam_idx] = smoother_result.model_fit

            pred[split_idx, lam_idx][:] = filter_result_full.filtered_state 

        for shm in (shm_y, shm_f, shm_r, shm_c, shm_pred):
            shm.close()
    
        return em_state

    def fit(self, y, F, R, em_state):
        """
        Fits the model from given m/eeg data, forward gain and noise covariance

        y : ndarray of shape (n_samples, n_channels)
        F : ndarray of shape (n_channels, n_sources)
        R : ndarray of shape (n_channels, n_channels)
        em_state: dataclass
        config: dataclass
        """
        kf = TimeSeriesSplit(n_splits=self.cv)

        cvsplits = [split for split in kf.split(y)]

        if self.config.validation.cv_type in ['GCV', 'DisturbanceCV']:
            # use split structure for normal cv, but with one 'dummy' split
            # including all of the data
            self.cv = 1
            idxs = np.arange(len(y))
            cvsplits = [(idxs, idxs)] # use complete data

        lambda_range = sorted(list(self.config.sparsity.lambda_range))
        
        # (cvmetric, split, lambda) -- cv info for each split
        cv_mat = np.zeros((3, len(cvsplits), len(lambda_range)), dtype=y.dtype)
        
        # (split, lambda, time, sensor) -- model prediction for each split
        pred_mat = np.zeros((len(cvsplits), 
                             len(lambda_range),
                             y.shape[0], 
                             F.shape[1]),
                             dtype=y.dtype)

        # Use parallel processing
        # A, b, mu_range, cv_mat needs to shared across processes
        shared_y, info_y, shm_y = create_shared_mem(y)
        shared_f, info_f, shm_f = create_shared_mem(F)
        shared_r, info_r, shm_r = create_shared_mem(R)
        shared_cv_mat, info_cv, shm_c = create_shared_mem(cv_mat)
        shared_pred_mat, info_pred, shm_p = create_shared_mem(pred_mat)
        initargs = (info_y, info_f, info_r, info_cv, info_pred, cvsplits, 
                    copy.deepcopy(em_state), self.config)

        if self.config.numerical.verbose:
            mode = self.config.validation.cv_type
            n_splits = len(cvsplits)
            print(f"Starting cross-validation with N_jobs:{self.n_jobs}, "
                  f"mode:{mode}, splits:{n_splits}")
    
        lastsplit_em_states = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._cvfit)((lam_idx, lam_), *initargs) 
                for lam_idx, lam_ in enumerate(lambda_range)
        )

        if self.config.numerical.verbose:
            print('Done cross-validation')

        self.cv_lambdas = lambda_range
        cv_mat[:] = np.reshape(shared_cv_mat, cv_mat.shape)
        pred_mat[:] = np.reshape(shared_pred_mat, pred_mat.shape)
        self.mse_path = cv_mat
        self.es_path = compute_es_criterion(pred_mat)

        for shm in (shm_y, shm_f, shm_r, shm_c, shm_p):
            shm.close()
            try:
                shm.unlink()
            except Exception as exc:
                print(f"Unlink shared-memory issue: {exc}")
        
        # find best lambda 
        # if Estimation Stability criterion we use cv_mat[0] and pred_mat else
        # we use the Generalized CV metric.
        if self.config.validation.use_es:
            # best log likelihood lambda
            index = self.mse_path[0].mean(axis=0).argmin()
            try:
                best_lambda = lambda_range[np.nanargmin(self.es_path[:index])]
            except ValueError as ex:
                print(f"Estimation Stability error: {ex} fallback to ML lambda")
                best_lambda = lambda_range[index]
            print(f'\n\nbest_regularizing parameter: {best_lambda} using es\n')

            em_state, smoother_result = self._fit(y, F, R, best_lambda, 
                                                  em_state)
        else:
            # lambda list
            x_gcv = lambda_range

            if self.config.validation.cv_type == 'GCV':
                # GCV values
                y_gcv = self.mse_path[1, 0, :] 
            elif self.config.validation.cv_type == 'DisturbanceCV':
                # raw disturbance smoother values
                y_gcv = self.mse_path[2, 0, :] 
            else:
                raise Exception(f"cv type {self.config.validation.cv_type} "
                                 "not supported.")

            # find knee in L-shaped curve
            best_lambda = KneeLocator(x_gcv, y_gcv, 
                                      direction='decreasing',
                                      curve='convex').knee
            
            print(f"\nmeasurement disturbance smoother CVs: {x_gcv}, {y_gcv}\n")
            
            print(f'\n\nbest_regularizing parameter: {best_lambda} using GCV\n')

            best_lambda_idx = np.where(x_gcv == best_lambda)[0][0]

            # already computed using full data (lastsplit is full data dummy
            # split)
            em_state = lastsplit_em_states[best_lambda_idx]
            em_state, smoother_result = rts_smoother_jax(y, F, R, em_state)

        m = em_state.N_sources_upper
        self._parameters = (
            self._unravel_a(em_state.A[:m]), 
            F[:, :m], 
            em_state.Q[:em_state.N_sources_upper, :em_state.N_sources_upper], 
            R, 
            smoother_result.smoothed_state
        )

        self.ll = -smoother_result.negative_log_likelihood
        self.lambda_ = best_lambda

        # y is time-major, so the sample count is axis 0. Reading axis 1 here
        # normalised aic/bic by the CHANNEL count.
        t, _ = y.shape
        df = (abs(em_state.A[:m]) > 1e-15).sum()
        self.aic = (2*df - 2*self.ll) / t
        self.bic = (np.log(t)*df - 2*self.ll) / t

        return em_state, smoother_result
    

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


def compute_es_criterion(pred):
    cv_split_repeats = np.arange(pred.shape[0]) + 1
    cv_split_repeats[:] = 1
    shape = pred.shape[:-2] + (-1,)
    pred.shape = shape
    es = np.empty(pred.shape[1], pred.dtype)
    for j in range(pred.shape[1]):
        this_pred = pred[:, j, :]
        this_pred_mean = (this_pred * cv_split_repeats[:, None]).mean(axis=0)
        fluctuation = (this_pred - this_pred_mean[None, :]) * \
            np.sqrt(cv_split_repeats[:, None])
        es[j] = (fluctuation ** 2).sum() / (this_pred_mean ** 2).sum()
    return es 
