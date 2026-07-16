from multiprocessing import shared_memory, current_process
import numpy as np
import re
import warnings
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import cpu_count
import mne
from mne.utils import logger
from nlgc.opt.em import solve_params
from nlgc.bias_utils import sample_path_bias, bias_by_idx
from nlgc.opt.kalman.filter import forward_filter_blas
from nlgc.utils.restriction import restriction_to_zeroed_index
import copy


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
    _zeroed_index = None
    restriction = None

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
        m = F.shape[1]
        if em_state.N_sources_upper is None:
            em_state.N_sources_upper = m

        p = self.config.latent.order

        # grab upper portions from previous step for forming other matrix 
        # companions forms 
        A = em_state.A[:m] 
        Q = em_state.Q[:m, :m]

        A, F, Q, R = self.companion_form(A, F, Q, R) 

        # overwrite if not already in companion form
        em_state.A = A 
        em_state.Q = Q 

        zeroed_index = restriction_to_zeroed_index(self.restriction, m, p)
        self._zeroed_index = zeroed_index

        em_state, smoother_result = solve_params(
                y, F, R,
                em_state=em_state,
                config=self.config,
                lambda_=lambda_,
                zeroed_index=zeroed_index,
        )
        
        return em_state, smoother_result

   
    def compute_norm_one(self, a):
        return np.sum(np.absolute(a))


    def compute_two_one_norm(self, a):
        print(f'a_ shape is {a.shape}')
        p = a.shape[0]
        m = a.shape[1]
        N = m // 3
        B = a.reshape(N, 3, p, N, 3)
        return (np.sqrt(np.sum(B * B, axis=(1, 4), keepdims=True)).sum())
    
    def compute_bias(self, em_state):
        m = em_state.N_sources_upper
        A = em_state.A[:m]
        Q = em_state.Q[:m, :m]
        x = em_state.smoothed_state[:, :m]
        A_mask = em_state.A_mask[:m]
        bias = sample_path_bias(Q, A, x, A_mask, 
                                self._n_eigenmodes, self._n_orients)
        return bias


    def compute_bias_idx(self, source, em_state, smoother_result):
        p = self.config.latent.order
        m = em_state.N_sources_upper
        A = em_state.A[:m]
        Q = em_state.Q[:m, :m]
        x = em_state.smoothed_state[:, :m]
        P = em_state.smoothed_cov[:m, :m]
        B = smoother_result.smoother_gain[:m, :m]
        A_mask = em_state.A_mask[:m]

        if isinstance(source, int):
            source = tuple(source)
            
        bias = sum([bias_by_idx(i, Q, A, x, P, B, m, p, A_mask) \
                    for i in source])
        return bias


    def fit(self, y, F, R, lambda_, em_state, restriction=None):
        if (restriction is None or re.search('->', restriction)) is False:
            raise ValueError(f"restriction:{restriction} should be None or should have format 'i->j'!")
        self.restriction = restriction
        em_state, smoother_result = self._fit(y, F, R, lambda_, em_state)
        self._parameters = (em_state.A, F, em_state.Q, 
                            R, smoother_result.smoothed_state)
        self.ll = -smoother_result.negative_log_likelihood

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


    @staticmethod
    def companion_form(a, f, q, r):
        n, m = f.shape
        assert len(a.shape) == 3 # unraveled a
        p = len(a)


        assert q.shape[0] == m
        q_upper = q
        a_companion = np.block([[np.zeros((m, m * p))],
                            [np.eye(m * (p - 1)), np.zeros((m * (p - 1), m))]])
        
        a_upper = NeuraLVAR._ravel_a(a)
        a_companion[:m] = a_upper

        q_companion = np.zeros((m * p, m * p))
        non_zero_indices = np.diag_indices_from(q_upper)
        q_companion[non_zero_indices] = q_upper[non_zero_indices]

        f_companion = np.hstack((f, np.zeros((n, m * (p - 1)))))
        
        r_companion = r # pass through

        return a_companion, f_companion, q_companion, r_companion


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
        self.config = config
        NeuraLVAR.__init__(self, order, self_history, n_eigenmodes, n_orients, 
                           copy, standardize, normalize, use_lapack)


    @classmethod
    def from_config(cls, config):
        return cls(
            order = config.latent.order,
            self_history = config.sparsity.self_history,
            n_eigenmodes = config.latent.n_eigenmodes,
            n_orients = config.latent.n_orients,
            cv = config.validation.cv,
            n_jobs = min(config.validation.cv, cpu_count()),
            copy = True,
            standardize = False,
            normalize = False,
            use_lapack = config.numerical.use_lapack,
            config=config,
        )

    def _cvfit(self, split, info_y, info_f, info_r, info_cv, info_pred, 
               splits, em_state, config):
        if config.numerical.verbose:
            mne.set_log_level(True)
            print(f"{current_process().name} working on {split}th split")
            logger.info(f"{current_process().name} working on {split}th split")
            logger.debug(f"{current_process().name} working on {split}th split")

        try:
            y, shm_y = link_share_memory(info_y)
            F, shm_f = link_share_memory(info_f)
            R, shm_r = link_share_memory(info_r)
            cv, shm_c = link_share_memory(info_cv)
            pred, shm_pred = link_share_memory(info_pred)
        except BaseException as exc:
            logger.error(f"Could not link to memory: {exc}")
            raise exc

        lambda_range = config.validation.lambda_range
        train, test = splits[split]
        y_train, y_test = y[:, train], y[:, test]

        # full data/train data scale factor since the ll depends to the number
        # of observations, and the training set has a reduced number 
        lambda_scale = np.sqrt(y.shape[-1] / y_train.shape[-1])

        for idx, lambda_full in enumerate(lambda_range):
            curr_lambda = lambda_full * lambda_scale
            if config.numerical.verbose:
                logger.info(f"{current_process().name} {split=} {curr_lambda=}")

            em_state, _ = self._fit(y_train, F, R, em_state)
            A_cv = em_state.A

            # test set prediction
            filter_result_test = forward_filter_blas(y_test, F, R, 
                                                     em_state=em_state, 
                                                     use_lapack=True)
            
            # full-data prediction
            filter_result_full = forward_filter_blas(y, F, R, 
                                                     em_state=em_state, 
                                                     use_lapack=True)
            

            # TODO this should probably use the disturbance smoother to evaluate
            # model fit
            # # different criteria for cross-validation
            cv[0, split, idx] = filter_result_test.negative_log_likelihood
            cv[1, split, idx] = curr_lambda * self.compute_norm_one(A_cv)
            pred[split, idx][:] = filter_result_full.filtered_state 

        for shm in (shm_y, shm_f, shm_r, shm_c):
            shm.close()
        return None
    

    def fit(self, y, F, R, em_state, restriction=None):
        """Fits the model from given m/eeg data, forward gain and noise 
        covariance

        y : ndarray of shape (n_channels, n_samples)
        F : ndarray of shape (n_channels, n_sources)
        R : ndarray of shape (n_channels, n_channels)
        em_state: 
        config: 
        restriction : regular expression like 'i->j', default = None
            i and j should be integers.
        """

        if (restriction is None or re.search('->', restriction)) is False:
            raise ValueError(f"restriction:{restriction} should be None or should have format 'i->j'!")
        self.restriction = restriction

        kf = TimeSeriesSplit(n_splits=self.cv)
        cvsplits = [split for split in kf.split(y.T)]
        lambda_range = self.config.validation.lambda_range

        cv_mat = np.zeros((2, len(cvsplits), len(lambda_range)), dtype=y.dtype)
        pred_mat = np.zeros((len(cvsplits), len(lambda_range)) + y.shape, 
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

        logger.info('Starting cross-validation')

        Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._cvfit)(i, *initargs) for i in range(len(cvsplits))
        )

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
            except Exception as exc:
                logger.info(f"Unlink shared-memory issue: {exc}")

        # Find best mu
        # If Estimation stability criterion is used we need cv_mat[0] and 
        # pred_mat else we just use $\lambda * ||A||_1$ as the metric.
        if self.config.validation.use_es:
            index = self.mse_path[0].mean(axis=0).argmax()
            try:
                best_lambda = lambda_range[np.nanargmin(self.es_path[:index])]
            except ValueError:
                best_lambda = lambda_range[index]
            logger.info(f'best_regularizing parameter: {best_lambda} using es')
        else:
            index = self.mse_path[1].mean(axis=0).argmax()
            best_lambda = lambda_range[index]
            logger.info(f'best_regularizing parameter: {best_lambda}')

        em_state, smoother_result = self._fit(y, F, R, best_lambda, em_state)
        m = em_state.N_sources_upper
        A = self._unravel_a(em_state.A[:m])
        Q = em_state.Q[:m, :m]
        x = em_state.smoothed_sources
        self._parameters = (A, F, Q, R, x)
        self.ll = -smoother_result.negative_log_likelihood
        self.lambda_ = best_lambda

        _, t = y.shape
        df = (abs(A) > 1e-15).sum()
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
