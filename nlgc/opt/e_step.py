import numpy as np
from scipy import linalg
import control


def sskf(y, a, f, q, r, xs=None, use_lapack=True):
    """Computes steady-state smoothed distribution

    y_{i} = fx_{i} + n_{i}   n_{i} ~ N(0, r)
    x_{i} = ax_{i-1} + u_{i},  u_{i} ~ N(0, q)

    Parameters
    ----------
    y: ndarray of shape (n_samples, n_channels)
    a: ndarray of shape (n_sources*order, n_sources*order)
    f: ndarray of shape (n_channels, n_sources*order)
    q: ndarray of shape (n_sources*order, n_sources*order)
    r: ndarray of shape (n_channels, n_channels)
    xs: tuple of two ndarrays of shape (n_samples, n_sources*order)
        if provided needs to be F contiguous

    Returns
    -------
    x_ : ndarray of shape (n_samples, n_sources*order)
    s : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoothing gain
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances for sampling
    Notes:
    See README and/or [1]_ for the difference between s and s_hat.
    [1]_ Fruhwirth-Schnatter, Sylvia (1992) Data Augmentation and Dynamic Linear Models.
    URL: https://epub.wu.ac.at/id/eprint/392
    """
    assert y.shape[1] == f.shape[0]
    t, dy = y.shape
    _, dx = f.shape
    if xs is None:
        _x = np.empty((t, dx), dtype=np.float64)
        x_ = np.empty_like(_x)
    else:
        _x, x_ = xs
        assert _x.shape[0] == y.shape[0]
        assert _x.flags['C_CONTIGUOUS']
        assert x_.flags['C_CONTIGUOUS']

    try:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)           
    except np.linalg.LinAlgError:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    except ValueError:
        try:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=True, method=None)
        except ValueError:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=False, method=None)


    temp = f.dot(_s)
    temp2 = temp.dot(f.T) + r
    (l, low) = linalg.cho_factor(temp2, check_finite=False)
    k = linalg.cho_solve((l, low), temp, check_finite=False)
    inv_innov_cov = linalg.cho_solve((l, low), np.eye(dy), check_finite=False)
    logdet_inno_cov = np.log(np.diag(l)).sum()  # already multiplied by 1/2
    k = k.T  # Kalman Gain
    s = _s.copy()
    s -= k.dot(temp)
    temp = a.dot(s)
    try:
        (l, low) = linalg.cho_factor(_s, lower=True, check_finite=False)
        b = linalg.cho_solve((l, low), temp, check_finite=False)
    except np.linalg.LinAlgError:
        b, *rest = linalg.lstsq(_s, temp, check_finite=False)

    b = b.T  # Smoother Gain
    s_hat = s - b.dot(_s).dot(b.T)  # See README what this means!
    s_ = linalg.solve_discrete_lyapunov(b, s_hat)
    if (np.diag(s_) <= 0).any():
        raise ValueError('diag(s_) values are not non-negative!')
    # s_ = s + b.dot(s - _s).dot(b.T)     # Approximation from Elvira's paper

    f, a, k, b = align_cast((f, a, k, b), use_lapack)

    temp = np.empty(dy, dtype=np.float64)
    temp1 = np.empty(dx, dtype=np.float64)
    temp2 = np.empty(dx, dtype=np.float64)
    if use_lapack:
        dot = linalg.get_blas_funcs(['gemv'], (a, x_[0]))[0]
    ll = 0.0
    for i in range(t):
        if i == 0:
            _x[i] = 0
        else:
            if not use_lapack:
                _x[i] = np.dot(a, x_[i - 1], out=_x[i])
            else:
                _x[i] = dot(1, a, x_[i - 1], y=_x[i], overwrite_y=True)
        x_[i][:] = _x[i][:]
        # x_[i] += k.dot(y[i]-f.dot(_x[i]))
        if not use_lapack:
            temp = np.dot(f, _x[i], out=temp)
            temp *= -1
            temp += y[i]
            temp1 = np.dot(k, temp, out=temp1)
            x_[i] += temp1
        else:
            temp[:] = y[i]
            # temp = - f.dot(_x[i]) + y[i]
            # temp =  - f.dot(_x[i]) + temp
            temp = dot(-1.0, f, _x[i], beta=1.0, y=temp, overwrite_y=True)
            # x_[i] = k.dot(temp) + x_[i]
            dot(1.0, k, temp, beta=1.0, y=x_[i], overwrite_y=True)

        ll += 0.5 * np.sum(inv_innov_cov.dot(temp) * temp) + logdet_inno_cov

    # i = t-1 case is already taken care of.
    for i in reversed(range(t - 1)):
        # temp = x_[i+1] - _x[i+1]
        # x_[i] += b.dot(temp)
        temp1[:] = x_[i + 1]
        temp1 -= _x[i + 1]
        if not use_lapack:
            temp2 = np.dot(b, temp1, out=temp2)
            x_[i] += temp2
        else:
            # x_[i] = b.dot(temp1) + x_[i]
            dot(1.0, b, temp1, beta=1.0, y=x_[i], overwrite_y=True)
    return x_, s, s_, b, s_hat, -ll


def sskfcv(y, a, f, q, r, xs=None, use_lapack=True):
    """Computes steady-state smoothed distribution

    y_{i} = fx_{i} + n_{i}   n_{i} ~ N(0, r)
    x_{i} = ax_{i-1} + u_{i},  u_{i} ~ N(0, q)

    Parameters
    ----------
    y: ndarray of shape (n_samples, n_channels)
    a: ndarray of shape (n_sources*order, n_sources*order)
    f: ndarray of shape (n_channels, n_sources*order)
    q: ndarray of shape (n_sources*order, n_sources*order)
    r: ndarray of shape (n_channels, n_channels)
    xs: tuple of two ndarrays of shape (n_samples, n_sources*order)
        if provided needs to be F contiguous

    Returns
    -------
    x_ : ndarray of shape (n_samples, n_sources*order)
    s : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoothing gain
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances for sampling
    Notes:
    See README and/or [1]_ for the difference between s and s_hat.
    [1]_ Fruhwirth-Schnatter, Sylvia (1992) Data Augmentation and Dynamic Linear Models.
    URL: https://epub.wu.ac.at/id/eprint/392
    """
    assert y.shape[1] == f.shape[0]
    t, dy = y.shape
    _, dx = f.shape
    if xs is None:
        _x = np.empty((t, dx), dtype=np.float64)
        x_ = np.empty_like(_x)
    else:
        _x, x_ = xs
        assert _x.shape[0] == y.shape[0]
        assert _x.flags['C_CONTIGUOUS']
        assert x_.flags['C_CONTIGUOUS']

    # try:
    #     _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    # except ValueError:
    #     _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)
    try:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)           
    except np.linalg.LinAlgError:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    except ValueError:
        try:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=True, method=None)
        except ValueError:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=False, method=None)

    temp = f.dot(_s)
    temp2 = temp.dot(f.T) + r
    (l, low) = linalg.cho_factor(temp2, check_finite=False)
    k = linalg.cho_solve((l, low), temp, check_finite=False)
    k = k.T  # Kalman Gain

    temp2_inv = linalg.cho_solve((l, low), np.eye(dy), check_finite=False)
    temp3 = f.T.dot(temp2_inv.dot(f))
    l = a.dot(np.eye(dx) - k.dot(f))
    u = linalg.solve_discrete_lyapunov(l, temp3)
    k_ = a.dot(k)
    c = temp2_inv + np.linalg.multi_dot((k_.T, u, k_))

    f, a, k = align_cast((f, a, k), use_lapack)

    temp = np.empty(dy, dtype=np.float64)
    temp1 = np.empty(dx, dtype=np.float64)
    if use_lapack:
        dot = linalg.get_blas_funcs(['gemv'], (a, x_[0]))[0]
    for i in range(t):
        if i == 0:
            _x[i] = 0
        else:
            if not use_lapack:
                _x[i] = np.dot(a, x_[i - 1], out=_x[i])
            else:
                _x[i] = dot(1, a, x_[i - 1], y=_x[i], overwrite_y=True)
        x_[i][:] = _x[i][:]
        # x_[i] += k.dot(y[i]-f.dot(_x[i]))
        if not use_lapack:
            temp = np.dot(f, _x[i], out=temp)
            temp *= -1
            temp += y[i]
            temp1 = np.dot(k, temp, out=temp1)
            x_[i] += temp1
        else:
            temp[:] = y[i]
            # temp = - f.dot(_x[i]) + y[i]
            # temp =  - f.dot(_x[i]) + temp
            temp = dot(-1.0, f, _x[i], beta=1.0, y=temp, overwrite_y=True)
            # x_[i] = k.dot(temp) + x_[i]
            dot(1.0, k, temp, beta=1.0, y=x_[i], overwrite_y=True)

    e = y - _x.dot(f.T)
    temp4 = e.dot(temp2_inv)
    n = np.empty_like(y)
    x_[-1] = 0.0
    for i in reversed(range(t)):
        n[i] = temp4[i] - k_.T.dot(x_[i])
        if i > 0:
            x_[i - 1] = f.T.dot(temp4[i]) + l.T.dot(x_[i])
    model_fit = (n * n).sum() / (t * np.diag(c).sum()) ** 2
    return -model_fit


def sskf_prediction(y, a, f, q, r, xs=None, use_lapack=True):
    """Computes steady-state smoothed distribution

    y_{i} = fx_{i} + n_{i}   n_{i} ~ N(0, r)
    x_{i} = ax_{i-1} + u_{i},  u_{i} ~ N(0, q)

    Parameters
    ----------
    y: ndarray of shape (n_samples, n_channels)
    a: ndarray of shape (n_sources*order, n_sources*order)
    f: ndarray of shape (n_channels, n_sources*order)
    q: ndarray of shape (n_sources*order, n_sources*order)
    r: ndarray of shape (n_channels, n_channels)
    xs: tuple of two ndarrays of shape (n_samples, n_sources*order)
        if provided needs to be F contiguous

    Returns
    -------
    x_ : ndarray of shape (n_samples, n_sources*order)
    s : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoothing gain
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances for sampling
    Notes:
    See README and/or [1]_ for the difference between s and s_hat.
    [1]_ Fruhwirth-Schnatter, Sylvia (1992) Data Augmentation and Dynamic Linear Models.
    URL: https://epub.wu.ac.at/id/eprint/392
    """
    assert y.shape[1] == f.shape[0]
    t, dy = y.shape
    _, dx = f.shape
    if xs is None:
        _x = np.empty((t, dx), dtype=np.float64)
        x_ = np.empty_like(_x)
    else:
        _x, x_ = xs
        assert _x.shape[0] == y.shape[0]
        assert _x.flags['C_CONTIGUOUS']
        assert x_.flags['C_CONTIGUOUS']

    # try:
    #     _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)
    # except np.linalg.LinAlgError:
    #     _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    try:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=False)           
    except np.linalg.LinAlgError:
        _s = linalg.solve_discrete_are(a.T, f.T, q, r, balanced=True)
    except ValueError:
        try:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=True, method=None)
        except ValueError:
            _s, _, _ = control.dare(a.T, f.T, q, r, stabilizing=False, method=None)

    temp = f.dot(_s)
    temp2 = temp.dot(f.T) + r
    (l, low) = linalg.cho_factor(temp2, check_finite=False)
    k = linalg.cho_solve((l, low), temp, check_finite=False)
    k = k.T  # Kalman Gain
    s = _s.copy()
    s -= k.dot(temp)
    temp = a.dot(s)
    try:
        (l, low) = linalg.cho_factor(_s, lower=True, check_finite=False)
        b = linalg.cho_solve((l, low), temp, check_finite=False)
    except np.linalg.LinAlgError:
        b, *rest = linalg.lstsq(_s, temp, check_finite=False)

    b = b.T  # Smoother Gain
    s_hat = s - b.dot(_s).dot(b.T)  # See README what this means!
    s_ = linalg.solve_discrete_lyapunov(b, s_hat)
    # s_ = s + b.dot(s - _s).dot(b.T)     # Approximation from Elvira's paper

    f, a, k, b = align_cast((f, a, k, b), use_lapack)

    temp = np.empty(dy, dtype=np.float64)
    temp1 = np.empty(dx, dtype=np.float64)
    temp2 = np.empty(dx, dtype=np.float64)
    pred = np.empty_like(y)
    if use_lapack:
        dot = linalg.get_blas_funcs(['gemv'], (a, x_[0]))[0]
    for i in range(t):
        if i == 0:
            _x[i] = 0
        else:
            if not use_lapack:
                _x[i] = np.dot(a, x_[i - 1], out=_x[i])
            else:
                _x[i] = dot(1, a, x_[i - 1], y=_x[i], overwrite_y=True)
        x_[i][:] = _x[i][:]
        # x_[i] += k.dot(y[i]-f.dot(_x[i]))
        if not use_lapack:
            temp = np.dot(f, _x[i], out=temp)
            temp *= -1
            temp += y[i]
            temp1 = np.dot(k, temp, out=temp1)
            x_[i] += temp1
            pred[i] = np.dot(f, _x[i], out=pred[i])
        else:
            temp[:] = y[i]
            # temp = - f.dot(_x[i]) + y[i]
            # temp =  - f.dot(_x[i]) + temp
            temp = dot(-1.0, f, _x[i], beta=1.0, y=temp, overwrite_y=True)
            # x_[i] = k.dot(temp) + x_[i]
            dot(1.0, k, temp, beta=1.0, y=x_[i], overwrite_y=True)
            pred[i] = dot(1.0, f, _x[i], y=pred[i], overwrite_y=True)
    return pred


def align_cast(args, use_lapack):
    """internal function to typecast (to np.float64) and/or memory-align ndarrays


    Parameters
    ----------
    args: tuple of ndarrays of arbitrary shape
    use_lapack: bool
        whether to make F_contiguous or not.
    Returns
    -------
    args: tuple
        after alignment and typecasting
    """
    args = tuple([arg if arg.dtype == np.float64 else arg.astype(np.float64) for arg in args])
    if use_lapack:
        args = tuple([arg if arg.flags['F_CONTIGUOUS'] else arg.copy(order='F') for arg in args])
    return args


def test_sskf(t=1000):
    from matplotlib import pyplot as plt

    # n, m = 155, 6*2*68
    n, m = 3, 3
    q = np.eye(m)
    r = 0.01 * np.eye(n)
    sn = np.random.standard_normal((m + n) * t)
    u = sn[:m * t]
    u.shape = (t, m)
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)
    v = sn[m * t:]
    v.shape = (t, n)
    l = linalg.cholesky(r, lower=True)
    v = v.dot(l.T)
    a = np.random.randn(m, m)
    a /= 1.1 * linalg.norm(a)
    # f = np.random.randn(n, m)
    f = np.eye(3)
    x = np.empty((t, m), dtype=np.float64)
    x[0] = 0.0
    for x_, _x, u_ in zip(x[1:], x, u):
        x_[:] = a.dot(_x) + u_
    y = x.dot(f.T) + v
    # fig, ax = plt.subplots()
    # ax.plot(x)
    # ax.plot(y)
    # fig.show()

    _x = np.empty((y.shape[0], m), dtype=np.float64)
    x_ = np.empty_like(_x)

    # pr = cProfile.Profile()
    # pr.enable()
    # x_, s_, b, _ = sskf(y, a, f, q, r, xs=(_x, x_), use_lapack=True)
    # pr.disable()
    # s1 = io.StringIO()
    # ps = pstats.Stats(pr, stream=s1).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s1.getvalue())
    #
    # pr = cProfile.Profile()
    # pr.enable()
    # x__, s__, b_, _ = sskf(y, a, f, q, r, xs=(_x, x_), use_lapack=False)
    # pr.disable()
    # s2 = io.StringIO()
    # ps = pstats.Stats(pr, stream=s2).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats()
    # print(s2.getvalue())

    from codetiming import Timer

    t1 = Timer(name='opt', logger=None)
    t2 = Timer(name='vanilla', logger=None)

    for _ in range(10):
        with t1:
            x_, s_, b, _, _ = sskf(y, a, f, q, r, xs=(_x, x_), use_lapack=True)
    print("Elapsed time: {:.4f}\pm{:.4f}".format(Timer.timers.mean("opt"), Timer.timers.stdev("opt")))
    for _ in range(10):
        with t2:
            x__, s__, b, _, _ = sskf(y, a, f, q, r, xs=(_x, x_), use_lapack=False)
    print("Elapsed time: {:.4f}\pm{:.4f}".format(Timer.timers.mean("vanilla"), Timer.timers.stdev("vanilla")))

    fig, axes = plt.subplots(x.shape[1])
    for xi, xi_, xi__, ax in zip(x.T, x_.T, x__.T, axes):
        ax.plot(xi)
        ax.plot(xi_)
        ax.plot(xi__)
    fig.show()
