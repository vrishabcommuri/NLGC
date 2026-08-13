import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

Array = NDArray[np.float64]

@dataclass
class SSMState:
    y: Array
    x: Array
    A: Array
    F: Array
    Q: Array
    R: Array
    N_times: int
    N_sources: int
    N_sensors: int


def gen_small_ssm(T=200, nx=4, ny=6):
    rng = np.random.default_rng(0)

    A = 0.8 * np.eye(nx)
    F = rng.normal(size=(ny, nx))
    Q = 0.1 * np.eye(nx)
    R = 0.05 * np.eye(ny)

    x = np.zeros((T, nx))
    y = np.zeros((T, ny))

    for t in range(1, T):
        x[t] = A @ x[t-1] + rng.multivariate_normal(np.zeros(nx), Q)

    for t in range(T):
        y[t] = F @ x[t] + rng.multivariate_normal(np.zeros(ny), R)

    ssm_state = SSMState(
        y=y,
        x=x,
        A=A,
        F=F,
        Q=Q,
        R=R,
        N_times=T,
        N_sources=nx,
        N_sensors=ny,
    )

    return ssm_state


def gen_large_ssm(
    T=1000,
    nx=300,
    ny=100,
    seed=0,
    state_scale=0.95,
    q_scale=0.1,
    r_scale=0.05,
):
    rng = np.random.default_rng(seed)

    # Random stable transition matrix
    A = rng.normal(size=(nx, nx))
    eigvals = np.linalg.eigvals(A)
    A = A / np.max(np.abs(eigvals)) * state_scale

    # Random observation matrix
    F = rng.normal(size=(ny, nx)) / np.sqrt(nx)

    Q = q_scale * np.eye(nx)
    R = r_scale * np.eye(ny)

    x = np.zeros((T, nx))
    y = np.zeros((T, ny))

    process_noise = rng.multivariate_normal(
        np.zeros(nx),
        Q,
        size=T,
    )

    observation_noise = rng.multivariate_normal(
        np.zeros(ny),
        R,
        size=T,
    )

    for t in range(1, T):
        x[t] = A @ x[t-1] + process_noise[t]

    y = x @ F.T + observation_noise

    ssm_state = SSMState(
        y=y,
        x=x,
        A=A,
        F=F,
        Q=Q,
        R=R,
        N_times=T,
        N_sources=nx,
        N_sensors=ny,
    )

    return ssm_state


def gen_sparse_var_ssm(
    T=1000,
    n_sources=None,
    n_sensors=None,
    order=2,
    n_orients=1,
    n_eigenmodes=1,
    sparsity=0.25,
    self_decay=0.8,
    cross_scale=0.1,
    q_scale=0.05,
    r_scale=0.05,
    seed=0,
    leadfield=None,
    measurement_noise_scale=1e2,
):
    """
    generate sparse VAR(p) state-space model.

    The latent process is:

        x_t = A_1 x_{t-1}
            + A_2 x_{t-2}
            + ...
            + w_t

    represented internally using companion form.

    Parameters
    ----------
    leadfield : dict | None
        Keyword arguments for `nlgc.nlgc_test_utils.lead_field_generation`
        (`root`, `subject_id`, `src_space`, ...). When given, a real leadfield
        replaces the random observation matrix and dictates `n_sources` and
        `n_sensors`, which must then be omitted. Default None.

    measurement_noise_scale : float
        Leadfield path only: signal power / noise power. A fixed `r_scale` is
        meaningless there because the gain is whitened and column-normalized.

    Returns
    -------
    ssm : SSMState

    A_lags : list[np.ndarray]
        True lag matrices.

    supports : list[np.ndarray]
        True sparse connectivity masks.
    """

    rng = np.random.default_rng(seed)

    if leadfield is not None:
        # ignoring these would return an SSMState disagreeing with its own args
        if n_sources is not None or n_sensors is not None:
            raise ValueError(
                'n_sources and n_sensors are dictated by the leadfield; omit them '
                'when passing leadfield')
        G, n_sources, n_sensors = _leadfield_observation(
            leadfield, n_eigenmodes, n_orients)
    else:
        G = None
        n_sources = 4 if n_sources is None else n_sources
        n_sensors = 8 if n_sensors is None else n_sensors

    A_lags, supports = gen_sparse_var_lags(
        n_sources=n_sources,
        order=order,
        block_size=n_eigenmodes * n_orients,
        sparsity=sparsity,
        self_decay=self_decay,
        cross_scale=cross_scale,
        seed=seed,
    )

    A = make_companion_from_lags(A_lags)
    rho = np.max(np.abs(np.linalg.eigvals(A)))

    # stable squash spectral radius
    while rho >= 0.95:
        scale = 0.95 / rho
        A_lags = [scale * Ak for Ak in A_lags]
        A = make_companion_from_lags(A_lags)
        rho = np.max(np.abs(np.linalg.eigvals(A)))

    state_dim = A.shape[0]

    # lag-0 block; the rest of the companion state is lagged copies
    n_units = n_sources * n_eigenmodes * n_orients

    F = np.zeros((n_sensors, state_dim))
    F[:, :n_units] = (
        G if G is not None
        # draw the full width and slice: a smaller draw would shift the rng stream
        else rng.normal(size=(n_sensors, state_dim))[:, :n_units] / np.sqrt(state_dim)
    )

    Q = np.zeros((state_dim, state_dim))

    Q[:n_units, :n_units] = (
        q_scale * np.eye(n_units)
    )

    x = np.zeros(
        (T, state_dim)
    )

    process_noise = rng.multivariate_normal(
        np.zeros(state_dim),
        Q,
        size=T,
    )

    for t in range(1, T):
        x[t] = (
            A @ x[t-1]
            + process_noise[t]
        )

    y_clean = x @ F.T

    if G is None:
        R = r_scale * np.eye(n_sensors)

        observation_noise = rng.multivariate_normal(
            np.zeros(n_sensors),
            R,
            size=T,
        )
    else:
        # gain is whitened and column-normalized, so sensor scale is arbitrary and
        # r_scale says nothing about SNR -- set noise from realized signal power
        white_noise = rng.standard_normal(y_clean.shape)

        var = np.trace(y_clean @ y_clean.T) / (
            measurement_noise_scale * np.trace(white_noise @ white_noise.T) + 1e-12)

        observation_noise = white_noise * np.sqrt(var)
        R = var * np.eye(n_sensors)

    y = y_clean + observation_noise

    ssm = SSMState(
        y=y,
        x=x,
        A=A,
        F=F,
        Q=Q,
        R=R,
        N_times=T,
        N_sources=n_units,
        N_sensors=n_sensors,
    )

    return ssm, A_lags, supports


#-------------------------------------------------------------------------------
# vector ssm and VAR(p) helpers
#-------------------------------------------------------------------------------

def make_companion_from_lags(A_lags):
    """
    convert VAR(p) lag matrices into companion form.

    Parameters
    ----------
    A_lags : list[np.ndarray]
        List of lag transition matrices.
        Each matrix has shape (nx, nx).

    Returns
    -------
    A_comp : np.ndarray
        Companion transition matrix of shape (nx*p, nx*p)
    """
    order = len(A_lags)
    nx = A_lags[0].shape[0]

    A_comp = np.zeros((nx * order, nx * order))

    # transition blocks
    for k, A_k in enumerate(A_lags):
        A_comp[:nx, k * nx:(k + 1) * nx] = A_k

    # companion shift
    for k in range(1, order):
        A_comp[
            k * nx:(k + 1) * nx,
            (k - 1) * nx:k * nx
        ] = np.eye(nx)

    return A_comp


def make_sparse_source_mask(n_sources, sparsity, rng):
    mask = rng.random((n_sources, n_sources)) < sparsity

    # avoid random self loops because those are added explicitly
    np.fill_diagonal(mask, False)

    return mask


def _leadfield_observation(leadfield, n_eigenmodes, n_orients):
    """
    (G, n_sources, n_sensors) from `lead_field_generation` keyword arguments.

    n_eigenmodes / n_orients are injected here rather than read from the dict, so
    the ground truth and the leadfield cannot disagree about block layout.
    """
    # local: nlgc_test_utils pulls in mne, matplotlib and ggc
    from nlgc.nlgc_test_utils import lead_field_generation

    G, *_ = lead_field_generation(          # (G, info, noise_cov, fwd, weights)
        n_eigenmodes=n_eigenmodes, n_orients=n_orients, **leadfield)

    n_sensors, m = G.shape
    block_size = n_eigenmodes * n_orients
    assert m % block_size == 0, (m, block_size)

    return G, m // block_size, n_sensors


def expand_source_blocks(source_mask, block_size):
    """
    expand source-level connectivity into per-source blocks. a source-source edge
    activates the full block.

    source edge i -> j becomes [block_size x block_size] i -> j

    Parameters
    ----------
    source_mask : ndarray(bool)
        Shape (n_sources, n_sources)

    block_size : int
        State units per source, n_eigenmodes * n_orients -- which is also the
        number of leadfield columns per patch.

    Returns
    -------
    block_mask : ndarray(bool)
        Shape (n_sources*block_size,
         n_sources*block_size)
    """
    n_sources = source_mask.shape[0]

    block_mask = np.zeros((n_sources * block_size,
                           n_sources * block_size), dtype=bool)

    for i in range(n_sources):
        for j in range(n_sources):
            if source_mask[i, j]:
                block_mask[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size,
                ] = True

    return block_mask


def gen_sparse_var_lags(n_sources, order, block_size=1, sparsity=0.25,
                        self_decay=0.8, cross_scale=0.1, seed=0):
    """
    generate sparse VAR(p) lag matrices. connectivity is sparse at the source
    (patch) level and expanded to blocks of `block_size` state units, which is
    n_eigenmodes * n_orients.

    Returns
    -------
    A_lags : list[np.ndarray]
        Lag matrices.

    support : list[np.ndarray]
        Boolean support masks.
    """

    rng = np.random.default_rng(seed)

    n_state = n_sources * block_size

    A_lags = []
    supports = []

    for lag in range(order):
        source_mask = make_sparse_source_mask(n_sources, sparsity, rng)

        block_mask = expand_source_blocks(source_mask, block_size)

        A = np.zeros((n_state, n_state))

        # stable self dynamics only on first lag
        if lag == 0:
            A += self_decay * np.eye(n_state)

        A[block_mask] = rng.normal(scale=cross_scale, size=block_mask.sum())

        # preserve diagonal dynamics
        if lag == 0:
            np.fill_diagonal(A, self_decay)

        A_lags.append(A)
        supports.append(block_mask)

    return A_lags, supports


