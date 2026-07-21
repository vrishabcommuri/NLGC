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
    n_sources=4,
    n_sensors=8,
    order=2,
    n_orients=1,
    sparsity=0.25,
    self_decay=0.8,
    cross_scale=0.1,
    q_scale=0.05,
    r_scale=0.05,
    seed=0,
):
    """
    generate sparse VAR(p) state-space model.

    The latent process is:

        x_t = A_1 x_{t-1}
            + A_2 x_{t-2}
            + ...
            + w_t

    represented internally using companion form.

    Returns
    -------
    ssm : SSMState

    A_lags : list[np.ndarray]
        True lag matrices.

    supports : list[np.ndarray]
        True sparse connectivity masks.
    """

    rng = np.random.default_rng(seed)

    A_lags, supports = gen_sparse_var_lags(
        n_sources=n_sources,
        order=order,
        n_orients=n_orients,
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

    F = (
        rng.normal(
            size=(n_sensors, state_dim)
        )
        / np.sqrt(state_dim)
    )
    F[:, state_dim//order:] = 0.0

    Q = np.zeros((state_dim, state_dim))

    Q[:n_sources*n_orients, :n_sources*n_orients] = (
        q_scale * np.eye(n_sources*n_orients)
    )

    R = r_scale * np.eye(n_sensors)

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

    observation_noise = rng.multivariate_normal(
        np.zeros(n_sensors),
        R,
        size=T,
    )

    y = x @ F.T + observation_noise

    ssm = SSMState(
        y=y,
        x=x,
        A=A,
        F=F,
        Q=Q,
        R=R,
        N_times=T,
        N_sources=n_sources * n_orients,
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


def expand_orientation_blocks(source_mask, n_orients):
    """
    expand source-level connectivity into orientation blocks. a source-source
    edge activates the full orientation block.

    source edge i -> j becomes [3x3 block] i -> j

    Parameters
    ----------
    source_mask : ndarray(bool)
        Shape (n_sources, n_sources)

    Returns
    -------
    block_mask : ndarray(bool)
        Shape (n_sources*n_orients,
         n_sources*n_orients)
    """
    n_sources = source_mask.shape[0]

    block_mask = np.zeros((n_sources * n_orients,
                           n_sources * n_orients), dtype=bool)

    for i in range(n_sources):
        for j in range(n_sources):
            if source_mask[i, j]:
                block_mask[
                    i*n_orients:(i+1)*n_orients,
                    j*n_orients:(j+1)*n_orients,
                ] = True

    return block_mask


def gen_sparse_var_lags(n_sources, order, n_orients=1, sparsity=0.25, 
                        self_decay=0.8, cross_scale=0.1, seed=0):
    """
    generate sparse VAR(p) lag matrices. connectivity is sparse at the source
    level and expanded to orientation blocks.

    Returns
    -------
    A_lags : list[np.ndarray]
        Lag matrices.

    support : list[np.ndarray]
        Boolean support masks.
    """

    rng = np.random.default_rng(seed)

    n_state = n_sources * n_orients

    A_lags = []
    supports = []

    for lag in range(order):
        source_mask = make_sparse_source_mask(n_sources, sparsity, rng)

        block_mask = expand_orientation_blocks(source_mask, n_orients)

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


