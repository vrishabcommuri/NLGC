import matplotlib.pyplot as plt
import numpy as np
import scipy

def prepare_transition_matrix(A, n_sources=None, order=None):
    """
    Convert different transition representations into a single
    concatenated VAR matrix [A1 A2 ... Ap].

    Parameters
    ----------
    A : ndarray or list of ndarray
        Transition representation.
    n_sources : int, optional
        Number of sources if A is companion form.
    order : int, optional
        VAR order if A is companion form.

    Returns
    -------
    A_plot : ndarray
        Concatenated transition matrix.
    """

    if isinstance(A, list):
        # list of lag matrices
        return np.concatenate(A, axis=1)

    if n_sources is not None and order is not None:
        # companion matrix
        return A[:n_sources, :n_sources * order]

    # already concatenated
    return A


def plot_transition(
    A,
    ax=None,
    title=None,
    vmax=None,
    cmap="seismic",
):
    """
    Plot a single transition matrix.
    """

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(12,5), constrained_layout=True)

    if vmax is None:
        vmax = np.max(np.abs(A))

    im = ax.imshow(
        A,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")

    if title is not None:
        ax.set_title(title)

    return im


def plot_transition_comparison(
    A_before,
    A_after,
    n_sources=None,
    order=None,
    titles=("Ground truth", "Recovered"),
    figsize=(12, 5),
    cmap="seismic",
    show_colorbar=False,
    bind_colorbars=True,
):
    """
    Plot two transition matrices with shared color scale.

    Parameters
    ----------
    A_before, A_after : ndarray or list
        Transition matrices.
    """

    A_before = prepare_transition_matrix(
        A_before,
        n_sources=n_sources,
        order=order,
    )

    A_after = prepare_transition_matrix(
        A_after,
        n_sources=n_sources,
        order=order,
    )

    if bind_colorbars:
        vmax1 = max(
            np.max(np.abs(A_before)),
            np.max(np.abs(A_after)),
        )
        vmax2 = vmax1
    else:
        vmax1 = np.max(np.abs(A_before))
        vmax2 = np.max(np.abs(A_after))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
    )

    im = plot_transition(
        A_before,
        axes[0],
        title=titles[0],
        vmax=vmax1,
        cmap=cmap,
    )

    plot_transition(
        A_after,
        axes[1],
        title=titles[1],
        vmax=vmax2,
        cmap=cmap,
    )

    if show_colorbar:
        fig.colorbar(
            im,
            ax=axes,
            shrink=0.2,
        )

    return fig, axes

def plot_transition_blurred(A, m, order):
    A = A[:m].reshape(m, order, m).mean(axis=1)
    A = scipy.signal.convolve2d(A, np.ones((3,3)), mode='same')
    vm = np.abs(A).max()
    plt.imshow(A, cmap='seismic', vmax=vm, vmin=-vm)


def plot_transition_and_mask(A, A_mask, m, src="", targ=""):
    A = A[:m]
    A_mask = A_mask[:m]
    fig, ax = plt.subplots(1,2)
    vm = np.abs(A).max()
    ax[0].imshow(A, cmap='seismic', vmax=vm, vmin=-vm)
    ax[1].imshow(A_mask, vmax=1, vmin=-1, cmap='seismic')
    plt.title(f"{src}->{targ}")


def plot_transition_and_mask_blurred(A, A_mask, m, p, src="", targ=""):
    A = A[:m].reshape(m, p, m).mean(axis=1)
    A = scipy.signal.convolve2d(A, np.ones((3,3)), mode='same')
    A_mask = A_mask[:m].reshape(m, p, m).mean(axis=1)
    A_mask = scipy.signal.convolve2d(A_mask, np.ones((3,3)), mode='same')

    plot_transition_and_mask(A, A_mask, m, src=src, targ=targ)
    
def plot_transition_single(
    A,
    n_sources=None,
    order=None,
    titles=("Ground truth", "Recovered"),
    figsize=(10, 5),
    cmap="seismic",
    show_colorbar=False,
):
    """
    Plot two transition matrices with shared color scale.

    Parameters
    ----------
    A_before, A_after : ndarray or list
        Transition matrices.
    """

    A = prepare_transition_matrix(
        A,
        n_sources=n_sources,
        order=order,
    )

    
    vmax = np.max(np.abs(A))

    fig, axes = plt.subplots(
        1,
        1,
        figsize=figsize,
        constrained_layout=True,
    )

    im = plot_transition(
        A,
        axes,
        title=titles[0],
        vmax=vmax,
        cmap=cmap,
    )

    if show_colorbar:
        fig.colorbar(
            im,
            ax=axes,
            shrink=0.2,
        )

    return fig, axes