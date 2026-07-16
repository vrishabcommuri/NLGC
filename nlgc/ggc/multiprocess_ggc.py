import os
# prophylactic limit on threads; combat thrashing when linalg libraries 
# attempt to fork within a child process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# import eelbrain as eel
import numpy as np
import mne
import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
from spectral_connectivity import Multitaper, Connectivity
import multiprocessing
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests


def PC1(block, standardize=True, verbose=True):
    # return first principal component time series from 4 eigenmodes 
    # (or any other larger block structure)
    block = block.T
    
    if standardize:
        scaler = StandardScaler()
        block = scaler.fit_transform(block)
    
    # Fit on all trials combined to find the common spatial weight
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(block)
    
    if verbose:
        # weights assigned to each component
        print(pca.components_.round(3).flatten())
    
    return pc1.T


def apply_fdr(observed_matrix, null_matrices, alpha=0.05):
    n = observed_matrix.shape[0]
    
    # isolate off diagonals
    rows, cols = np.where(~np.eye(n, dtype=bool))
    obs_vals = observed_matrix[rows, cols]
    
    # sort null values for empirical p-value calculation
    null_pool = null_matrices[:, rows, cols].flatten()
    null_pool = np.sort(null_pool) 
    n_null = len(null_pool)
    
    # empirical p-values
    indices = np.searchsorted(null_pool, obs_vals, side='left')
    p_values = (n_null - indices + 1) / (n_null + 1)
    
    reject, p_corrected, _, _ = multipletests(
        p_values, 
        alpha=alpha, 
        method='fdr_bh'
    )
    
    # reconstruct the thresholded matrix
    significant_ggc = np.zeros((n, n))
    significant_ggc[rows, cols] = obs_vals * reject
    
    mask = np.zeros((n, n), dtype=bool)
    mask[rows, cols] = reject
    
    return significant_ggc, mask


def imshow_ggc(ggc_tf_array, frequencies, f_min=4, f_max=8):
    n_win, n_freq_total, n_src, _ = ggc_tf_array.shape
    
    f_mask = (frequencies >= f_min) & (frequencies <= f_max)
    plot_freqs = frequencies[f_mask]
    n_f_bins = len(plot_freqs)
    
    data = ggc_tf_array[:, f_mask, :, :]
    
    # we want the final 2D shape to be (n_src * n_f_bins, n_src * n_win)
    # to have 'from' (i) on the vertical axis and 'to' (j) on the horizontal:
    # 0:time, 1:freq, 2:from, 3:to
    
    # transpose to (from, freq, to, time)
    # keeps the freq-time spectrograms intact within each i,j block
    tiled_data = data.transpose(2, 1, 3, 0)
    final_image = tiled_data.reshape(n_src * n_f_bins, n_src * n_win)
    
    plt.figure(figsize=(25, 25))
    
    vmax = np.nanquantile(final_image[final_image > 0], 0.98)

    plt.imshow(final_image, aspect='auto', cmap='viridis', 
               origin='upper', vmin=0, vmax=vmax)
    
    for i in range(n_src):
        plt.axvline(i * n_win - 0.5, color='white', linestyle='-', linewidth=0.7, alpha=0.4)
        plt.axhline(i * n_f_bins - 0.5, color='white', linestyle='-', linewidth=0.7, alpha=0.4)
        
        plt.text(i * n_win + n_win/2, -5, f"To {i}", color='black', ha='center', fontsize=9)
        plt.text(-10, i * n_f_bins + n_f_bins/2, f"From {i}", color='black', va='center', rotation=90, fontsize=9)

    plt.title(f"Time-Frequency GGC Matrix: Driver (Y) to Target (X)\n(Theta Band: {f_min}-{f_max} Hz)", fontsize=20)
    plt.xlabel("Target Source (Internal: Time Windows 0-46)", fontsize=14)
    plt.ylabel("Driver Source (Internal: Frequency Bins)", fontsize=14)
    plt.colorbar(label='GC', fraction=0.02, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def plot_J_comparison(pexp, J, anat):
    cols = anat.index
    anat = anat.values
    fig, ax = plt.subplots(1, 4, figsize=(15,10))
    ax[0].imshow(anat@pexp.T@anat.T)
    ax[1].imshow(anat@J.T@anat.T)
    ax[2].imshow(pexp)
    ax[3].imshow(J)
    
    ax[0].set_xticks(range(len(cols)), cols, rotation=90)
    ax[0].set_yticks(range(len(cols)), cols)
    ax[1].set_xticks(range(len(cols)), cols, rotation=90)
    ax[1].set_yticks(range(len(cols)), cols)
    plt.show()


def _ggc(data):
    sources, seed, shuffle, modelkwargs, multitaperkwargs = data[0], \
                                        data[1], data[2], data[3], data[4]
    print(f"processing {shuffle} model with seed {seed}")

    n_times = sources.shape[1]
    n_patches = modelkwargs['n_patches']
    fs = multitaperkwargs['sampling_frequency']

    data_reshaped = np.expand_dims(sources, 1).T
    
    if shuffle:
        np.random.seed(seed)
        roll_amounts = np.random.uniform(0, n_times, size=(n_patches)).astype(int)
        for i in range(n_patches):
            data_reshaped[:, :, i] = np.roll(data_reshaped[:, :, i], roll_amounts[i], axis=0)

    multitaper = Multitaper(
        data_reshaped,
        **multitaperkwargs,
    )
    
    connectivity = Connectivity.from_multitaper(multitaper)

    print(f"{shuffle} model with seed {seed} done multitaper")
    ggc = connectivity.pairwise_spectral_granger_prediction()
    print(f"{shuffle} model with seed {seed} done gc, sending")
    return [ggc, connectivity]


class GGC():    
    def __init__(self, modelkwargs=None, multitaperkwargs=None):
        __default_modelkwargs = {
            'n_eigs': 4,
            'n_patches': 84,
            'n_nulls': 10,
            'l_freq': 4,
            'h_freq': 8,
        }

        __default_multitaperkwargs = {
            'sampling_frequency': 25,
            'time_halfbandwidth_product': 15*0.5, # duration * half_bandwidth
            'time_window_duration': 15,           # duration
            'time_window_step': 5,
            'start_time': 0,
        }

        if modelkwargs is None:
            self.modelkwargs = __default_modelkwargs
        else:
            self.modelkwargs = modelkwargs

        if multitaperkwargs is None:
            self.multitaperkwargs = __default_multitaperkwargs
        else:
            self.multitaperkwargs = multitaperkwargs


    def fit_multitaper(self, modelparams):
        n_times = modelparams[4].shape[0]
        n_eigs = self.modelkwargs['n_eigs']
        n_patches = self.modelkwargs['n_patches']
        n_nulls = self.modelkwargs['n_nulls']
    
        ei = np.array([modelparams[4][:,:n_patches*n_eigs].T[i::4] for i in range(4)])
        
        sourcepcs = np.zeros((n_patches, n_times))
        
        for i in range(n_patches):
            # don't standardize the eigenmodes since scales are meaningful
            sourcepcs[i] = PC1(ei[:, i, :], standardize=False, verbose=False)
        
        np.random.seed(0)
        sourcepcs += np.random.normal(0, 1e-10, sourcepcs.shape)
        
        observedseed = 0

        # # nulls: wilson factorization is single core, so we can do multiple at once
        with multiprocessing.get_context('spawn').Pool() as pool:
            # farm out observed + nulls
            ggcs = list(pool.imap(_ggc,  
                [(sourcepcs, observedseed, False, self.modelkwargs, self.multitaperkwargs)] + \
                [(sourcepcs, nullseed, True, self.modelkwargs, self.multitaperkwargs) for nullseed in range(n_nulls)]))

        print("done gc")
        return ggcs


    def significance_map(self, ggcs, alpha=0.1, frameno=0):
        l_freq = self.modelkwargs['l_freq']
        h_freq = self.modelkwargs['h_freq']

        observed_ggc = ggcs[0][0] # observed
        connectivity = ggcs[0][1] # connectivity object from observed (for frequencies)
        null_ggcs = np.array([i[0] for i in ggcs[1:]]) # nulls
            
        freqmask = (connectivity.frequencies > l_freq) & (connectivity.frequencies < h_freq)

        nullmats = np.nan_to_num(np.array(null_ggcs))[:, frameno, freqmask, ...].mean(axis=1)
        observedmat = np.nan_to_num(observed_ggc)[frameno][freqmask, ...].mean(axis=0)
        observedmat_broadband = np.nan_to_num(observed_ggc)[frameno].mean(axis=0)
        
        significant_results, binary_mask = apply_fdr(observedmat, nullmats, alpha=alpha)
        
        # convert nats to percent explained
        # only keep links that satisfy FDR procedure
        pexp = (1 - np.exp(-observedmat_broadband)) * binary_mask
        obs = observedmat * binary_mask
        
        return pexp, obs, binary_mask


def ggc_map(modelparams, modelkwargs=None, multitaperkwargs=None, alpha=0.1, frameno=0, J=None):
    ggc = GGC(modelkwargs=modelkwargs, multitaperkwargs=multitaperkwargs)
    ggc_mt = ggc.fit_multitaper(modelparams)
    pexp, obs, binary_mask = ggc.significance_map(ggc_mt, alpha=alpha, frameno=frameno)
    return ggc, ggc_mt, pexp, obs, binary_mask, ggc.modelkwargs, ggc.multitaperkwargs, J

