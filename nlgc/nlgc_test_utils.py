import numpy as np
import scipy
from scipy import linalg
import os
import time
from .nlgc_utils import gc_extraction, NLGC
from .opt import NeuraLVAR
from .utils.leadfield import prepare_eigenmodes
from .utils.transforms import surface_ico4_to_surface_eigs
from .config import ModelConfig
from .utils.initialize import initialize_em_state
from mne.minimum_norm import apply_inverse, make_inverse_operator, InverseOperator
from mne.source_space import SourceSpaces
from mne.utils import (logger, _check_option, _validate_type)
from mne.inverse_sparse.mxne_inverse import _prepare_gain
from mne import (Forward, Label)
from mne.forward import is_fixed_orient
import warnings
warnings.filterwarnings('ignore')
from .ggc.multiprocess_ggc import GGC, ggc_map
import matplotlib.pyplot as plt
import mne
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import json
import os
import zipfile
from matplotlib import patches
from matplotlib.pyplot import axvline, axhline
from collections import defaultdict

def find_poles_and_zeros(a_true, model, order):
    A = model._model_f[0]._parameters[0]
    mask = np.abs(a_true).mean(axis=0)
    mask = mask > 0.1

    locx, locy = np.nonzero(mask)
    a_t = []
    b_t = []
    a_s = []
    b_s = []

    for i in range(order):
        a_t.append(a_true[i][locx, locy])
        b_t.append(a_true[i][locx, locx])
        a_s.append(A[i][locx, locy])
        b_s.append(A[i][locx, locx])

    a_t = np.array(a_t)
    b_t = np.array(b_t)
    a_s = np.array(a_s)
    b_s = np.array(b_s)

    zs, ps = [], []
    zs_t, ps_t = [], []

    for i in range(len(a_s[0])):
        bt = [1] + (-b_t[:,i]).tolist()
        at = a_t[:,i].tolist()
        zt, pt, kt = scipy.signal.tf2zpk(at, bt)
        zs_t.extend(zt)
        ps_t.extend(pt)


        bsig = [1] + (-b_s[:,i]).tolist()
        asig = a_s[:,i].tolist()
        zsig, psig, ksig = scipy.signal.tf2zpk(asig, bsig)
        zs.extend(zsig)
        ps.extend(psig)

    return zs_t, ps_t, zs, ps


def zplane(z, p, title, lim=1):


    """Plot the complex z-plane given zeros and poles.
    """
    
    # get a figure/plot
    fig = plt.figure(figsize=(10,10))

    ax = plt.subplot(2, 2, 1)
    # TODO: should just inherit whatever subplot it's called in?

    # Add unit circle and zero axes    
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.1)
    ax.add_patch(unit_circle)
    axvline(0, color='0.7')
    axhline(0, color='0.7')
    
    # Plot the poles and set marker properties
    poles = plt.plot(p.real, p.imag, 'x', markersize=9, alpha=0.5)
    
    # Plot the zeros and set marker properties
    zeros = plt.plot(z.real, z.imag,  'o', markersize=9, 
             color='none', alpha=0.5,
             markeredgecolor=poles[0].get_color(), # same color as poles
             )

    # Scale axes to fit
    r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    plt.axis('scaled')
    if lim is None:
        plt.axis([-r, r, -r, r])
    else:
        plt.axis([-lim, lim, -lim, lim])
    
    
#    ticks = [-1, -.5, .5, 1]
#    plt.xticks(ticks)
#    plt.yticks(ticks)

    """
    If there are multiple poles or zeros at the same point, put a 
    superscript next to them.
    TODO: can this be made to self-update when zoomed?
    """
    # Finding duplicates by same pixel coordinates (hacky for now):
    poles_xy = ax.transData.transform(np.vstack(poles[0].get_data()).T)
    zeros_xy = ax.transData.transform(np.vstack(zeros[0].get_data()).T)    

    # dict keys should be ints for matching, but coords should be floats for 
    # keeping location of text accurate while zooming

    # TODO make less hacky, reduce duplication of code
    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in poles_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )

    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in zeros_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )
    plt.title(title, fontsize = 20)
    return fig

# Save information in 
def save_info(dir, a, JG, model, order, param_dict, ggc_model = None, J_GGC = None, ggc_model_extras = None,  zip_pkl = True, debug_report = False):

    conv = int(np.floor((5/350)*a.shape[1]) + 1)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory '{dir}' created.")
    else:
        print(f"Directory '{dir}' already exists.")

    with PdfPages(dir + 'model-comparison.pdf') as pdf:
        plt.figure(figsize=(75, 75))
        arr = np.concatenate(a[:], axis = 1)
        plt.imshow(scipy.signal.convolve2d(arr, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
        plt.title('A Coefficients Concatenated', fontsize = 80)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        # if LaTeX is not installed or error caught, change to `False`
        plt.figure(figsize=(75, 75))
        model_params = model._model_f[0]._parameters[0]
        arr_model = np.concatenate(model_params[:], axis = 1)
        plt.imshow(scipy.signal.convolve2d(arr_model, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
        plt.title('Derived Model Parameters Concatenated', fontsize = 80)
        # pdf.attach_note("plot of sin(x)")  # attach metadata (as pdf note) to page
        pdf.savefig()
        plt.close()

        
        fig = plt.figure(figsize=(75, 75))
        plt.imshow(JG)
        plt.title('Ground Truth J Statistics', fontsize = 80)
        pdf.savefig(fig)  
        plt.close()
        
        fig = plt.figure(figsize=(75, 75))
        plt.imshow(model.get_J_statistics())
        plt.title('Derived J Statistics', fontsize = 80)
        pdf.savefig(fig)  
        plt.close()


        # if ggc_model != None:
        #     fig = plt.figure(figsize=(75, 75))
        #     plt.imshow(J_GGC)
        #     plt.title('Ground Truth GGC J Statistics', fontsize = 80)
        #     pdf.savefig(fig)  
        #     plt.close()

            
        negated_identity = np.abs(np.eye(a.shape[1]) - 1)

        a_abs = np.abs(a[:]*negated_identity)
        a_summed = np.sum(a_abs[:], axis = 0)
        fig = plt.figure(figsize=(75, 75))
        plt.imshow(scipy.signal.convolve2d(a_summed, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
        plt.title('No Diagonal Absolute Summed Lags A Coeffs', fontsize = 80)
        pdf.savefig(fig)  
        plt.close()


        model_params_abs = np.abs(model_params[:]*negated_identity)
        model_params_summed = np.sum(model_params_abs[:], axis = 0)
        fig = plt.figure(figsize=(75, 75))
        
        plt.imshow(scipy.signal.convolve2d(model_params_summed, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
        plt.title('No Diagonal Absolute Summed Lags Derived Model Params', fontsize = 80)
        pdf.savefig(fig)  
        plt.close()


        zs_t, ps_t, zs, ps = find_poles_and_zeros(a, model, order)

        fig = zplane(np.array(zs_t), np.array(ps_t), 'Ground Truth Pole Zero Plot')
        pdf.savefig(fig)
        plt.close()

        fig = zplane(np.array(zs),np.array(ps), 'Model Parameters Pole Zero Plot')
        pdf.savefig(fig)
        plt.close()


        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Model Analyatics PDF'
        d['Author'] = 'Kavin Loganathan'

        model_path = dir + 'model.pkl'
        A_path = dir + 'G-Coeffs.pkl'
        JG_path = dir + 'JG.pkl'

        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        with open(A_path, 'wb') as file:
            pickle.dump(a, file)
        with open(JG_path, 'wb') as file:
            pickle.dump(JG, file)


        if ggc_model != None:
            ggc_model_path = dir + 'ggc_model.pkl'
            J_GGC_path = dir + 'J_GGC.pkl'
            ggc_model_extras_path = dir + 'ggc_model_extras.pkl'
            with open(ggc_model_path, 'wb') as file:
                pickle.dump(ggc_model, file)
            with open(J_GGC_path, 'wb') as file:
                pickle.dump(J_GGC, file)
            with open(ggc_model_extras_path, 'wb') as file:
                pickle.dump(ggc_model_extras, file)

        with zipfile.ZipFile(dir + "data.zip", "w") as zip_file:
            zip_file.write(model_path, arcname="model.pkl")
            zip_file.write(A_path, arcname="G-Coeffs.pkl") 
            zip_file.write(JG_path, arcname= 'JG.pkl')
            if ggc_model != None:
                zip_file.write(ggc_model_path, arcname='ggc_model.pkl')
                zip_file.write(J_GGC_path, arcname='J_GGC.pkl')
                zip_file.write(ggc_model_extras_path, arcname='ggc_model_extras.pkl')

        os.remove(model_path)
        os.remove(A_path)
        os.remove(JG_path)
        if ggc_model != None:
            os.remove(ggc_model_path)
            os.remove(J_GGC_path)
            os.remove(ggc_model_extras_path)

        with open(dir + "params.json", "w") as f:
            json.dump(param_dict, f, indent=4)
        




'''
Params:
seed: random generator seed
band: what freq band the signal lies in: beta, theta, delta, wide
fs: sampling frequency
natures: Possible natures of the GC links

Optional:
f_opt_fname: Patient/Subject Name
empty_room_fname: Name of files for empty_room raw fif
path: path to folder of MRI data of all patients

Outputs:
f: lead-field matrix
y: simulated sensor data
r_cov: forward eqn covariance matrix
p: model order
JG: Ground Truth GC Links

pow_actives: power of sources
a: ground truth a matrix which VAR model is trying to estimate, contains GC links

'''




    
# Assume folder setup follows eelbrain pipeline
def lead_field_generation(root, subject_id, src_space, n_eigenmodes, n_orients, loose=0.0, depth=0.0, pca=True, rank=None, trans = None,
                          vol_pos_origin=10.0, vol_pos_target=30.0):
    
    full_empty_room_path = root + "meg/" + subject_id + "/" + subject_id + "_emptyroom-raw.fif"
    raw_empty_room = mne.io.read_raw_fif(full_empty_room_path)
    info = raw_empty_room.info
    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
    if trans == None:
        expected_trans_file= root + "meg/" + subject_id + "/" + subject_id + "-trans.fif"
        if (os.path.exists(expected_trans_file)):
            print(" trans file found")
            trans_file = expected_trans_file
        else: 
            print("No trans file provided using mne fsaverage instead")
            trans_file = "fsaverage"
    else:
        trans_file = trans
    bem_folder= root + "/mri/" + subject_id + "/bem/"
    print(bem_folder)
    subjects_dir = root + "/mri/"
    print(src_space)
    if src_space == 'surf':
        fwd_origin = mne.make_forward_solution(info, trans_file, src = bem_folder + subject_id + "-ico-4-src.fif",
                                            bem = bem_folder + subject_id  + "-inner_skull-bem-sol.fif")
        fwd_target = mne.make_forward_solution(info, trans_file, src = bem_folder + subject_id  + "-ico-1-src.fif",
                                            bem = bem_folder + subject_id + "-inner_skull-bem-sol.fif")
        fwd_origin = mne.convert_forward_solution(
            fwd_origin,
            surf_ori=True,      # align dipoles to cortical surface normals
            force_fixed=True,   # reduce to 1 orientation per source (fixed)
            use_cps=True
        )
        fwd_target = mne.convert_forward_solution(
            fwd_target,
            surf_ori=True,      # align dipoles to cortical surface normals
            force_fixed=True,   # reduce to 1 orientation per source (fixed)
            use_cps=True
        )
    elif src_space == 'vol' or src_space == 'mixed':
        # inner_skull.surf is watershed output and is not always present; the BEM
        # surfaces carry the same boundary, so fall back to those when it is missing.
        inner_skull_surf = bem_folder + 'inner_skull.surf'
        if os.path.exists(inner_skull_surf):
            vol_bounds = dict(surface=inner_skull_surf)
        else:
            bem_file = bem_folder + subject_id + "-inner_skull-bem.fif"
            if not os.path.exists(bem_file):
                raise FileNotFoundError(
                    f'Need either {inner_skull_surf} or {bem_file} to bound the '
                    f'volume source space')
            print(f'inner_skull.surf not found, bounding volume with {bem_file}')
            vol_bounds = dict(bem=bem_file)
        src_origin = mne.setup_volume_source_space(subject = subject_id, pos = vol_pos_origin, subjects_dir = subjects_dir, **vol_bounds)
        src_target = mne.setup_volume_source_space(subject = subject_id, pos = vol_pos_target, subjects_dir = subjects_dir, **vol_bounds)
        if src_space == 'mixed':
            surf_src = mne.setup_source_space(subject = subject_id, spacing = 'ico4', surface = 'white', subjects_dir = subjects_dir, add_dist = 'patch', verbose = None)
            src_origin = surf_src + src_origin
        
        fwd_origin = mne.make_forward_solution(info = info, trans = trans_file, src = src_origin, bem = bem_folder + subject_id + "-inner_skull-bem-sol.fif", ignore_ref = True)
        fwd_target = mne.make_forward_solution(info = info, trans = trans_file, src = src_target, bem = bem_folder + subject_id + "-inner_skull-bem-sol.fif", ignore_ref = True)
    # fwd_origin_data = fwd_origin['sol']
    weights, G, label_vertidx, label_names, gain_info, whitener = prepare_eigenmodes(info, fwd_origin, noise_cov, fwd_target, n_eigenmodes=n_eigenmodes, n_orients = n_orients, loose=loose, depth=depth, pca=pca, rank=rank,
    mode='svd_flip')
    print(f'G shape: {G.shape}')
    return G, info, noise_cov, fwd_origin, weights



def _voxel_block(v, n_orient=3):
    """Return slice for RAS components of voxel v."""
    return slice(n_orient * v, n_orient * (v + 1))


def _make_3d_coupling(strength=0.15, mode="mixed", rng=None):
    """
    Create a 3x3 voxel-to-voxel coupling block.
    Rows = target RAS components
    Cols = source RAS components
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "same_axis":
        B = np.eye(3)

    elif mode == "cross_axis":
        # R->A, A->S, S->R style cross-orientation coupling
        B = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

    elif mode == "mixed":
        B = rng.normal(size=(3, 3))
        B /= np.linalg.norm(B, ord="fro") + 1e-12

    else:
        raise ValueError(f"Unknown coupling mode: {mode}")

    return strength * B


def _companion_spectral_radius(a):
    """
    Compute spectral radius of VAR companion matrix.
    a shape: (p, m, m)
    x[t] = sum_k a[k] @ x[t-k-1] + u[t]
    """
    p, m, _ = a.shape

    companion = np.zeros((p * m, p * m), dtype=float)
    companion[:m, :] = np.concatenate(a, axis=1)

    if p > 1:
        companion[m:, :-m] = np.eye((p - 1) * m)

    eigvals = np.linalg.eigvals(companion)
    return np.max(np.abs(eigvals))


def _stabilize_var(a, target_spec_rad=0.9, max_iter=20):
    """
    Rescale VAR coefficients if companion spectral radius is too large.
    """
    a = a.copy()

    for _ in range(max_iter):
        rho = _companion_spectral_radius(a)

        if rho < target_spec_rad:
            return a, rho

        scale = target_spec_rad / (rho + 1e-12)
        a *= 0.98 * scale

    rho = _companion_spectral_radius(a)
    return a, rho

def _voxel_mode_block(v, n_eigenmodes, n_orients=3):
    block_size = n_eigenmodes * n_orients
    return slice(v * block_size, (v + 1) * block_size)


def vol_data_generation(seed=0, band="wide", fs=50, natures="all", n_eigenmodes = 2, G=None, p=2, t=500, n_active_voxels=10, n_links=10, target_spec_rad=0.9, coupling_mode="mixed",
    process_noise_active=0.1, process_noise_inactive=0.001, measurement_noise_scale=1e2, plot_psd=False, n_orients=3, verbose=False):

    if p < 1:
        raise ValueError("p should be at least 1")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # taken as a parameter so the ground truth and the leadfield can never
    # silently disagree about the orientation count
    if n_orients != 3:
        raise NotImplementedError(
            f'vol_data_generation assumes 3 orientations per voxel, got {n_orients}')

    if G is None:
        
        n_sensors = 156
        n_voxels = int(np.floor(2*n_sensors/(n_eigenmodes*n_orients)))
        m = n_eigenmodes * n_orients * n_voxels

        f = rng.normal(size=(n_sensors, m))
        f /= np.sqrt(np.sum(f ** 2, axis=0, keepdims=True)) + 1e-12

    else:
        f = G
        n_sensors, m = f.shape

        if m % n_orients != 0:
            raise ValueError(
                f"G has {m} columns, which is not divisible by 3. "
                "Expected G.shape = (n_sensors, 3*n_voxels)."
            )

        n_voxels = m // (n_eigenmodes * n_orients)

    # active units are (voxel, eigenmode) pairs -- see the rng.choice below,
    # which draws from n_voxels * n_eigenmodes
    n_active_units = n_voxels * n_eigenmodes
    if n_active_voxels > n_active_units:
        raise ValueError(
            f"n_active_voxels ({n_active_voxels}) cannot exceed "
            f"n_voxels * n_eigenmodes ({n_active_units})")

    if verbose:
        print(f"G shape is {f.shape}")
        print(f"n_sensors {n_sensors}, n_voxels {n_voxels}, "
              f"state dimension m = {m}")

    burnin = max(200, 10 * fs, 10 * p * m)

    q = process_noise_inactive * np.eye(m)
    a = np.zeros((p, m, m), dtype=np.float64)

    active_voxels = rng.choice(n_active_units, size=n_active_voxels, replace=False)
    if verbose:
        print(f"Active (voxel, eigenmode) units: {active_voxels}")

    if band == "wide":
        for v in active_voxels:
            block = _voxel_block(v, n_orients)

            q[block, block] = process_noise_active * np.eye(n_orients)

            # Independent AR(1)-style self-history for R/A/S
            a[0, block, block] = target_spec_rad * np.eye(n_orients)

    else:
        band_dict = {
            "delta": (0.1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (13, 23),
        }

        if band not in band_dict:
            raise ValueError(f"band {band} not implemented")

        f_low, f_high = band_dict[band]

        for v in active_voxels:
            block = _voxel_block(v, n_orients)

            q[block, block] = process_noise_active * np.eye(n_orients) # Look into making this randn but make sure that the array is positive definite and symmetric

            f0 = rng.uniform(f_low, f_high)
            w0 = 2 * np.pi * f0 / fs

            # AR(2) oscillator per RAS component
            a[0, block, block] = target_spec_rad * 2 * np.cos(w0) * np.eye(n_orients)
            a[1, block, block] = -(target_spec_rad ** 2) * np.eye(n_orients)

    link_power = target_spec_rad / 2
    if verbose:
        print(f"Link Power {link_power}")

    links = []

    for link_idx in range(n_links):
        source_v, target_v = rng.choice(active_voxels, size=2, replace=False)

        source_block = _voxel_block(source_v, n_orients)
        target_block = _voxel_block(target_v, n_orients)

        links.append((target_v, source_v))

        for lag in range(p):
            strength = rng.uniform(0.05, link_power)

            B = _make_3d_coupling(
                strength=strength,
                mode=coupling_mode,
                rng=rng,
            )

            if natures == "all":
                pass

            elif natures == "excitatory":
                B = np.abs(B)

            elif natures == "inhibitory":
                B = -np.abs(B)

            elif natures == "sharpening1":
                if lag % 2 == 0:
                    B = np.abs(B)
                else:
                    B = -np.abs(B)

            elif natures == "sharpening2":
                if lag % 2 == 0:
                    B = -np.abs(B)
                else:
                    B = np.abs(B)

            else:
                raise ValueError(f"nature {natures} not implemented")

            a[lag, target_block, source_block] = B

    # Stabilize full VAR after adding off-diagonal 3D couplings
    a, rho = _stabilize_var(a, target_spec_rad=target_spec_rad)
    if verbose:
        print(f"Final companion spectral radius: {rho:.4f}")

    # Ground-truth voxel-level GC matrix
    temp_JG = np.sum(np.abs(a), axis=0)
    JG = np.zeros((n_voxels, n_voxels), dtype=bool)

    for target_v in range(n_voxels):
        target_block = _voxel_mode_block(target_v, n_eigenmodes, n_orients)

        for source_v in range(n_voxels):
            source_block = _voxel_mode_block(source_v, n_eigenmodes, n_orients)

            if target_v == source_v:
                continue

            JG[target_v, source_v] = temp_JG[target_block, source_block].sum() > 0

    T = burnin + t

    u = rng.standard_normal((T, m))
    L = linalg.cholesky(q, lower=True)
    u = u @ L.T

    x = np.zeros((T, m), dtype=np.float64)

    for tt in range(p, T):
        x[tt] = u[tt]

        for lag in range(p):
            x[tt] += a[lag] @ x[tt - lag - 1]

    x = x[burnin:]

    if verbose:
        print("band", band, x.shape)

    if plot_psd:
        for comp in range(x.shape[1]):
            plt.psd(x[:, comp], Fs=fs)
        plt.title("Source/RAS component PSDs")
        plt.show()

    pow_actives = [np.mean(x[:, i] ** 2) for i in range(m)]

    if verbose:
        print(f"f: max {np.max(f):.4g} mean {np.mean(f):.4g} "
              f"median {np.median(f):.4g} min {np.min(f):.4g}")
        print(f"x: max {np.max(x):.4g} median {np.median(x):.4g}")

    y = x @ f.T

    px = np.trace(y @ y.T)

    noise = rng.standard_normal(y.shape)
    pn = np.trace(noise @ noise.T)

    multiplier = measurement_noise_scale * pn / (px + 1e-12)

    if verbose:
        print(f"pn {pn:.4g}, px {px:.4g}, multiplier {multiplier:.4g}, "
              f"r_cov {1/multiplier:.4g}")

    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return f, y, x, r_cov, p, JG, pow_actives, a


def data_generation(seed=0, band='wide', fs=50, natures='all', n_eigenmodes = 2, G = None, p = 2, t = 500, m_active = 10, n_links = 10, target_spec_rad = .9):
    print(f't is {t}')
    if p < 1:
        raise Exception('p should be at least 1')
    np.random.seed(seed)
    if (type(G) == type(None)):
        n = 156 # number of sensors
        
        n_patches = int(np.floor(2*n/n_eigenmodes))
        print(f'n_patches is {n_patches}')
        m = n_patches*n_eigenmodes # number of sources

        
        # 4*4*1  x*1000
        # 168*168*2 155*60*50
    else:
        n, m = G.shape

        print(f'G shape is {G.shape}')

        n_patches = m // (n_eigenmodes)

    burnin = max(200, 10*fs, 10* p * m)
    
    q = 0.001*np.eye(m)
    a = np.zeros((p, m, m), dtype=np.float64)
    idx_i = np.random.randint(0, m, size = m_active)

    print(f'n_patches is {n_patches}')
    print(f'A shape is {a.shape}')
    print(f' idx_i is {idx_i}')


    if band == 'wide':
        for ii in idx_i:
            q[ii, ii] = .1
            a[0, ii, ii] = target_spec_rad
    else:

        band_dict = {
            'delta': (0.1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 23)
        }

        if band not in band_dict:
            raise Exception(f'band {band} not implemented')
        
        f_low, f_high = band_dict[band]

        f0 = np.random.uniform(f_low, f_high, size=m_active)
        for ii, f in zip(idx_i, f0):
            w0 = 2 * np.pi * f / fs

            q[ii, ii] = .1
            a[0, ii, ii] = target_spec_rad*2*np.cos(w0)
            print(f'Pole {target_spec_rad*2*np.cos(w0)}')
            a[1, ii, ii] = -(target_spec_rad**2)
    link_power = target_spec_rad/2
    print(f'Link Power {link_power}')
    # (i,j) pairs to add a link to
    i_idx = np.random.randint(0, m_active, n_links)
    j_idx = np.random.randint(0, m_active - 1, n_links)
    j_idx += (j_idx >= i_idx)  # to prevent self-links, if j >= i, add 1 to j
    for i, j in zip(i_idx, j_idx):
        # (i,j) pair has a random link nature
        if natures == 'all':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(-link_power, link_power)
        elif natures == 'excitatory':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, link_power)
        elif natures == 'inhibitory':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -link_power)
        elif natures == 'sharpening1':
            for k in range(p):
                if k % 2 == 0:
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, link_power)
                else:      
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -link_power) 
        elif natures == 'sharpening2':
            for k in range(p):
                if k % 2 == 0:
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -link_power)
                else:      
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, link_power) 
        else:
            raise Exception(f"nature {natures} not implemented")

    temp_JG = np.sum(np.abs(a), axis=0)
    JG = temp_JG != 0
    np.fill_diagonal(JG, 0)
    JG_ = JG.copy()
    JG = np.zeros((n_patches, n_patches))
    for ei in range(n_eigenmodes):
        for ej in range(n_eigenmodes):
            JG += JG_[ei::n_eigenmodes, ej::n_eigenmodes]
    
    T = burnin + t
    u = np.random.standard_normal(m * T)
    u.shape = (T, m)
    
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)

    print(f'Max of u {np.max(u)}')
    # u /= np.sqrt(np.sum(u ** 2, axis=0))
    print(f'Max of u scaled{np.max(u)}')
    print(f'Med of u {np.median(u)}')

    print(f'Max of a scaled {np.max(a)}')
    x = np.zeros((T, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0
    for i in range(p, T):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])

    x = x[burnin:]

    print('band', band, x.shape)
    for i in range(x.T.shape[0]):
        plt.psd(x.T[i], Fs=fs)
    plt.show()
    
    pow_actives = []
    for i in range(m):
        pow_actives.append(np.mean(x[:, i]**2))

    if (type(G) == type(None)):
        f = np.random.randn(n, m)
        f /= np.sqrt(np.sum(f ** 2, axis=0))
        
        
    else:
        f = G
    print(f'Max of f {np.max(f)}')
    print(f'Mean of f {np.mean(f)}')
    print(f'Median of f {np.median(f)}')
    print(f'min of f {np.min(f)}')

    print(f'max of x {np.max(x)}')
    print(f'median of x {np.median(x)}')
    y = x.dot(f.T)
    px = y.dot(y.T).trace()

    noise = np.random.standard_normal(y.shape)
    
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2 * pn / px
    print(f'pn {pn}')
    print(f'px {px}')

    print(f'Multiplier {multiplier}')
    print(f'Max noise/sqrt measurement noise {np.max(noise/np.sqrt(multiplier))}')
    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return f, y, x, r_cov, p, JG, pow_actives, a


'''
Params: 
M = y.T = simulated sensor data
G = f = lead-field matrix
r = covariance_matrix for forward equation  
order = VAR model order
self_history = num of removed self history logs to prevent overfitting of VAR model
lambda_range = Values to check for L1 regularization
n_segments = num of segments to divide MEG recording for non-centrality param estimation
var_thr = threshold to limit num of reduced models by considering only possible links between sources that reach a
certain threshold of power
max_iter = num of EM iters to converge on param estimation
max_cyclic_iter = num of FASTA iters per EM iter
tol = tolerance for EM convergence
sparsity_factor = threshold to remove reduced models with very small VAR coefficients

xs_init: 
    initializes eigenmode time sources to this for warm start
use_es: bool
        Default: False, Use estimation stability
verbose: bool
        Default: False, Run GC extraction with verbose mode or not
'''
def nlgc_map_opt(M, G, r, order, self_history=None, var_thr=1.0, n_segments=1, lambda_range=None, max_iter=500,
                 max_cyclic_iter=3, tol=1e-5, sparsity_factor=0.0, cv=5, n_eigenmodes = 2, n_orients = 1, xs_init = None, a_init = None, use_es = False, patch_idx = None, verbose = False,
                 parallel_mode = 'serial', n_devices = 1, n_workers = 1, n_warmup_iter = 25,
                 use_wald_screen = True, wald_screen_alpha = 0.05,
                 use_empirical_null = True):
    n_sensors, nnx = G.shape
    len_patch_idx = nnx // (n_eigenmodes * n_orients)
    _, t = M.shape
    tt = t // n_segments

    # The refactored core takes a ModelConfig plus a companion-form EMState rather
    # than the old kwarg list; build them here so run_GT_sim's interface is unchanged.
    config = ModelConfig.from_legacy_kwargs(dict(
        order=order, self_history=self_history, n_eigenmodes=n_eigenmodes,
        n_orients=n_orients, n_segments=n_segments, var_thr=var_thr,
        sparsity_factor=sparsity_factor, lambda_range=lambda_range,
        max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
        cv=cv, use_es=use_es, parallel_mode=parallel_mode,
        n_devices=n_devices, n_workers=n_workers,
        n_warmup_iter=n_warmup_iter,
        use_wald_screen=use_wald_screen,
        wald_screen_alpha=wald_screen_alpha,
        use_empirical_null=use_empirical_null,
        patch_idx=tuple(patch_idx) if patch_idx is not None else (),
        verbose=verbose))

    d_raw = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    bias_r = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    bias_f = np.zeros((n_segments, 1))
    conv_flag = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    models = []
    ROI_list = list(range(len_patch_idx))
    if patch_idx is not None:
        ROI_list = patch_idx

    for seg in range(0, n_segments):
        if verbose:
            print('Segment: ', seg + 1)

        y_seg = M[:, seg * tt: (seg + 1) * tt]   # (n_sensors, n_times)

        # Keyword args: initialize_em_state's signature is (y, F, r, config, ...),
        # so the positional call used in nlgc_map lands `evoked` in `config`.
        # It expects sensor-major y -- data_driven_Q_init does U.T @ y with U
        # shaped (n_sensors, n_sensors).
        F_companion, R_companion, em_state = initialize_em_state(
            y=y_seg, F=G, r=r, config=config)

        if xs_init is not None:
            # EMState has no `smoothed_state` field (nlgc/opt/em.py) -- assigning
            # one just sets an instance attribute that jax's tree flattening never
            # reads, so this used to be a silent no-op.
            raise NotImplementedError(
                'xs_init is not supported; warm starting goes through '
                'config.optimizer.warm_start and '
                'nlgc.utils.warm_start.warm_start_sources')
        if a_init is not None:
            # a_init is (order, m, m); em_state.A is the (m*p, m*p) companion,
            # whose top block row holds [A_1 ... A_p] raveled along the columns.
            a_ravelled = NeuraLVAR._ravel_a(np.asarray(a_init))
            if a_ravelled.shape != em_state.A[:nnx].shape:
                raise ValueError(
                    f'a_init ravels to {a_ravelled.shape}, expected '
                    f'{em_state.A[:nnx].shape} for order={order}, m={nnx}')
            em_state.A[:nnx] = a_ravelled

        # gc_extraction wants TIME-major y: it feeds the kalman layer, which
        # asserts y.shape[1] == F.shape[0] (see nlgc/test/test_gc.py, which
        # passes ssm.y built as x @ F.T). Opposite of initialize_em_state above.
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            gc_extraction(y_seg.T, F_companion, R_companion,
                          ROIs=ROI_list, em_state=em_state, config=config)
        d_raw[seg] = d_raw_
        bias_r[seg] = bias_r_
        bias_f[seg] = bias_f_
        models.append(model_f)
        conv_flag[seg] = conv_flag_

    nlgc_obj = NLGC('Simulation_rnd', len_patch_idx, n_sensors, t, order,
                    n_eigenmodes, n_orients, n_segments, d_raw, bias_f, bias_r,
                    models, conv_flag, [], [], None, None, None, None)

    return nlgc_obj

'''
Params:
lead_field_gen: Boolean 
    decides if you want to load a seperate lead field matrix from existing MRI files
lf: 
    lead_field matrix you can pass in directly if you have a pre-loaded lead-field that has been whitened already
seed: int
    random seed to be used throughout
band: str
    band to generate ground truth data on, can be wide, delta, theta, alpha, beta
fs: int
    sampling frequency
natures: str
    'link nature types that are acceptable to be generated, can be all, excitatory, inhibitory, sharpening1, sharpening2



order: int
    VAR model order
t: int
    number of timestamps to generate for ground truth
n_eigenmodes: int
    number of eigenvectors to use for lead field matrix when doing low rank approx
n_segments: int
    number of segments which divides the MEG recording into equal parts for non-centrality parameter estimation
{loose, depth, pca, rank}: float/boolean
    forward model computation parameters, check mne.inverse_sparse.mxne_inverse for more inf
lambda_range: numpy 1d array
        an array of the regularization coefficients for cross-validation
max_iter: int
    maximum number of iterations for EM-based parameter estimation
max_cyclic_iter: int
    maximum number of cyclic iterations to update VAR coefficients (A's) and covariance (q's)
tol: float
    tolerance for EM convergence (in terms of relative jump of log-likelihood function)
sparsity_factor: float
    the threshold to remove reduced models with sufficiently small VAR coefficients in their corresponding
    full models for speeding up the calculations (None = all possible reduced models)
cv: int
    number of folds used for cross-validation
var_thr: float
        the threshold to limit the number of reduced models by considering only the possible links between the active
        sources which explain 'var_thr' of the total power
        (default = 1, i.e., all sources)
alpha: int | float
        Inv-Gamma(alpha*t/2 - 1, beta*t) prior on the state noise covariance matrix
m_active: int
        Number of active sources in the generated data
n_links: int
        Number of links present in active sources, this must be less than or equal to m_active
warm_start: bool
        Default: False, if you want to use initialize xs_init 
self_history: int
        Number of lags that you want to use <= order, default is order
passed_evoked: dict
        Dictionary of file path to the following: noise_cov, fwd, evoked, src_target. 
        Allows you to directly do _prep_eigenmodes rather than generate these values using lead_field_gen function
use_es: bool
        Default: False, Use estimation stability
verbose: bool
        Default: False, Run GC extraction with verbose mode or not
save_dir: string
        Default: None, Pass in directory for saving analytics and model/data
'''
def run_GT_sim(lead_field_gen = False, lf = None, src_space = 'surf', seed = 0, band = "wide", fs = 50, natures = 'all', target_spec_rad = .45,
        root = None, subject_id = None, session_name = None, trans = None, order = 2, t = 500, n_eigenmodes = 1, n_orients = 1,
        n_segments = 1, loose = 0.0, depth = 0.0, pca = True, rank = None, lambda_range = None,
        max_iter = 500, max_cyclic_iter = 3, tol = 1e-5, sparsity_factor = 0.0, cv = 5 ,var_thr = 1.0, alpha = .1, 
        m_active = 10, n_links = 10, warm_start = False, self_history = None, passed_evoked = None, use_es = False, 
        verbose = False, diff_lf = False, patch_idx = None, a_init = None, save_dir = None, run_ggc = False, ggc_kwargs = None,
        parallel_mode = 'serial', n_devices = 1, n_workers = 1, n_warmup_iter = 25,
        use_wald_screen = True, wald_screen_alpha = 0.05,
        use_empirical_null = True,
        vol_pos_origin = 10.0, vol_pos_target = 30.0, debug_report = False):

    if src_space not in ['surf', 'vol', 'mixed']:
        raise Exception(f'src_space {src_space} not implemented')

    if lambda_range is not None:
        _lams = ((lambda_range,) if isinstance(lambda_range, (int, float))
                 else lambda_range)
        if any(l <= 0 for l in _lams):
            raise ValueError(
                f'only positive lambdas are allowed, got {lambda_range}')

    if (passed_evoked != None):
        print('using passed in evoked')
        noise_cov = mne.read_cov(passed_evoked['noise_cov'])
        fwd = mne.read_forward_solution(passed_evoked['fwd'])
        evoked = mne.read_evokeds(passed_evoked['evoked'])
        src_target = mne.read_source_spaces(passed_evoked['src_target'])
        info = evoked[0].info
        weights, G, label_vertidx, label_names, gain_info, whitener = prepare_eigenmodes(info, fwd, noise_cov, src_target, 
                                                                            n_eigenmodes=n_eigenmodes, n_orients = n_orients, loose=loose, depth=depth, pca=pca, rank=rank, mode='svd_flip')
    elif (lead_field_gen):
        # keyword args: the positional form silently shifted n_orients<-loose and
        # dropped `trans`, so the real trans file was never used
        G, info, noise_cov, fwd, weights = lead_field_generation(
            root=root, subject_id=subject_id, src_space=src_space,
            n_eigenmodes=n_eigenmodes, n_orients=n_orients, loose=loose,
            depth=depth, pca=pca, rank=rank, trans=trans,
            vol_pos_origin=vol_pos_origin, vol_pos_target=vol_pos_target)
    elif (type(lf) != type(None)):
        print('Using passed in lead field')
        G = lf
    else:
        G = None
    if src_space == 'surf':
        f, y, x, r_cov, p, JG, pow_actives, a = data_generation(seed, band, fs, natures, n_eigenmodes, G, order, t, m_active, n_links, target_spec_rad)
    else:
        f, y, x, r_cov, p, JG, pow_actives, a = vol_data_generation(seed = seed, band = band, fs = fs, natures = natures, n_eigenmodes = n_eigenmodes, G = G, p = order, t = t
                                                                    ,n_active_voxels = m_active, n_links = n_links, target_spec_rad = target_spec_rad, n_orients = n_orients,
                                                                    verbose = verbose)
    if verbose:
        print(f"ground truth a: max {np.max(a):.4g} min {np.min(a):.4g} "
              f"median {np.median(a):.4g}")
        print(f"Completed data gen; {int(JG.sum())} ground truth links "
              f"across {JG.shape[0]} ROIs")
        plt.imshow(JG)
        plt.show()
        print('Start nlgc_map_opt')

    stc_init = None
    if lead_field_gen and warm_start:
        # This branch reads `evoked`, which is only bound by the passed_evoked path,
        # so it has never been runnable. nlgc/utils/warm_start.py:warm_start_sources
        # is the supported route if this is needed.
        raise NotImplementedError(
            'warm_start is not supported with lead_field_gen=True; the block below '
            'requires an `evoked` that this path never builds. Use passed_evoked, or '
            'route through nlgc.utils.warm_start.warm_start_sources.')
    if False:
        evoked[0].drop_channels(evoked[0].info["bads"])
        evoked[0]._pick_drop_channels(mne.pick_types(evoked[0].info, meg = True))
        evoked[0].data = y.T
        evoked = evoked[0]
        # raw = mne.io.RawArray(y.T, info)
        # epochs = mne.Epochs(raw)
        # evoked = epochs.average()
        inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=loose, depth=depth, rank=rank, fixed=True)
        inv_stc = apply_inverse(evoked, inv)
        _x = np.zeros((inv_stc.data.shape[1], n_eigenmodes * 84 * order))
        stc_init = surface_ico4_to_surface_eigs(inv_stc, weights, n_eigenmodes)
        for _p in range(order):
            _x[:, _p * n_eigenmodes * 84: (_p+1) * n_eigenmodes * 84] = np.ascontiguousarray(np.roll(stc_init, -_p, axis=0))
        print(_x.shape)
        stc_init = (_x, np.ascontiguousarray(np.roll(_x, -1, axis=0)))
        # print(f'mean is {np.mean(x)}')
        # mean = 0  
        # std_dev = 0.000001
        # print(x.shape)
        # noise = np.random.normal(mean, std_dev, x.shape)
        # stc_init = x + noise
        # print(stc_init)
    if diff_lf:
        f, info, noise_cov, fwd, weights = lead_field_generation(
            root=root, subject_id=subject_id, src_space=src_space,
            n_eigenmodes=n_eigenmodes, n_orients=n_orients, loose=loose,
            depth=depth, pca=pca, rank=rank, trans=trans,
            vol_pos_origin=vol_pos_origin, vol_pos_target=vol_pos_target)
        print(f'Creating diff lf; shape of second lead field: {f.shape}')
    start_time = time.time()
    if run_ggc == False or (run_ggc and ggc_kwargs != None and ggc_kwargs['model_params'] == None):
        temp_obj = nlgc_map_opt(y.T, f, r=r_cov, order=p, self_history=p, lambda_range=lambda_range, n_segments=n_segments,
                                    var_thr=var_thr, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                                    sparsity_factor=sparsity_factor, n_eigenmodes = n_eigenmodes, n_orients = n_orients, xs_init = stc_init, a_init = a_init, use_es = use_es, patch_idx = patch_idx, verbose = verbose,
                                    parallel_mode = parallel_mode, n_devices = n_devices, n_workers = n_workers,
                                    n_warmup_iter = n_warmup_iter, use_wald_screen = use_wald_screen,
                                    wald_screen_alpha = wald_screen_alpha,
                                    use_empirical_null = use_empirical_null)
    else:
        temp_obj = ggc_kwargs['model']
    end_time = time.time()
    total_time = end_time - start_time
    
    J = temp_obj.get_J_statistics(alpha)

    if run_ggc:
        print('Running GGC')
        if ggc_kwargs != None:
            model_kwargs = ggc_kwargs['model_kwargs']
            multitaper_kwargs = ggc_kwargs['multitaper_kwargs']
            if ggc_kwargs['model_params'] != None:
                model_params = ggc_kwargs['model_params']
            else:
                model_params = temp_obj._model_f[0]._parameters
        else:
            model_kwargs = None
            multitaper_kwargs = None
        ggc_obj, ggc_mt, pexp, obs, binary_mask, modelkwargs, multitaperkwargs, J_GGC = ggc_map(model_params, model_kwargs, multitaper_kwargs, alpha=0.1, frameno=0, J=None)
        print('Completed GGC')

    if save_dir != None:
        data_gen_dict = {
            'seed': seed,
            'band': band,
            'fs': fs,
            'natures': natures,
            'm_active': m_active,
            'n_links': n_links,
        }

        ggc_dict = {
            'run_ggc': run_ggc,
            'model_kwargs': ggc_kwargs['model_kwargs'] if ggc_kwargs != None else None,
            'multitaper_kwargs': ggc_kwargs['multitaper_kwargs'] if ggc_kwargs != None else None,
        }

        if lead_field_gen:
            lead_gen_dict = {
                'root_dir': root,
                'subject_id': subject_id,
                'trans': trans,
            }
        else:
            lead_gen_dict = None

        param_dict = {
            'best_lambda': temp_obj._model_f[0].lambda_,
            'lambda_range': lambda_range,
            'order': order,
            'n_eigenmodes': n_eigenmodes,
            't': t,
            'use_es': use_es,
            'data_gen': data_gen_dict,
            'warm_start': warm_start,
            'self_history': self_history,
            'lead_field_gen': lead_gen_dict,
            'passed_evoked': passed_evoked,
            'ggc_params':ggc_dict,
            'nlgc_map_time': total_time,
        }
        if run_ggc:
            ggc_model_extras = {
                'pexp': pexp,
                'obs': obs,
                'binary_mask': binary_mask,
                'ggc_mt': ggc_mt,
            }
        if run_ggc:
            save_info(dir = save_dir,a = a, JG = JG, model = temp_obj, order = order, param_dict = param_dict, ggc_model = ggc_obj, J_GGC = J_GGC, ggc_model_extras = ggc_model_extras, debug_report = debug_report)
        else:
            save_info(dir = save_dir,a = a, JG = JG, model = temp_obj, order = order, param_dict = param_dict, debug_report = debug_report)



    plt.imshow(JG)
    plt.show()
    plt.imshow(J)
    plt.show()
    a_concat = np.concatenate(a[:], axis = 1)
    plt.imshow(a_concat, cmap = 'seismic')
    plt.show()
    a_model = temp_obj._model_f[0]._parameters[0]
    a_model = np.concatenate(a_model[:], axis = 1)
    plt.imshow(a_model, cmap = 'seismic',)
    plt.show()
    return temp_obj


