import numpy as np
import scipy
from scipy import linalg
import os
from ._nlgc_utils import _gc_extraction , _prepare_eigenmodes, NLGC, surface_ico4_to_surface_eigs
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
def save_info(dir, a, JG, model, order, param_dict, ggc_model = None, J_GGC = None, ggc_model_extras = None,  zip_pkl = True):

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
def lead_field_generation(root, subject_id, n_eigenmodes, loose=0.0, depth=0.0, pca=True, rank=None, trans = None):
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
    bem_folder= root + "/bem/" + subject_id + "/"
    f_opt = mne.make_forward_solution(info, trans_file, src = bem_folder + subject_id + "-ico-4-src.fif",
                                        bem = bem_folder + subject_id  + "-inner_skull-bem-sol.fif")
    f_opt_ico_1 = mne.make_forward_solution(info, trans_file, src = bem_folder + subject_id  + "-ico-1-src.fif",
                                        bem = bem_folder + subject_id + "-inner_skull-bem-sol.fif")
    f_opt_data = f_opt['sol']
    weights, G, label_vertidx, label_names, gain_info, whitener = _prepare_eigenmodes(info, f_opt, noise_cov, f_opt_ico_1, n_eigenmodes=n_eigenmodes, loose=loose, depth=depth, pca=pca, rank=rank,
    mode='svd_flip')
    G_normalizing_factor = np.sqrt(np.sum(G ** 2, axis=0))
    G /= G_normalizing_factor

    return G, info, noise_cov, f_opt, weights



def data_generation(seed=0, band='wide', fs=50, natures='all', n_eigenmodes = 2, G = None, p = 2, t = 500, m_active = 10, n_links = 10, target_spec_rad = .9):
    print(f't is {t}')
    if p < 1:
        raise Exception('p should be at least 1')
    np.random.seed(seed)
    if (type(G) == type(None)):
        n = 100 # number of sensors
        
        n_patches = 20
        print(f'n_patches is {n_patches}')
        m = n_patches*n_eigenmodes # number of sources

        
        # 4*4*1  x*1000
        # 168*168*2 155*60*50
    else:
        n, m = G.shape

        print(f'G shape is {G.shape}')

        n_patches = m // n_eigenmodes

    burnin = max(200, 10*fs, 10* p * m)
    
    q = 0.01*np.eye(m)
    a = np.zeros((p, m, m), dtype=np.float64)
    idx_i = np.random.randint(0, m, size = m_active)

    print(f'n_patches is {n_patches}')
    print(f'A shape is {a.shape}')
    print(f' idx_i is {idx_i}')


    if band == 'wide':
        for ii in idx_i:
            q[ii, ii] = 1
            a[0, ii, ii] = 0.9
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
            q[ii, ii] = 1
            a[0, ii, ii] = 0.45*2*np.cos(w0)
            a[1, ii, ii] = -(.45**2)
        
    # (i,j) pairs to add a link to
    i_idx = np.random.randint(0, m_active, n_links)
    j_idx = np.random.randint(0, m_active - 1, n_links)
    j_idx += (j_idx >= i_idx)  # to prevent self-links, if j >= i, add 1 to j
    for i, j in zip(i_idx, j_idx):
        # (i,j) pair has a random link nature
        if natures == 'all':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(-0.25, 0.25)
        elif natures == 'excitatory':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.25)
        elif natures == 'inhibitory':
            for k in range(p):
                a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.25)
        elif natures == 'sharpening1':
            for k in range(p):
                if k % 2 == 0:
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.25)
                else:      
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.25) 
        elif natures == 'sharpening2':
            for k in range(p):
                if k % 2 == 0:
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.25)
                else:      
                    a[k, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.25) 
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

    u = np.random.standard_normal(m * t)
    u.shape = (t, m)
    
    l = linalg.cholesky(q, lower=True)
    u = u.dot(l.T)

    print(f'Max of u {np.max(u)}')

    u /= (np.median(u)/.001)
    # u /= np.sqrt(np.sum(u ** 2, axis=0))
    print(f'Max of u scaled{np.max(u)}')


    print(f'Max of a scaled {np.max(a)}')
    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    for i in range(p, t):
        x[i] = u[i]
        for k in range(p):
            x[i] += a[k].dot(x[i - k - 1])


    print('band', band, x.shape)
    for i in range(x.T.shape[0]):
        plt.psd(x.T[i], Fs=fs)
    plt.show()
    
    pow_actives = []
    for i in range(m):
        pow_actives.append(np.mean(x[:, i]**2))

    if (type(G) == type(None)):
        f = np.random.randn(n, m) + np.eye(n, m)
        f /= np.sqrt(np.sum(f ** 2, axis=0))
        
        
    else:
        f = G
    print(f'Max of f {np.max(f)}')
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
                 max_cyclic_iter=3, tol=1e-5, sparsity_factor=0.0, cv=5, n_eigenmodes = 2, xs_init = None, use_es = False, patch_idx = None, verbose = False):
    n, nnx = G.shape
    len_patch_idx = nnx // n_eigenmodes
    _, t = M.shape
    tt = t // n_segments
    print(f'r is {r}')
    d_raw = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    bias_r = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    bias_f = np.zeros((n_segments, 1))
    conv_flag = np.zeros((n_segments, len_patch_idx, len_patch_idx))
    models = []
    ROI_list = list(range(len_patch_idx))
    if patch_idx != None:
        ROI_list = patch_idx
    print('Starting loop')
    for n in range(0, n_segments):
        print('Segment: ', n + 1)
        print(lambda_range)
        print(nnx)
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            _gc_extraction(M[:, n * tt: (n + 1) * tt], G, r, p=order, p1=self_history, n_eigenmodes=n_eigenmodes,
                           ROIs=ROI_list, cv=cv, lambda_range=lambda_range, max_iter=max_iter,
                           max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor,
                           use_lapack=True, use_es=use_es, var_thr=var_thr, xs_init = xs_init, verbose = verbose)
        d_raw[n] = d_raw_
        bias_r[n] = bias_r_
        bias_f[n] = bias_f_
        models.append(model_f)
        conv_flag[n] = conv_flag_
        nlgc_obj = NLGC('Simulation_rnd', len_patch_idx, n, t, order, n_eigenmodes, n_segments, d_raw, bias_f, bias_r,
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
def run_GT_sim(lead_field_gen = False, lf = None, seed = 0, band = "wide", fs = 50, natures = 'all', 
        root = None, subject_id = None, session_name = None, trans = None, order = 2, t = 500, n_eigenmodes = 1,
        n_segments = 1, loose = 0.0, depth = 0.0, pca = True, rank = None, lambda_range = None,
        max_iter = 500, max_cyclic_iter = 3, tol = 1e-5, sparsity_factor = 0.0, cv = 5 ,var_thr = 1.0, alpha = .1, 
        m_active = 10, n_links = 10, warm_start = False, self_history = None, passed_evoked = None, use_es = False, 
        verbose = False, diff_lf = False, patch_idx = None, save_dir = None, run_ggc = False, ggc_kwargs = None):
    
    if (passed_evoked != None):
        print('using passed in evoked')
        noise_cov = mne.read_cov(passed_evoked['noise_cov'])
        fwd = mne.read_forward_solution(passed_evoked['fwd'])
        evoked = mne.read_evokeds(passed_evoked['evoked'])
        src_target = mne.read_source_spaces(passed_evoked['src_target'])
        info = evoked[0].info
        weights, G, label_vertidx, label_names, gain_info, whitener = _prepare_eigenmodes(info, fwd, noise_cov, src_target, 
                                                                            n_eigenmodes=n_eigenmodes, loose=loose, depth=depth, pca=pca, rank=rank, mode='svd_flip')
    elif (lead_field_gen):
        G, info, noise_cov, fwd, weights = lead_field_generation(root, subject_id, n_eigenmodes, loose, depth, pca, rank, trans)
    elif (type(lf) != type(None)):
        print('Using passed in lead field')
        G = lf
    else:
        G = None
    f, y, x, r_cov, p, JG, pow_actives, a = data_generation(seed, band, fs, natures, n_eigenmodes, G, order, t, m_active, n_links)
    print("Completed data gen")
    plt.imshow(JG)
    plt.show()
    print('Start nglc_map_opt')

    stc_init = None
    if lead_field_gen & warm_start:
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
    print(stc_init == None)

    if diff_lf:
        f, info, noise_cov, fwd, weights = lead_field_generation(root, subject_id, n_eigenmodes, loose, depth, pca, rank, trans)
        print('Creating diff lf')
        print(f'Shape of second lead field: {f.shape}')
    print(patch_idx)
    if run_ggc and ggc_kwargs != None and ggc_kwargs['model_params'] == None:
        temp_obj = nlgc_map_opt(y.T, f, r=r_cov, order=p, self_history=p, lambda_range=lambda_range, n_segments=n_segments,
                                    var_thr=var_thr, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                                    sparsity_factor=sparsity_factor, n_eigenmodes = n_eigenmodes, xs_init = stc_init, use_es = use_es, patch_idx = patch_idx, verbose = verbose)
    else:
        temp_obj = ggc_kwargs['model']
        
    
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
        }

        ggc_model_extras = {
            'pexp': pexp,
            'obs': obs,
            'binary_mask': binary_mask,
            'ggc_mt': ggc_mt,
        }
        
        save_info(dir = save_dir,a = a, JG = JG, model = temp_obj, order = order, param_dict = param_dict, ggc_model = ggc_obj, J_GGC = J_GGC, ggc_model_extras = ggc_model_extras)


    plt.imshow(JG)
    plt.imshow(J)
    plt.show()
    return temp_obj


