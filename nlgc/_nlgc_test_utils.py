import numpy as np
import scipy
from scipy import linalg
import os
from ._nlgc_utils import _gc_extraction , _prepare_eigenmodes, NLGC

from mne.source_space import SourceSpaces
from mne.utils import (logger, _check_option, _validate_type)
from mne.inverse_sparse.mxne_inverse import _prepare_gain
from mne import (Forward, Label)
from mne.forward import is_fixed_orient
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import mne


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
def lead_field_generation(root, subject_id, n_eigenmodes, trans = None):
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
    weights, G, label_vertidx, label_names, gain_info, whitener = _prepare_eigenmodes(info, f_opt, noise_cov, f_opt_ico_1, n_eigenmodes=n_eigenmodes, loose=0.0, depth=0.0, pca=True, rank=None,
    mode='svd_flip')
    G_normalizing_factor = np.sqrt(np.sum(G ** 2, axis=0))
    G /= G_normalizing_factor

    return G

def data_generation(seed=0, band='wide', fs=50, natures='all', n_eigenmodes = 2, G = None, p = 2, t = 500, m_active = 10, n_links = 10):
    print(f't is {t}')
    np.random.seed(seed)
    if (type(G) == type(None)):
        n = 100 # number of sensors
        
        n_patches = 10
        m = n_patches*n_eigenmodes # number of sources

        
        # 4*4*1  x*1000
        # 168*168*2 155*60*50
    else:
        n, m = G.shape
        print(G.shape)
        n_patches = m // n_eigenmodes


    
    q = 0.01*np.eye(m)

    a = np.zeros(p * m * m, dtype=np.float64)
    a.shape = (p, m, m)
    print(n_patches)
    print(a.shape)
    idx_i = np.random.randint(0, m, size = m_active)
    print(idx_i)
    for i in range(m_active):
        q[idx_i[i], idx_i[i]] = 1
        a[0, idx_i[i], idx_i[i]] = 0.9
    
    # if n_links > m_active:
    #     print('n_links is greater than m_active setting it to size m_active')
    #     n_links = m_active
        
    # (i,j) pairs to add a link to
    for i, j in zip(np.random.randint(0, m_active, n_links), 
                    np.random.randint(0, m_active, n_links)):
        # (i,j) pair has a random link nature
        if natures == 'all':
            a[0, idx_i[i], idx_i[j]] = np.random.uniform(-0.5, 0.5)
            a[1, idx_i[i], idx_i[j]] = np.random.uniform(-0.5, 0.5)
        elif natures == 'excitatory':
            a[0, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.5)
            a[1, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.5)
        elif natures == 'inhibitory':
            a[0, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.5)
            a[1, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.5)
        elif natures == 'sharpening1':
            a[0, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.5)
            a[1, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.5)
        elif natures == 'sharpening2':
            a[0, idx_i[i], idx_i[j]] = np.random.uniform(0, -0.5)
            a[1, idx_i[i], idx_i[j]] = np.random.uniform(0, 0.5)
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
    
    x = np.empty((t, m), dtype=np.float64)
    for i in range(p):
        x[i] = 0.0

    if band == 'wide':
        for i in range(p, t):
            x[i] = u[i]
            for k in range(p):
                x[i] += a[k].dot(x[i - k - 1])
    else:
        # added artificial data filtering
        if (band == 'delta'):
            sos = scipy.signal.butter(25, [0.1, 4], 'bp', fs=fs, output='sos')
            filt = scipy.signal.sosfilt(sos, u.T).T
        elif (band == 'theta'):
            sos = scipy.signal.butter(25, [4, 8], 'bp', fs=fs, output='sos')
            filt = scipy.signal.sosfilt(sos, u.T).T
        elif (band == 'alpha'):
            sos = scipy.signal.butter(25, [8, 12], 'bp', fs=fs, output='sos')
            filt = scipy.signal.sosfilt(sos, u.T).T
        elif (band == 'beta'):
            sos = scipy.signal.butter(25, [13, 23], 'bp', fs=fs, output='sos')
            filt = scipy.signal.sosfilt(sos, u.T).T
        else:
            raise Exception(f'band {band} not implemented')
        
        for i in range(p, t):
            x[i] = filt[i]

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
        f = np.random.randn(n, m)
        f /= np.sqrt(np.sum(f ** 2, axis=0))
        
    else:
        f = G

    y = x.dot(f.T)
    px = y.dot(y.T).trace()

    noise = np.random.standard_normal(y.shape)
    
    pn = noise.dot(noise.T).trace()
    multiplier = 1e2 * pn / px

    y += noise / np.sqrt(multiplier)
    r_cov = 1 / multiplier

    return f, y, r_cov, p, JG, pow_actives, a


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
'''
def nlgc_map_opt(M, G, r, order, self_history=None, var_thr=1.0, n_segments=1, lambda_range=None, max_iter=500,
                 max_cyclic_iter=3, tol=1e-5, sparsity_factor=0.0, cv=5, n_eigenmodes = 2, use_es = False):
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
    print('Starting loop')
    for n in range(0, n_segments):
        print('Segment: ', n + 1)
        print(lambda_range)
        print(nnx)
        d_raw_, bias_r_, bias_f_, model_f, conv_flag_ = \
            _gc_extraction(M[:, n * tt: (n + 1) * tt], G, r, p=order, p1=self_history, n_eigenmodes=n_eigenmodes,
                           ROIs=list(range(len_patch_idx)), cv=cv, lambda_range=lambda_range, max_iter=max_iter,
                           max_cyclic_iter=max_cyclic_iter, tol=tol, sparsity_factor=sparsity_factor,
                           use_lapack=True, use_es=use_es, var_thr=var_thr)
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
'''
def run_GT_sim(lead_field_gen = False, lf = None, seed = 0, band = "wide", fs = 50, natures = 'all', 
        root = None, subject_id = None, session_name = None, trans = None, order = 2, t = 500, n_eigenmodes = 1,
        n_segments = 1, loose = 0.0, depth = 0.0, pca = True, rank = None, lambda_range = None,
        max_iter = 500, max_cyclic_iter = 3, tol = 1e-5, sparsity_factor = 0.0, cv = 5 ,var_thr = 1.0, alpha = .1, m_active = 10, n_links = 10, use_es = False):
    
    if (lead_field_gen):
        G = lead_field_generation(root, subject_id, n_eigenmodes, trans)
    elif (type(lf) != type(None)):
        G = lf
    else:
        G = None
    f, y, r_cov, p, JG, pow_actives, a = data_generation(seed, band, fs, natures, n_eigenmodes, G, order, t, m_active, n_links)
    print("Completed data gen")
    plt.imshow(JG)
    plt.show()
    print('Start nglc_map_opt')
    temp_obj = nlgc_map_opt(y.T, f, r=r_cov, order=p, self_history=p, lambda_range=lambda_range, n_segments=n_segments,
                                var_thr=var_thr, max_iter=max_iter, max_cyclic_iter=max_cyclic_iter, tol=tol,
                                sparsity_factor=sparsity_factor, n_eigenmodes = n_eigenmodes, use_es = use_es)
    
    J = temp_obj.get_J_statistics(alpha)

    plt.imshow(JG)
    plt.imshow(J)
    plt.show()
    return temp_obj


