import pickle
import json
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from ..nlgc_test_utils import zplane


def find_poles_and_zeros(model, order):
    A = model._model_f[0]._parameters[0]
    mask = np.abs(A).mean(axis=0)
    mask = mask > 0.1

    locx, locy = np.nonzero(mask)
    a_s = []
    b_s = []

    for i in range(order):
        a_s.append(A[i][locx, locy])
        b_s.append(A[i][locx, locx])

    a_s = np.array(a_s)
    b_s = np.array(b_s)

    zs, ps = [], []
    zs_t, ps_t = [], []

    for i in range(len(a_s[0])):
        bsig = [1] + (-b_s[:,i]).tolist()
        asig = a_s[:,i].tolist()
        zsig, psig, ksig = scipy.signal.tf2zpk(asig, bsig)
        zs.extend(zsig)
        ps.extend(psig)

    return zs, ps


# Plot coeff matrix with lags side by side
def generate_coefficient_matrix_plot(model_params):
    conv = int(np.floor((5/350)*model_params.shape[1]) + 1)
    arr_model = np.concatenate(model_params[:], axis = 1)
    fig, ax = plt.subplots(figsize=(75, 75))
    ax.imshow(scipy.signal.convolve2d(arr_model, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
    ax.title('Derived Model Parameters Concatenated', fontsize = 80)

    return fig

# Plot coeff matrix with diagonal removed and lags side by side
# TODO: Update to work with 3x3 blocked lags
def generate_no_diagonal_summed_coeff_matrix_plot(model_params):
    conv = int(np.floor((5/350)*model_params.shape[1]) + 1)

    negated_identity = np.abs(np.eye(model_params.shape[1]) - 1)
    model_params_abs = np.abs(model_params[:]*negated_identity)
    model_params_summed = np.sum(model_params_abs[:], axis = 0)

    fig, ax = plt.subplots(figsize=(75, 75))
    ax.imshow(scipy.signal.convolve2d(model_params_summed, np.ones((conv,conv))), cmap = 'seismic', vmin=-1, vmax=1)
    ax.title('No Diagonal Absolute Summed Lags Derived Model Params', fontsize = 80)

    return fig



# Generate json of parameters used for running NLGC map
def generate_report(save_dir, model, param_dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")
    else:
        print(f"Directory '{save_dir}' already exists.")

    with open(save_dir + "nlgc_map_params.json", "w") as f:
        json.dump(param_dict, f, indent=4)

    model_path = save_dir + 'model.pkl'

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


    model_params = model._model_f[0]._parameters[0]

    zs, ps = find_poles_and_zeros(model, param_dict.order)

    figs = [generate_coefficient_matrix_plot(model_params = model_params),
            generate_no_diagonal_summed_coeff_matrix_plot(model_params = model_params),
            zplane(np.array(zs),np.array(ps), 'Model Parameters Pole Zero Plot'),
            ]

    with PdfPages(save_dir + 'model-report.pdf') as pdf:

        for fig in figs:
            pdf.savefig(fig)
            plt.close()

        d = pdf.infodict()
        d['Title'] = 'Model Analyatics PDF'
        d['Author'] = 'Kavin Loganathan'

    



    

