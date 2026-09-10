"""HDF5 persistence for fitted NLGC models.

    save_model(nlgc_obj, 'sub01_model.h5')
    nlgc_obj = load_model('sub01_model.h5')

Arrays become datasets named for their position in NLGC.to_dict's output;
everything else goes into a JSON 'skeleton' root attribute. The mne.Forward
goes to a sidecar <stem>-fwd.fif.
"""
import json
import os
import warnings
from datetime import datetime, timezone

import h5py
import numpy as np

from nlgc.nlgc_utils import NLGC, FORMAT_VERSION

__all__ = ['save_model', 'load_model', 'FORMAT_VERSION']

# only smoothed_state is routinely this large, but keep the rule generic
_COMPRESS_MIN_SIZE = 10_000


def _nlgc_version():
    try:
        from importlib.metadata import version
        return version('nlgc')
    except Exception:
        return 'unknown'


def _forward_path(path):
    stem, _ = os.path.splitext(path)
    return stem + '-fwd.fif'


def _flatten(obj, f, path=''):
    """Write arrays into `f`; return the structure with their paths in place."""
    if isinstance(obj, np.generic):  # numpy scalars are not JSON-serializable
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _flatten(v, f, f'{path}/{k}'.lstrip('/'))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flatten(v, f, f'{path}/{i}'.lstrip('/'))
                for i, v in enumerate(obj)]
    # __array__ not ndarray: solve_params returns jax arrays
    if hasattr(obj, '__array__'):
        arr = np.asarray(obj)
        f.create_dataset(path, data=arr,
                         compression='gzip' if arr.size >= _COMPRESS_MIN_SIZE
                         else None)
        return {'__arr__': path}
    return obj


def _unflatten(skeleton, f):
    """Inverse of _flatten."""
    if isinstance(skeleton, dict):
        if set(skeleton) == {'__arr__'}:
            path = skeleton['__arr__']
            if path not in f:
                raise ValueError(f"{f.filename!r} is missing dataset {path!r}; "
                                 f"the file is likely truncated")
            return f[path][()]
        return {key: _unflatten(value, f) for key, value in skeleton.items()}
    if isinstance(skeleton, list):
        return [_unflatten(value, f) for value in skeleton]
    return skeleton


def save_model(model, path, save_forward=True):
    """Write a fitted NLGC model to `path`, returning the path written.

    A .h5 suffix is appended if absent; save_forward=False skips the sidecar.
    """
    path = str(path)
    if not path.endswith(('.h5', '.hdf5')):
        path += '.h5'

    d = model.to_dict()
    with h5py.File(path, 'w') as f:
        # also a root attr, so h5dump can read it without parsing
        f.attrs['format_version'] = d['format_version']
        f.attrs['nlgc_version'] = _nlgc_version()
        f.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
        f.attrs['skeleton'] = json.dumps(_flatten(d, f))

        forward = getattr(model, 'forward_orig', None)
        if save_forward and forward is not None:
            import mne
            forward_path = _forward_path(path)
            mne.write_forward_solution(forward_path, forward, overwrite=True)
            f.attrs['forward_file'] = os.path.basename(forward_path)

    return path


def load_model(path):
    """Read an NLGC model written by save_model.

    Raises ValueError on an unrecognized format_version. A missing forward
    sidecar warns and leaves forward_orig as None rather than failing.
    """
    path = str(path)
    with h5py.File(path, 'r') as f:
        skeleton = f.attrs.get('skeleton')
        d = _unflatten(json.loads(skeleton), f) if skeleton else {}
        # the root attr is the one h5dump shows, so let it be authoritative
        if 'format_version' in f.attrs:
            d['format_version'] = int(f.attrs['format_version'])
        forward_file = f.attrs.get('forward_file')

    model = NLGC.from_dict(d)

    if forward_file is not None:
        forward_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                    forward_file)
        if os.path.exists(forward_path):
            import mne
            model.forward_orig = mne.read_forward_solution(forward_path,
                                                           verbose='error')
        else:
            warnings.warn(
                f"Forward solution sidecar {forward_file!r} not found next to "
                f"{path!r}; loading with forward_orig=None.")

    return model
