"""Tee stdout and stderr into a run.log alongside a run's other output."""
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime


def _pump(read_fd, log, original_fd):
    """Copy the pipe to both the log and the real terminal."""
    with os.fdopen(read_fd, 'rb', 0) as reader:
        for chunk in iter(lambda: reader.read(4096), b''):
            try:
                os.write(original_fd, chunk)
                log.write(chunk)
                log.flush()
            except (OSError, ValueError):
                return  # block exited; a lingering pool can still write


def _shutdown_worker_pool():
    """Stop loky's workers so their block-buffered stdout flushes in time.

    Without this the buffers flush after the log has closed.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
    except ImportError:
        return
    try:
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


@contextmanager
def tee_output(directory, filename='run.log'):
    """Tee everything written to fd 1/2 inside the block into run.log.

    fd level, not sys.stdout, so workers and native jax/BLAS output are caught
    too. Yields the log path, or None if `directory` is None.
    """
    if directory is None:
        yield None
        return

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    read_fd, write_fd = os.pipe()

    log = open(path, 'ab')
    log.write(f'\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n'.encode())
    reader = threading.Thread(target=_pump, args=(read_fd, log, saved_out),
                              daemon=True)
    reader.start()

    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    # read at child startup, so only reaches workers spawned from here on
    previous_unbuffered = os.environ.get('PYTHONUNBUFFERED')
    os.environ['PYTHONUNBUFFERED'] = '1'
    failure = None
    try:
        yield path
    except BaseException:
        # python prints it only after fd 2 is restored, so capture it here
        failure = traceback.format_exc()
        raise
    finally:
        # order matters: drain the workers before taking the pipe away
        _shutdown_worker_pool()
        sys.stdout.flush()
        sys.stderr.flush()
        # restoring fd 1/2 drops the pipe's last write end, so _pump sees EOF.
        # Join before closing saved_out -- the thread still writes to it.
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        reader.join(timeout=5)
        os.close(saved_out)
        os.close(saved_err)
        if failure is not None:
            log.write(failure.encode())
        log.close()
        if previous_unbuffered is None:
            os.environ.pop('PYTHONUNBUFFERED', None)
        else:
            os.environ['PYTHONUNBUFFERED'] = previous_unbuffered
