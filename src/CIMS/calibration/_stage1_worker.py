"""
Subprocess entry point for `run_stage1_node_with_timeout` and
`run_stage1_nodes_parallel` (Calibration/Optimization/optimize_ms.py).

    python _stage1_worker.py <job.pkl> <result.pkl>

`job.pkl` is a dict with:
    'nodeName'                the node to fit
    'calibration_output_dir'  where fitted fics/lifetimes are written
    'fit_kwargs'              passed to the fit function
    'mode'                    'total' (fics + lifetimes) or 'new_share' (fics)
    'log_dir'                 per-year solver log directory (optional)
    and ONE of
    'model'                   the model object itself
    'model_path'              a pickle (plain or gzip) to load it from

`result.pkl` receives the same `(status, payload)` pair the runner returns.
Anything that raises becomes `('error', traceback)`; the runner treats a
missing result file as a crash and reports the subprocess's stderr.

Output CSVs are rewritten whole per region by `write_rows_by_region`, so when
several workers run at once the write is serialised through a lock file in the
output directory. The lock is a plain `O_EXCL` create, which works on Windows
and POSIX alike without extra packages.
"""
import gzip
import os
import pickle
import re
import sys
import time
import traceback

# Make `CIMS` importable when the package is not installed into the interpreter
# (the usual case is that it is; this only adds `src` when it is not).
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(os.path.dirname(_here))
if _src not in sys.path:
    sys.path.insert(0, _src)
if _here not in sys.path:
    sys.path.insert(0, _here)


class _OutputLock:
    """Exclusive lock on `<directory>/.write_lock`, held while CSVs are rewritten."""

    def __init__(self, directory, stale_after=600, poll=0.2):
        self.path = os.path.join(directory, '.write_lock')
        self.stale_after = stale_after
        self.poll = poll
        self.fd = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                # A worker that was killed mid-write (timeout) leaves the lock
                # behind; treat an old one as abandoned.
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_after:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                time.sleep(self.poll)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def _load_model(job):
    if 'model' in job:
        return job['model']
    path = job['model_path']
    with open(path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def main(in_path, out_path):
    with open(in_path, 'rb') as f:
        job = pickle.load(f)

    try:
        from Calibration.Optimization.optimize_ms import (
            optimize_total_market_share_fic_lifetime,
            optimize_new_market_share_fic,
        )
        from Calibration.Utility.write_fics import write_fics
        from Calibration.Utility.write_lifetimes import write_lifetimes

        node = job['nodeName']
        out_dir = job['calibration_output_dir']
        mode = job.get('mode', 'total')
        fit_kwargs = dict(job.get('fit_kwargs') or {})

        log_dir = job.get('log_dir') or os.path.join(out_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fit_kwargs.setdefault(
            'logFile',
            os.path.join(log_dir, re.sub(r'[^A-Za-z0-9]+', '_', node) + '_stage1.log'))

        model = _load_model(job)
        t0 = time.time()

        if mode == 'total':
            result = optimize_total_market_share_fic_lifetime(
                model, node, plot=False, verbose=False, **fit_kwargs)
            payload = {
                'final_baseline': result['final_baseline'],
                'final': result['final'],
                'changed': result['changed'],
                'lifetimes': result['lifetimes'],
            }
            with _OutputLock(out_dir):
                write_lifetimes(model, node, out_dir, include_subtree=False)
                write_fics(model, node, out_dir, include_subtree=False)
        elif mode == 'new_share':
            result = optimize_new_market_share_fic(
                model, node, verbose=False, **fit_kwargs)
            payload = {
                'final': sum(r['end'] for r in result.values()),
                'final_baseline': sum(r['start'] for r in result.values()),
                'changed': {},
            }
            with _OutputLock(out_dir):
                write_fics(model, node, out_dir, include_subtree=False)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        payload['elapsed'] = time.time() - t0
        outcome = ('ok', payload)
    except Exception:
        outcome = ('error', traceback.format_exc())

    with open(out_path, 'wb') as f:
        pickle.dump(outcome, f, protocol=-1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
