"""Logging helpers that keep training runs readable when output is redirected.

Progress bars are silenced automatically outside an interactive terminal, and
events are written as single timestamped lines with an upper-case tag, so a
long run can be followed, timed and resumed from its log file alone.

    from logging_utils import tqdm, alog, fmt_elapsed, suppress_hf_noise
    alog('EPOCH_END', fold=1, ep=3, sec='12.3', val_mcc='0.416')
    # -> [2026-07-12 12:03:41][EPOCH_END] fold=1 ep=3 sec=12.3 val_mcc=0.416

Tags used by the training scripts: RUN_START / RUN_END, FOLD_START / FOLD_DONE,
EPOCH / EPOCH_END, STEP, FOLD_RESULT / RUN_SUMMARY, FOLD_MISSING, PARAM_GROUPS,
COMPILE, RESUME, VOTE_HINT."""
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm as _TqdmBase


class tqdm(_TqdmBase):
    """A tqdm that disables itself when stderr is not a terminal.

    Redirected and background runs therefore produce no carriage-return spam,
    while an interactive session behaves exactly as before. An explicit
    `disable=` argument still wins.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("disable", not sys.stderr.isatty())
        super().__init__(*args, **kwargs)


def ts():
    """Current wall-clock timestamp, with the date, so multi-day runs stay unambiguous."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def rank():
    """Process rank in a distributed run (0 when not distributed); only rank 0 logs."""
    for k in ('RANK', 'SLURM_PROCID', 'LOCAL_RANK'):
        v = os.environ.get(k, '').strip()
        if v:
            try:
                return int(v)
            except ValueError:
                return 0
    return 0


def alog(tag, msg='', **kv):
    """Print one event line: [YYYY-MM-DD HH:MM:SS][TAG] msg k=v ... (flushed immediately)."""
    if rank() != 0:
        return
    parts = [f'[{ts()}][{tag}]']
    if msg:
        parts.append(str(msg))
    parts += [f'{k}={v}' for k, v in kv.items()]
    print(' '.join(parts), flush=True)


def fmt_elapsed(seconds):
    """Format seconds as [Nd]HH:MM:SS; hours accumulate into days rather than wrapping."""
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f'{d}d' if d else '') + f'{h:02d}:{m:02d}:{s:02d}'


class Heartbeat:
    """Throttled progress line for long epochs, for use when the progress bar is off.

    At most one line is printed every `interval` seconds, so a short epoch stays
    silent. The interval comes from the argument, else DUET_HEARTBEAT_SEC,
    else 60 seconds when redirected and 0 (off) in an interactive terminal.

        hb = Heartbeat('STEP', total=len(loader), ep='3/30')
        for i, batch in enumerate(loader):
            ...
            hb.beat(i + 1)
    """

    def __init__(self, tag='STEP', total=None, interval=None, **ctx):
        if interval is None:
            env = os.environ.get('DUET_HEARTBEAT_SEC')
            interval = float(env) if env is not None else (0.0 if sys.stderr.isatty() else 60.0)
        self.tag, self.total, self.interval, self.ctx = tag, total, interval, ctx
        self.t0 = self.last = time.time()

    def beat(self, done):
        """Call at the end of each batch with the number done; returns silently until due."""
        if self.interval <= 0:
            return
        now = time.time()
        if now - self.last < self.interval:
            return
        self.last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        kv = dict(self.ctx)
        kv['step'] = f'{done}/{self.total}' if self.total else done
        kv['elapsed'] = fmt_elapsed(elapsed)
        kv['it_s'] = f'{rate:.1f}'
        if self.total and rate > 0:
            kv['eta_ep'] = fmt_elapsed((self.total - done) / rate)
        alog(self.tag, **kv)


class Progress:
    """Coarse progress and ETA over a series of units (for example folds), printed as event lines."""

    def __init__(self, total, key='series', done=0, interval=60.0, tag='PROGRESS'):
        self.total, self.key, self.n, self.interval, self.tag = int(total), key, int(done), interval, tag
        self.durs = []
        self.t_unit = None
        self.ctx = {}
        self.last = 0.0
        self.stepped = False

    def start(self, step=None, **ctx):
        self.t_unit, self.ctx = time.time(), ctx
        if step is not None:
            self.ctx, self.stepped = {'step': step, **ctx}, True
            alog('STEP_START', **self._ctx({}))

    def skip(self, step=None, **ctx):
        self.t_unit = None
        self.n = min(self.n + 1, self.total)
        if step is None:
            return
        self.ctx, self.stepped = {'step': step, **ctx}, True
        alog('SKIP', **self._ctx({}))
        self._emit()

    def _ctx(self, kv):
        return {(f'{k}_' if k in kv or k in ('tag', 'msg') else k): v for k, v in self.ctx.items()}

    def _emit(self):
        now = time.time()
        self.last = now
        cur = now - self.t_unit if self.t_unit is not None else 0.0
        kv = {}
        mean = sum(self.durs) / len(self.durs) if self.durs else 0.0
        if mean > 0:
            a = self.n + (min(cur / mean, 0.95) if self.t_unit is not None and self.n < self.total else 0.0)
            left = max(0, self.total - self.n - (1 if self.t_unit is not None else 0))
            eta = left * mean + (max(mean - cur, 0.0) if self.t_unit is not None else 0.0)
            kv[self.key] = f'{a:.2f}/{self.total}' if a != int(a) else f'{int(a)}/{self.total}'
            if not self.stepped:
                kv['eta_s'] = int(round(eta))
                kv['unit_s'] = int(round(mean))
        else:
            kv[self.key] = f'{self.n}/{self.total}'
        alog(self.tag, **kv, **self._ctx(kv))

    def tick(self):
        if self.interval > 0 and time.time() - self.last >= self.interval:
            self._emit()

    def done(self, **ctx):
        d = time.time() - self.t_unit if self.t_unit is not None else 0.0
        if d > 0:
            self.durs.append(d)
        self.n = min(self.n + 1, self.total)
        self.t_unit = None
        self.ctx.update(ctx)
        self._emit()


def suppress_hf_noise():
    """Silence the repeated transformers weight-initialisation warnings."""
    import warnings
    warnings.filterwarnings('ignore')
    try:
        from transformers import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass
