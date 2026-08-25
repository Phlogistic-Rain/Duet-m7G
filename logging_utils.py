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


def alog(tag, msg='', **kv):
    """Print one event line: [YYYY-MM-DD HH:MM:SS][TAG] msg k=v ... (flushed immediately)."""
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


def suppress_hf_noise():
    """Silence the repeated transformers weight-initialisation warnings."""
    import warnings
    warnings.filterwarnings('ignore')
    try:
        from transformers import logging as _hf_logging
        _hf_logging.set_verbosity_error()
    except Exception:
        pass
