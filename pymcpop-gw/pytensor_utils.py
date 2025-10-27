import time
import sys
import psutil
import os



def make_tqdm_callback_full(pbar):
    t0 = time.perf_counter()
    last_refresh = [t0]
    div_count    = [0]
    last_nsteps  = [None]   # last observed n_steps (after warmup)
    last_ss      = [None]   # last observed step_size (after warmup)

    def _get_stat(name, args, kwargs, chain):
        # 1) kwarg directly
        if name in kwargs:
            try:
                return kwargs[name]
            except Exception:
                pass
        # 2) Draw-like object in kwargs
        d = kwargs.get("draw", None)
        if d is not None and hasattr(d, name):
            try:
                return getattr(d, name)
            except Exception:
                pass
        # 3) Trace.get_sampler_stats (first positional arg often is the trace)
        if len(args) >= 2:
            trace = args[0]
            try:
                arr = trace.get_sampler_stats(name, chains=[chain] if chain is not None else None)
                if len(arr):
                    return arr[-1]
            except Exception:
                pass
        return None

    def _get_diverging(args, kwargs, tuning, chain):
        if "diverging" in kwargs:
            return bool(kwargs["diverging"])
        d = kwargs.get("draw", None)
        if d is not None and hasattr(d, "diverging"):
            return bool(getattr(d, "diverging"))
        if len(args) >= 2:
            trace = args[0]
            try:
                arr = trace.get_sampler_stats("diverging", chains=[chain] if chain is not None else None)
                if len(arr):
                    return bool(arr[-1])
            except Exception:
                pass
        return False

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain) or kwargs with a Draw-like object
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)
        else:
            draw  = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)

        # Normalize tuning flag if present on Draw-like object
        if hasattr(draw, "tuning") and tuning is None:
            tuning = bool(getattr(draw, "tuning", False))

        # === stats & divergence (only after warmup) ===
        if tuning is False:
            if _get_diverging(args, kwargs, tuning, chain):
                div_count[0] += 1

            nsteps = _get_stat("n_steps", args, kwargs, chain)
            if nsteps is not None:
                try:
                    last_nsteps[0] = int(nsteps)
                except Exception:
                    pass

            ss = _get_stat("step_size", args, kwargs, chain)
            if ss is not None:
                try:
                    last_ss[0] = float(ss)
                except Exception:
                    pass

        # === progress updates (same cadence as your original, lightly throttled) ===
        pbar.update(1)

        now = time.perf_counter()
        if (pbar.n % 25) == 0 and (now - last_refresh[0]) >= 0.25:
            phase = "warmup" if tuning else "sampling"
            rate  = pbar.n / max(now - t0, 1e-9)
            # Build a tiny postfix string without heavy formatting
            extras = [f"div={div_count[0]}"]
            if last_nsteps[0] is not None:
                extras.append(f"nsteps={last_nsteps[0]}")
            if last_ss[0] is not None:
                # format step size compactly
                extras.append(f"ss={last_ss[0]:.3g}")
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s | " + " ".join(extras), refresh=False)
            last_refresh[0] = now

    return cb

def make_tqdm_callback(pbar):
    t0 = time.perf_counter()
    last_refresh = [t0]  # mutable box to avoid nonlocal

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain) or Draw-like object in kwargs
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # If PyMC passed a Draw-like object, grab tuning flag from it (we don't need draw as int)
        if hasattr(draw, "tuning") and tuning is None:
            tuning = bool(getattr(draw, "tuning", False))

        # Update bar every callback (same as your original)
        pbar.update(1)

        # Postfix every 25 iters, but throttle by time to reduce overhead
        now = time.perf_counter()
        if (pbar.n % 25) == 0 and (now - last_refresh[0]) >= 0.25:
            phase = "warmup" if tuning else "sampling"
            rate = pbar.n / max(now - t0, 1e-9)
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)
            last_refresh[0] = now

    return cb


def make_tqdm_callback_frequent(pbar):
    t0 = time.perf_counter()

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain)
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # update bar (throttle if you like)
        pbar.update(1)
        if (pbar.n % 25) == 0:
            phase = "warmup" if tuning else "sampling"
            rate = pbar.n / max(time.perf_counter() - t0, 1e-9)
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)

    return cb





# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
