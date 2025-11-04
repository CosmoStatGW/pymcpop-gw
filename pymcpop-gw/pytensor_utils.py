import time
import sys
import psutil
import os
import pytensor.tensor as at

from pytensor.graph.basic import graph_inputs
from pytensor.tensor.random.op import RandomVariable
from pytensor.graph.basic import graph_inputs, Variable

from pytensor.graph.basic import io_toposort
from pytensor.printing import debugprint
from pytensor import shared
import numpy as onp



def _bin_indices(x, edges):
    """
    x:      (B, K) values to bin
    edges:  (NBINS+1,) monotonically increasing bin edges
    returns idx: (B, K) int64 in [0, NBINS-1]
    """
    idx = at.searchsorted(edges, x, side="right") - 1
    idx = at.clip(idx, 0, edges.shape[0] - 2)
    return idx.astype("int64")


def _scatter_sum_batched(values, idx, nbins):
    """
    values: (B, K) floatX  -> per-sample contribution to add to the bin
    idx:    (B, K) int64   -> bin index per sample
    nbins:  python int or 0-d tensor, number of bins
    return: (B, nbins) floatX, sum of 'values' per bin for every batch row
    """
    B, K = values.shape
    base = at.arange(B, dtype="int64") * nbins        # (B,)
    pos = (base[:, None] + idx).flatten()             # (B*K,)
    val = values.flatten()                            # (B*K,)
    out = at.zeros((B * nbins,), values.dtype)        # (B*nbins,)
    out = at.inc_subtensor(out[pos], val, inplace=False)
    return out.reshape((B, nbins))

def pt_vec(x, DT="float64"):
    x = onp.asarray(x, dtype=DT).reshape(-1)
    return shared(x, borrow=True)  # or at.as_tensor_variable(x) if truly tiny


def dump_uniform_sources(outputs, context=2):
    outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    nodes = io_toposort([], outs)
    rv_nodes = [n for n in nodes if isinstance(getattr(n, "op", None), RandomVariable)]
    print(f"Found {len(rv_nodes)} RandomVariable nodes")
    for k, n in enumerate(rv_nodes, 1):
        print(f"\n[{k}] RV op: {n.op}  | owner: {n}")
        # who uses this random draw?
        for out in n.outputs:
            for client in out.clients:
                print("  used by:", client)
        # small subgraph around it (VERY helpful)
        try:
            print("\n--- debugprint around this node ---")
            print(debugprint(n.outputs, print_type=True, stop_on_name=True, depth=context))
        except Exception as e:
            print("debugprint failed:", e)
            
# ---------- flatten containers ----------
def _flatten(name, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(f"{name}.{k}", v)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(f"{name}[{i}]", v)
    else:
        yield name, obj

# ---------- classification ----------
def classify_tensor(x, model=None):
    """
    Classify x w.r.t. RandomVariables.
    Returns dict with:
      has_rv: bool
      rv_nodes: list[str] of RV op types
      is_value_var: bool
      source_rv: RV (if x is a value var)
    """
    info = {
        "has_rv": False,
        "rv_nodes": [],
        "is_value_var": False,
        "source_rv": None,
    }

    # Non-PyTensor objects are deterministic
    if not isinstance(x, Variable):
        return info

    # Is x itself a value-var?
    if model is not None and x in model.values_to_rvs:
        info["is_value_var"] = True
        info["source_rv"] = model.values_to_rvs[x]

    # Find any RVs in x's graph
    try:
        ins_gen = graph_inputs([x])
        ins = list(ins_gen)  # materialize here so exceptions are caught
        rv_ops = []
        for v in ins:
            if getattr(v, "owner", None) and isinstance(getattr(v.owner, "op", None), RandomVariable):
                rv_ops.append(type(v.owner.op).__name__)
        if rv_ops:
            info["has_rv"] = True
            info["rv_nodes"] = sorted(set(rv_ops))
    except Exception:
        # If graph introspection fails, assume deterministic for safety in reporting
        return info

    return info

def print_input_rv_report(model=None, **kwargs):
    """
    Example:
        print_input_rv_report(
            model=model,
            m1inj=m1inj, m2inj=m2inj, dLinj=dLinj, spinsInj=spinsInj,
            log_p_draw=lpdinj, Lambda=Lambda_val, dL_grid=dL_grid_val, z_grid=z_grid_val
        )
    """
    lines = []
    for name, obj in kwargs.items():
        for leaf_name, leaf in _flatten(name, obj):
            info = classify_tensor(leaf, model=model)

            status_bits = []
            if info["is_value_var"]:
                status_bits.append("VALUE_VAR")
            if info["has_rv"]:
                status_bits.append("CONTAINS_RV")
            status = "deterministic" if not status_bits else ",".join(status_bits)

            src = ""
            if info["is_value_var"] and info["source_rv"] is not None:
                src_name = getattr(info["source_rv"], "name", "<unnamed RV>")
                src = f" (from {src_name})"

            rv_types = f" rv_ops={info['rv_nodes']}" if info["rv_nodes"] else ""
            dtype = getattr(leaf, "dtype", type(leaf).__name__)
            shape_str = str(getattr(leaf, "shape", ""))

            lines.append(f"- {leaf_name}: {status}{src}{rv_types} | dtype={dtype} | shape={shape_str}")

    print("\n".join(lines))




def as_value_var(x, model):
    # If x is an RV, return its value-var; otherwise return x unchanged
    return model.rvs_to_values.get(x, x)

def stack_as_values(elems, model):
    # Apply as_value_var to each element, then stack
    vals = [as_value_var(e, model) for e in elems]
    return at.stack(vals)


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
