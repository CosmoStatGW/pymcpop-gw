# Installation

Use **Python 3.12**. The inference requirements pin **JAX/jaxlib 0.4.35** and
**NumPyro 0.19.0**, together with the data and plotting dependencies. Choose
one of the following environments; run these commands from the repository root.

## CPU

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r environment/requirements-cpu.txt
python -m pip check
```

The [CPU requirements](../environment/requirements-cpu.txt) provide the complete
pinned inference environment for Linux x86-64. The code runs directly from
the checkout; there is no separate package-install step.

## NVIDIA GPU

Use a Linux x86-64 machine with an NVIDIA GPU and a CUDA-12-compatible NVIDIA
driver. The driver must be installed on the host; `pip` cannot install it.
The CUDA 12 runtime is supplied by the GPU package dependencies.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r environment/requirements-gpu.txt
python -m pip check
nvidia-smi
```

The [GPU requirements](../environment/requirements-gpu.txt) add the matching
JAX CUDA 12 plugin and runtime package to the pinned inference environment.
They retain the same JAX, jaxlib, and NumPyro versions. The NVIDIA runtime
dependencies are resolved by pip; this file is not a complete CUDA lock file.

On a cluster, run the device check inside a GPU allocation. A login node may
have no visible GPU. Follow the site's instructions for driver/library modules
and avoid loading conflicting CUDA libraries. Consult the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
for driver requirements and platform support. These instructions target CUDA
12; do not substitute the latest CUDA/JAX command without checking compatibility.

## Check the environment

```bash
python - <<'PY'
import jax
import numpyro
print("JAX:", jax.__version__)
print("NumPyro:", numpyro.__version__)
print("Devices:", jax.devices())
PY
```

Expect JAX 0.4.35, NumPyro 0.19.0, and the intended CPU or GPU device.
For GPU execution, require a device with `platform == "gpu"` before submitting
a long run. To select CPU explicitly, prefix the launcher command with
`JAX_PLATFORMS=cpu`. To require CUDA rather than accepting CPU fallback, use
`JAX_PLATFORMS=cuda`.

## Optional event-mixture fitting

Inference from prepared data does not require the fitting dependencies.
To run `fit_posterior_samples.py`, also install scikit-learn and the data readers
from [MGCosmoPop](https://github.com/CosmoStatGW/MGCosmoPop). Follow that project's
installation instructions and verify that `MGCosmoPop.dataStructures` imports.
If using its legacy `dataStructures` layout, add the directory containing that
package to `PYTHONPATH`. See [input preparation](input-data.md).

Next: [prepare inputs, configure, and run an analysis](usage.md).
