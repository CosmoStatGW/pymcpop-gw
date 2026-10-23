# Run your analysis

This guide takes you from prepared event data to an inspected posterior. Run
commands from the repository root, on a workstation or an allocated cluster node.

## 1. Install and activate an environment

Follow [installation](environment.md) to create either the CPU or NVIDIA GPU
environment. Activate that same environment in every terminal or batch job
that launches the analysis. On a cluster, request the appropriate resources
through your scheduler before running the commands below.

## 2. Prepare the inputs

You need:

- **Event data:** parameter-estimation posterior samples, or event
  Gaussian-mixture fits with matching coordinate/transform metadata.
- **Selection injections:** a compatible injection campaign, including its
  sampling density, generated count, and observing time.
- **Population priors:** a JSON file for your chosen model.

The [input guide](input-data.md) specifies the layouts and event-mixture fitting
entry point. Event and injection datasets are not bundled with this repository.
Raw catalogue downloads must be converted to one of the supported layouts.

The example below uses marginalization over posterior samples, the Double
Power Law plus Double Peak (DPLDP) mass model, a Madau–Dickinson merger rate,
and no population-spin model.

## 3. Write an INI file and choose priors

Copy the complete example and its prior file:

```bash
cp examples/analysis.ini analysis.ini
cp examples/priors/dpldp.json priors.json
```

Open `analysis.ini`. Replace every value in angle brackets:

| Setting | What to provide |
|---|---|
| `fin_data` | Your posterior-sample HDF5 file, or a NumPy filename prefix |
| `fin_injections` | Your injection filename prefix, including its trailing separator |
| `fout` | A new output directory for this analysis |
| `fin_priors` | The prior file you copied and reviewed; initially `priors.json` |

For example, a prefix `/path/to/injections_` identifies
`/path/to/injections_m1d.npy`, `/path/to/injections_m2d.npy`, and the other
injection files. It is not a directory.

The main settings in the example are:

```ini
[settings]
fin_data = <EVENT_POSTERIOR_FILE_OR_PREFIX>
fin_injections = <INJECTION_PREFIX>
fin_priors = priors.json
fout = <OUTPUT_DIRECTORY>

mass_model = DPLDP
rate_model = MD
spin_model = none
pop_only = 1
sampling_gw = samples
ntune = 1000
nsteps = 1500
nchains = 4
chain_method = sequential
```

Use the complete copied file: this excerpt only highlights its main controls.
Review the prior values and sampling settings for your scientific question.
The provided values are a starting point, not a convergence guarantee.
For another model, choose matching priors and settings using
[model support](model-support.md) and [configuration](configuration.md).

Paths resolve from the working directory. Use absolute paths for jobs that
may start elsewhere. Placeholders, `~`, and shell variables inside settings
files are not automatically expanded; replace them with actual paths.

## 4. Run

```bash
python pymcpop-gw/fit_model_numpyro.py --settings analysis.ini
```

Explicit options override the INI file. For example:

```bash
python pymcpop-gw/fit_model_numpyro.py --settings analysis.ini --seed 42
```

For a cluster job, activate the environment, change to the repository root,
and execute the same launcher command in your job script. Output paths must
be writable from the compute node. `sequential` runs chains one after another;
`parallel` needs enough visible devices, and `vectorized` uses additional
memory. Start with `sequential` unless you have configured the other modes.

Use a fresh output directory for each independent analysis. Reusing a directory
can combine earlier trace files with the new run.

## 5. Open and interpret the results

Files are written to `fout`:

| File | Contents |
|---|---|
| `input_args.json` | Effective settings, including command-line overrides |
| `priors.json` | Priors used for this run |
| `logfile.txt` | Setup messages, progress, diagnostics, and errors |
| `trace_0.nc`, `trace_1.nc`, … | Numbered exports of sampling invocations; an export can contain multiple chains |
| `trace.nc` | Aggregate ArviZ trace |
| `trace_0.npz`, … | NumPy arrays of sampled variables |
| `trace.pdf`, `corner_all.pdf` | Summary plots, when plotting succeeds |

A completed sampling loop alone does not confirm successful saving. Check the
log and open the trace. With the output directory substituted below:

```python
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt

output = Path("/path/to/your/results")
idata = az.from_netcdf(output / "trace.nc")
print(idata.groups())
print(idata.posterior.sizes)
print(list(idata.posterior.data_vars))

# Select physical parameters present in this analysis.
parameters = [name for name in ("H0", "Om", "gamma") if name in idata.posterior]
if parameters:
    print(az.summary(idata, var_names=parameters))
    az.plot_trace(idata, var_names=parameters)
    plt.show()

if "diverging" in idata.sample_stats:
    print("Divergent draws:", int(idata.sample_stats["diverging"].sum()))
```

The `posterior` group holds sampled and derived variables with chain/draw
dimensions. Variables ending in `_raw` are sampling coordinates; use the
corresponding physical variables for scientific interpretation. Prior intervals
and posterior credible intervals are different quantities.

Inspect chain mixing, divergences, effective sample size (`ess_bulk`,
`ess_tail`), and between-chain agreement (`r_hat`, requiring multiple chains).
A value of `r_hat` near one is useful but does not establish convergence by
itself. Increase sampling and investigate poor diagnostics before drawing
scientific conclusions. See the [ArviZ diagnostics documentation](https://python.arviz.org/en/v0.22.0/api/diagnostics.html).

## Check your setup before a long run

First run the device check in [installation](environment.md#check-the-environment)
and confirm that all input prefixes, events, priors, and output paths are correct.
Inspect the launcher options without starting inference:

```bash
python pymcpop-gw/fit_model_numpyro.py --help
```

Optionally make a separate smoke-test copy of your settings:

```bash
cp analysis.ini smoke.ini
```

Select a small
event subset with `events_use`, set `nsamplesmax=10` for marginal likelihoods,
and reduce `n_inj_use` for each injection input. Keep the production file intact.
For a single injection input, a Unix/GNU-timeout example is:

```bash
timeout --signal=TERM --kill-after=10s 180s \
  python pymcpop-gw/fit_model_numpyro.py --settings smoke.ini \
  --ntune 2 --nsteps 2 --nchains 1 --chain_method sequential \
  --n_inj_use 0.05
```

Set a distinct `fout` in `smoke.ini`. A timeout reports that the execution
budget was exceeded; it does not establish a model failure. These two-draw
runs check startup and saving, not posterior quality. GP initialization must
still match the selected events and grid: do not subset it arbitrarily.

## Examples from papers

For a published analysis rather than a new configuration, see the
[paper examples](paper-recipes.md). They use the same launcher and placeholder
rules as this guide.
