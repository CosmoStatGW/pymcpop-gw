# pymcpop-gw


Python package for running GW population inference with Hamiltonian Monte Carlo in the high-dimensional space of individual events and population parameters. Uses ```pymc``` with the possibility of GPU (and not)-accelerated inference with ```jax``` or ```numpyro```.



### Usage

##### Fitting GMM interpolants of posterior samples

The module ```fit_posterior_samples``` fits a GMM on each GW event posterior distribution taken from public LVK data releases. 
This module uses the package [MGCosmoPop](https://github.com/CosmoStatGW/MGCosmoPop) to load the LVK posterior samples. This needs to be cloned in a directory at the same level of pymcpop (but note that is not included in requirements). 

Fits of the 69 events of the GWTC-3 catalog detected with FAR<1/yr will be made available on [Zenodo](https://zenodo.org/records/14826108), together with a version of public LVK software injections to compute the selection effects, ready for the code to run.

More docs to come...

##### Run

To analyze the GWTC-3 catalog with this package (see referenced paper below), download the data products from [Zenodo](https://zenodo.org/records/14826108). 

Then, to run the population-only (fixed cosmology) analysis that reproduces LVK results:

```
> fdata=<path_to_data>
> 
> fout=<name_of_output_folder>
>
> mkdir $fout
>
> python fit_model.py --fin_data $fdata/GWTC-fits_lMlqld_defaultspin_nmax-500/O1O2/snrth-0_farth-1/dil_factor-1/O1O2_ $fdata/GWTC-fits_lMlqld_defaultspin_nmax-500/O3a/snrth-0_farth-1/dil_factor-1/nocosmo/O3a_ $fdata/GWTC-fits_lMlqld_defaultspin_nmax-500/O3b/snrth-0_farth-1/dil_factor-1/nocosmo/O3b_ --fin_injections $fdata/injections_LVK/injections_bbh_rwspin/snrthO1O2-10_farth-1_pycbcbbh_gstlal_mbta/injections_LVK_ --fin_priors='priors_files/priors_GWTC2.json' --rate_model='PL' --mass_model='PLPreg' --spin_model='default' --dLprior='dLsq' --use_sel_spin=1 --sampling_gw='gmm' --ivals='initvals_files/init_GWTC3_lowVar.json' --fout=$fout --sampler='std' --nchains=4 --ncores=4 --target_accept=0.9 --spin_inj='chi12xyz' --sel_uncertainty=0 --sel_smoothing='sigmoid' --min_Neff=0 --log_lik_var_min=1 --alpha_beta_prior='sigmoid' --nsteps=1000 --ntune=500 --fix_H0Om=1 --fix_Xi0n=1
```

To run with varying Hubble constant and matter density parameter, use ```--fix_H0Om=0```


### Sampling with jax and numpyro (needed for GPU)


Use

```
--sampler='jax'
```

or 

```
--sampler='numpyro'
```


pymc 5.10.4 (i.e. the lates stable version on which the code is tested) with pytensor 2.18.6 does not support jax conversion of all operations used in the code. We need to use a modified version located at ```https://github.com/Mik3M4n/pymc```

First create a conda environment and activate it. We directly require git and pip to be installed.

```
conda create -c conda-forge -n pymcpop_env python=3.11.7 git pip 

conda activate pymcpop_env
```

Then, for installation on a Mac OS 14.3.1 with M2 chip, one needs to install the correct version of the ```clang``` compiler. If needed, run 
 ```
 conda install -c conda-forge clangxx_osx-64
```

If your environment/machine has already the good compiler, skip to the next step: install the modified versions of pymc and pytensor. 


```
pip install git+https://github.com/Mik3M4n/pytensor@main

pip install git+https://github.com/Mik3M4n/pymc@main
```

Finally, some other libraries are needed:

```
conda install -c conda-forge jaxlib jax numpyro blackjax corner
```

Note: for some linux machines, I experience issues with the installed version of xarray. In this case I could not fix the issue and had to manually edit the installed version of ```pytensor```.


##### Running on Google Colab

In this case we have to uninstall the existing version of ```pymc``` first, then follow the previous instructions:

```
! pip uninstall pymc pytensor; pip install git+https://github.com/Mik3M4n/pytensor@main; pip install git+https://github.com/Mik3M4n/pymc@main; ! pip install blackjax numpyro corner
```

### Citation

If using this code, please cite this repository and the paper [Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy](<https://arxiv.org/abs/2502.12156>). Bibtex:

```
@article{Mancarella:2025uat,
    author = "Mancarella, Michele and Gerosa, Davide",
    title = "{Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy}",
    eprint = "2502.12156",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "2",
    year = "2025"
}
```