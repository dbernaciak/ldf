# Loss Discounting Framework
## Source code for paper "A loss discounting framework for model averaging and selection in time series models"
The repository contains the python, R and matlab code for the ldf method as well as the reproducibe notebooks for the simulation and empirical studies presented in the paper.

## Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dbernaciak/ldf/main?urlpath=lab)

## NBViewer
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/dbernaciak/ldf/tree/main/notebooks/)

### Code for Beckmann et al.
https://sites.google.com/site/dimitriskorobilis/matlab/dynamic-bayesian-learning
### Code for McAlinn and West (2019) & McAlinn et al. (2020) 
https://www2.stat.duke.edu/~mw/mwsoftware/BPS/index.html
### Code for DeCo "Time-varying combinations of predictive densities using nonlinear filtering" Billio et al. (2013)
http://www.francescoravazzolo.com/pages/DeCo.html

**Note**: the results of the competing models: BPS and DeCo are saved down as the runs take hours/days. The results can be replicated using authors' code linked above.

## Run How-To and Paper Replication
1. In order to launch Jupyter environment press the binder icon above.
   * Binder does not support reading from Git Large File Storage (LFS) so you will not be able to read the large .mat files.
   * If you wish to read the large .mat files you need to clone the repo to your local PC
2. Once in Jupyter environment navigate to the notebooks folder.
3. The following notebook replicate sections of the paper:
   * simulation_study.ipynb - replicates subsection 4.1 "Simulation study" in the paper. In particular, 
     * ```cell 22``` replicates *Figure 1*, 
     * ```cell 24``` replicates *Figure 3*,  
     * ```cell 50``` replicates *Figure 2b*, 
     * ```cell 54``` replicates *Figure 2a*.
* FX_forecasts.ipynb and FX_investment.ipynb replicate subsection 4.2 "Foreign Exchange Forecasts"
  * FX_forecasts.ipynb ```cell 25``` replicates *Figure 4*, 2 upper subfigures and cell 45 (small pool with cell 2 set to ```python log_lik = np.load(r"..//data//FX//uip_l1_save_loglik.npz")["x"]``` and the large with cell 2 ```python log_lik = np.load(r"..//data//FX//full_save_loglik.npz")["x"]```)
  * FX_forecasts.ipynb ```cell 36``` replicates *Figure 5* (cell 2 set to ```python log_lik = np.load(r"..//data//FX//uip_l1_save_loglik.npz")["x"]```)
  * FX_investment.ipynb ```cell 26``` and ```cell 58``` replicate *Figure 6*
  * FX_investment.ipynb ```cell 45``` replicates *Figure 8*
  * FX_investment.ipynb ```cell 65``` replicates *Figure 7*
* US_inflation.ipynb ```cell 27``` replicates *Figure 9* in subsection "US Inflation Forecasts"