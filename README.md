# OTProf: estimating high-resolution profiles of optical turbulence ($C_n^2$) from reanalysis using deep learning

Code accompanying the publication

> Pierzyna, Maximilian, et al. “OTProf: Estimating High-Resolution Profiles of Optical Turbulence ($C_n^2$) from Reanalysis Using Deep Learning.” arXiv:2604.09346, arXiv, 10 Apr. 2026. arXiv.org, https://doi.org/10.48550/arXiv.2604.09346.

This repository covers the two main parts of the project:

1. **Dataset generation** using the Weather Research and Forecasting model (WRF). The [`wrf-massive`](https://github.com/mpierzyna/wrf-massive) package is used to orchestrate WRF runs to generate one year of high-resolution WRF data for training as detailed in [`data/nl/wrf`](data/nl/wrf). The final processed training dataset is available on [Zenodo](https://zenodo.org/records/20733692).
2. **Deep learning pipeline**. The PyTorch pipeline to train the model is contained in [`code`](code). The trained models and a snapshot of the code is also available on[Zenodo](https://zenodo.org/records/20733692).
