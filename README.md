# PO2.0
_Code for a fast and efficient Bayesian method to search for strongly lensed gravitational waves_

This repo contains the main python scripts that can be used to calculate the strong lensing Bayes factor $\mathcal{B}^L_U$ for a pair of gravitational wave signals' posterior distributions, as described in https://arxiv.org/abs/2412.01278.

### Requirements
`numpy`, `scipy`, `pandas`, `pyarrow` (for `.feather` files)

### Usage
For calculating the Bayes factor, you may simply follow the comments in `calculate_PO2.0_Bayes_factor.py` and edit the file paths to match your data.

If a lensed event has been identified, you may obtain a combined posterior over its parameters using `lensed_parameter_estimation.py`
