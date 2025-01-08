# You may refer to Appendix C of https://arxiv.org/pdf/2412.01278 to see what is being calculated

### User inputs

# The posterior files should have columns containing samples corresponding to
# 'mass_1','mass_2','a_1','a_2','cos_theta_jn', 'psi', 'ra', 'sindec', 'phase', 'luminsity_distance', 'geocent_time'
posterior_file1 = '/path/to/posterior/samples1.feather'
posterior_file1 = '/path/to/posterior/samples2.feather'

# The unlensed prior file should have columns containing samples corresponding to
# 'mass_1','mass_2','a_1','a_2','cos_theta_jn', 'psi', 'ra', 'sindec', 'phase', 'luminsity_distance', 'geocent_time'
unlensed_priors_file = '/path/to/unlensed/prior/samples.feather'

# The lensed prior file should have columns containing samples corresponding to
# 'mass_1','mass_2','a_1','a_2','cos_theta_jn', 'psi', 'ra', 'sindec', 'phase', 'luminsity_distance', 'geocent_time'
# as well as
# 'morse_factor' which should be 0 for type I and 1 for type II
# 'sqrt_mu_rel' and 'log10_sqrt_mu_rel' defined as dL1/dL2 or its log10
# 'log10_deltaT' of the lensing time delay in seconds
lensed_priors_file = '/path/to/lensed/prior/samples.feather'

# If you do not wish to ignore the non-trivial dL prior, you can save it in a txt file readable as
# dL, P_dL = np.loadtxt(dL_PE_prior_file).T
# Else set the following to None
dL_PE_prior_file = '/path/to/dL/PE/prior.txt' # or None

Tobs = 1.5 * 365.25 * 24 * 3600 # Observing duration in seconds. Needed for calculating RLU

outfile = '/path/where/output/will/be/saved.feather'

verbose = True # Will print progress messages if True

##############

# Ensure only one CPU core is used
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import sys
from lensing_utils_PO2 import *
from copy import deepcopy

tst = time.time()

posterior_samples1 = pd.read_feather(posterior_file1)
posterior_samples2 = pd.read_feather(posterior_file2)
prior1, prior2 = None, None # Ignoring PE priors since they are flat

if verbose: print('Loaded posterior samples')

# Z's are the evidences. BLU is the final Bayes factor including everything.
# Everything is clipped to lie between -30 to 30
Out = {'QuickCheck':None, 'log10_Skyoverlap':None, 'log10_RLU0':None, 'log10_RLU1':None,\
       'log10_ZL0':None, 'log10_ZL1':None, 'log10_ZU1':None, 'log10_ZU2':None,\
       'log10_BLU':None, 'wall_time':None}

# 2D sky overlap, sqrt rule for number of bins
Skyoverlap = calc_histogram2D_overlap(posterior_samples1['ra'], posterior_samples1['sindec'], \
                                      posterior_samples2['ra'], posterior_samples2['sindec']) * 4 * np.pi
Out['log10_Skyoverlap'] = np.log10(np.clip(Skyoverlap, 1e-30, 1e30))
if verbose: print('Calculated sky overlap:', Out['log10_Skyoverlap'], time.time() - tst)

theta_eq = ['mass_1','mass_2','a_1','a_2','cos_theta_jn']

if Skyoverlap == 0: ov, qc = 0, 3
else: ov, qc = QuickChecks.perform_quickchecks(posterior_samples1, posterior_samples2, theta_eq + ['psi'], prior1, prior2, [1,2,3])
Out['QuickCheck'] = qc
if ov == 0:
    if verbose: print('QuickCheck failed', ov, qc)
    Out['log10_BLU'] = -30
    Out['time'] = time.time() - tst
    pd.DataFrame([Out]).to_feather(outfile)
    import sys
    sys.exit()
if verbose: print('Performed quickchecks', time.time() - tst)

samples_P_HU = pd.read_feather(unlensed_priors_file)
samples = pd.read_feather(lensed_priors_file)
idx = samples['morse_factor'] == 0
samples_P_HL0 = samples[idx]
samples_P_HL1 = samples[~idx]
P_dphi0_HL = np.mean(idx)

if dL_PE_prior_file is not None:
    dL, P_dL = np.loadtxt(dL_PE_prior_file).T
    dL_PE_prior = CubicSpline(dL, P_dL)
else:
    dL_PE_prior = None

if verbose: print('Loaded priors', time.time() - tst)

# All lensing quantities are defined such that event "1" arrives first
deltaT = np.median(posterior_samples2['geocent_time']) - np.median(posterior_samples1['geocent_time'])
if deltaT < 0:
    temp = deepcopy(posterior_samples1)
    posterior_samples1 = deepcopy(posterior_samples2)
    posterior_samples2 = deepcopy(temp)
    temp = deepcopy(prior1)
    prior1 = deepcopy(prior2)
    prior2 = deepcopy(temp)
    deltaT = -deltaT
    if verbose: print('Flipped events', time.time() - tst)

# Lensed evidences
ZL = calc_lensed_evidence(posterior_samples1, posterior_samples2, prior1, prior2, \
                           samples_P_HL0, theta_eq, morse_factor=None, use_distance=True, deltaT=deltaT, \
                           quickcheck=[], verbose=verbose, noPE=True, dL_PE_prior=dL_PE_prior)[0]
ZL0 = ZL * calc_lensed_evidence(posterior_samples1, posterior_samples2, prior1, prior2, \
                               samples_P_HL0, ['psi'], morse_factor=0, use_distance=False, deltaT=deltaT, \
                               quickcheck=[], verbose=verbose, noPE=True)[0]
ZL1 = ZL * calc_lensed_evidence(posterior_samples1, posterior_samples2, prior1, prior2, \
                               samples_P_HL1, ['psi'], morse_factor=1, use_distance=False, deltaT=deltaT, \
                               quickcheck=[], verbose=verbose, noPE=True)[0]
Out['log10_ZL0'] = np.log10(np.clip(ZL0, 1e-30, 1e30))
Out['log10_ZL1'] = np.log10(np.clip(ZL1, 1e-30, 1e30))
if verbose: print('Calculated lensed evidences', time.time() - tst)

# Unlensed evidences
ZU1 = calc_unlensed_evidence(posterior_samples1, prior1, samples_P_HU, theta_eq, morse_factor=None, use_distance=True, noPE=True, dL_PE_prior=dL_PE_prior)[0] * \
      calc_unlensed_evidence(posterior_samples1, prior1, samples_P_HU, ['psi'], morse_factor=0, use_distance=False, noPE=True)[0]
ZU2 = calc_unlensed_evidence(posterior_samples2, prior2, samples_P_HU, theta_eq, morse_factor=None, use_distance=True, noPE=True, dL_PE_prior=dL_PE_prior)[0] * \
      calc_unlensed_evidence(posterior_samples2, prior2, samples_P_HU, ['psi'], morse_factor=0, use_distance=False, noPE=True)[0]
Out['log10_ZU1'] = np.log10(np.clip(ZU1, 1e-30, 1e30))
Out['log10_ZU2'] = np.log10(np.clip(ZU2, 1e-30, 1e30))
if verbose: print('Calculated unlensed evidences', time.time() - tst)

# RLUs
RLU0 = calc_RLU(deltaT, Tobs, samples_P_HL0['log10_deltaT'])
RLU1 = calc_RLU(deltaT, Tobs, samples_P_HL1['log10_deltaT'])
Out['log10_RLU0'] = np.log10(np.clip(RLU0, 1e-30, 1e30))
Out['log10_RLU1'] = np.log10(np.clip(RLU1, 1e-30, 1e30))
if verbose: print('Calculated RLUs', time.time() - tst)

# Combined BLU
BLU = P_dphi0_HL * ZL0 * RLU0 + (1 - P_dphi0_HL) * ZL1 * RLU1
if BLU != 0: BLU = BLU * Skyoverlap / ZU1 / ZU2
Out['log10_BLU'] = np.log10(np.clip(BLU, 1e-30, 1e30))

Out['time'] = time.time() - tst
pd.DataFrame([Out]).to_feather(outfile)
if verbose: print('Calculation complete:', Out)
