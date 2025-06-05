# You may refer to Section 5 of https://arxiv.org/pdf/2412.01278 to see what is being calculated
# This is to be run only after a lensed event has already been identified.

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

# Ignoring the non-trivial luminosity_distance (dL) prior may bias magnification posterior,
# So you can save the dL PE prior in a .txt file readable as
# dL, P_dL = np.loadtxt(dL_PE_prior_file).T
# Else set the following to None
dL_PE_prior_file = '/path/to/dL/PE/prior.txt' # or None (may cause biases)

Tobs = 1.5 * 365.25 * 24 * 3600 # Observing duration in seconds. Needed for calculating RLU

outfile = '/path/where/output/will/be/saved.npz'

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
from lensing_utils_PO2 import calc_histogram2D_overlap, calc_RLU
from copy import deepcopy

rng = np.random.default_rng()
from scipy.stats import gaussian_kde as KDE
from scipy.interpolate import CubicSpline

tst = time.time()
use_dL_PE_prior = True
rbw_fac = 1.7


posterior_samples1 = pd.read_feather(posterior_file1)
posterior_samples2 = pd.read_feather(posterior_file2)
prior1, prior2 = None, None
if verbose: print('Loaded posterior samples')

# Z's are the evidences.
# BLU is the final Bayes factor between lensed and unlensed
# B10 is the final Bayes factor between type II and type I
# Everything is clipped to lie between -30 to 30
# posterior contains the posterior samples of the combined PE
Out = {'QuickCheck':None, 'log10_Skyoverlap':None, 'log10_RLU0':None, 'log10_RLU1':None,\
       'log10_ZL0':None, 'log10_ZL1':None, 'log10_ZU1':None, 'log10_ZU2':None,\
       'log10_B10':None, 'log10_BLU':None, 'posterior':None, 'wall_time':None}

# 2D sky overlap, sqrt bins
Skyoverlap = calc_histogram2D_overlap(posterior_samples1['ra'], posterior_samples1['sindec'], \
                                      posterior_samples2['ra'], posterior_samples2['sindec']) * 4 * np.pi
Out['log10_Skyoverlap'] = np.log10(np.clip(Skyoverlap, 1e-30, 1e30))

kde = KDE(posterior_samples2[['ra', 'sindec']].T)
kde.set_bandwidth(bw_method=kde.factor * 0.5)
post2_sky = kde(posterior_samples1[['ra', 'sindec']].T)

if verbose: print('Calculated sky probability and sky overlap:', Out['log10_Skyoverlap'], time.time() - tst)

theta_unbiased = ['mass_1','mass_2','a_1','a_2','cos_theta_jn', 'psi']

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
    if verbose: print('Flipped names', time.time() - tst)

# Lensed evidences
def calc_lensed_posterior(posterior_samples1, posterior_samples2, dL_PE_prior, samples_P_HL, theta_unbiased, \
                          morse_factor=0, use_distance=True, deltaT=None, verbose=False, \
                          bw_fac_dist=1, bw_fac_kde2=1, bw_fac_P_HL=1):
    theta_posterior = [p for p in theta_unbiased]
    if morse_factor is not None: theta_posterior = theta_posterior + ['phase']
    if use_distance: theta_posterior = theta_posterior + ['luminosity_distance']
    
    # load samples
    samples1 = posterior_samples1[theta_posterior]
    samples2 = posterior_samples2[theta_posterior]
    Nsamples1 = samples1.shape[0]
    if verbose: print('Loaded samples', flush=True)
    
    samples2_from1 = samples1.copy()
    if morse_factor is not None:
        # this requires that samples_P_HL correspond to only that morse factor
        samples2_from1['phase'] = samples2_from1['phase'] + np.pi / 2.0 * morse_factor
        if verbose: print('Added phase difference', flush=True)
    
    samples_theta_deltatheta = samples1.copy()
    if deltaT is not None:
        # this requires that samples_P_HL contain samples for log10_deltaT corresponding to the
        # supplied morse factor or for any morse factor if supplied morse_factor is None
        samples_theta_deltatheta['log10_deltaT'] = np.ones((Nsamples1)) * np.log10(deltaT)
        if verbose: print('Added timedelay', flush=True)
    
    if use_distance:
        # this requires that samples_P_HL contain samples for delta_log10_dL corresponding to the
        # supplied morse factor or for any morse factor if supplied morse_factor is None
        choose_idx = rng.integers(0, samples_P_HL.shape[0], (Nsamples1))
        samples_sqrt_mu_rel = np.array(samples_P_HL['sqrt_mu_rel'].iloc[choose_idx])
        samples_log10_sqrt_mu_rel = np.array(samples_P_HL['log10_sqrt_mu_rel'].iloc[choose_idx])
        samples2_from1['luminosity_distance'] = samples2_from1['luminosity_distance'] / samples_sqrt_mu_rel
        samples_theta_deltatheta['log10_sqrt_mu_rel'] = samples_log10_sqrt_mu_rel.copy()
        kde = KDE(samples_P_HL['log10_sqrt_mu_rel'])
        kde.set_bandwidth(bw_method=kde.factor * bw_fac_dist)
        P_samples_sqrt_mu_rel = kde(samples_log10_sqrt_mu_rel)
        if verbose: print('Added distance and magnification', flush=True)
    else: samples_log10_sqrt_mu_rel = None
        
    if verbose: print('Created samples2_from1, samples_theta_deltatheta', flush=True)
        
    # begin integration
    if dL_PE_prior is not None and use_distance:
        P_PE12 = dL_PE_prior(samples1['luminosity_distance']) * \
                 dL_PE_prior(samples2_from1['luminosity_distance']) * \
                 P_samples_sqrt_mu_rel
    else:
        P_PE12 = np.ones((samples1.shape[0]))
    if verbose: print('PE priors evaluated', flush=True)
    
    post2 = P_PE12 * 0
    kde = KDE(samples2.T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac_kde2)
    post2 = kde(samples2_from1.T)
    if verbose: print('KDE2 evaluated', flush=True)
    
    kde = KDE(samples_P_HL[samples_theta_deltatheta.keys()].T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac_P_HL)
    P_pop = kde(samples_theta_deltatheta.T)
    if verbose: print('P_HL evaluated', flush=True)
    
    post_weights = post2 * P_pop / P_PE12
    post_weights[np.isnan(post_weights)] = 0
    return post_weights, samples_log10_sqrt_mu_rel

w0, lsmr0 = calc_lensed_posterior(posterior_samples1, posterior_samples2, dL_PE_prior, \
                                  samples_P_HL0, theta_unbiased, morse_factor=0, use_distance=True, deltaT=deltaT, \
                                  verbose=verbose, bw_fac_dist=1./rbw_fac, bw_fac_kde2=1./rbw_fac, bw_fac_P_HL=1./rbw_fac)
w1, lsmr1 = calc_lensed_posterior(posterior_samples1, posterior_samples2, dL_PE_prior, \
                                  samples_P_HL1, theta_unbiased, morse_factor=1, use_distance=True, deltaT=deltaT, \
                                  verbose=verbose, bw_fac_dist=1./rbw_fac, bw_fac_kde2=1./rbw_fac, bw_fac_P_HL=1./rbw_fac)
ZL0 = np.mean(w0)
Out['log10_ZL0'] = np.log10(np.clip(ZL0, 1e-30, 1e30))
ZL1 = np.mean(w1)
Out['log10_ZL1'] = np.log10(np.clip(ZL1, 1e-30, 1e30))
if verbose: print('Calculated lensed evidences', time.time() - tst)

# RLUs
RLU0 = calc_RLU(deltaT, Tobs, samples_P_HL0['log10_deltaT'])
RLU1 = calc_RLU(deltaT, Tobs, samples_P_HL1['log10_deltaT'])
Out['log10_RLU0'] = np.log10(np.clip(RLU0, 1e-30, 1e30))
Out['log10_RLU1'] = np.log10(np.clip(RLU1, 1e-30, 1e30))
if verbose: print('Calculated RLUs', time.time() - tst)

# Unlensed evidences
def calc_unlensed_evidence(posterior_samples, dL_PE_prior, samples_P_HU, theta_unbiased, \
                           morse_factor=0, use_distance=True, bw_fac=1):
    theta_posterior = [p for p in theta_unbiased]
    if morse_factor is not None: theta_posterior = theta_posterior + ['phase']
    if use_distance: theta_posterior = theta_posterior + ['luminosity_distance']
    
    Nsamples = posterior_samples.shape[0]
    if use_distance and dL_PE_prior is not None:
        denom = dL_PE_prior(posterior_samples['luminosity_distance'])
    else:
        denom = np.ones((Nsamples))
    nnz_idx = np.where(denom > 0)[0]
    num = denom * 0
    kde = KDE(samples_P_HU[theta_posterior].T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac)
    num[nnz_idx] = kde(posterior_samples[theta_posterior].iloc[nnz_idx].T)
    nnz_idx = np.where(num > 0)[0]
    return np.sum(num[nnz_idx] / denom[nnz_idx]) / Nsamples, np.sum(num[nnz_idx]) / Nsamples

ZU1 = calc_unlensed_evidence(posterior_samples1, dL_PE_prior, samples_P_HU, theta_unbiased, morse_factor=0, use_distance=True)[0]
ZU2 = calc_unlensed_evidence(posterior_samples2, dL_PE_prior, samples_P_HU, theta_unbiased, morse_factor=0, use_distance=True)[0]
Out['log10_ZU1'] = np.log10(np.clip(ZU1, 1e-30, 1e30))
Out['log10_ZU2'] = np.log10(np.clip(ZU2, 1e-30, 1e30))
if verbose: print('Calculated unlensed evidences', time.time() - tst)

# Combined BLU
BLU = P_dphi0_HL * ZL0 * RLU0 + (1 - P_dphi0_HL) * ZL1 * RLU1
if BLU != 0: BLU = BLU * Skyoverlap / ZU1 / ZU2
Out['log10_BLU'] = np.log10(np.clip(BLU, 1e-30, 1e30))

# B10
log10_B10 = Out['log10_ZL1'] - Out['log10_ZL0'] + Out['log10_RLU1'] - Out['log10_RLU0'] + np.log10(1 - P_dphi0_HL) - np.log10(P_dphi0_HL)
Out['log10_B10'] = np.clip(log10_B10, -30, 30)

w0 = w0 * post2_sky
w1 = w1 * post2_sky

Out['posterior'] = posterior_samples1.copy()
choose_idx = rng.integers(0, posterior_samples2.shape[0], (posterior_samples1.shape[0]))
Out['posterior']['naive_log10_sqrt_mu_rel0'] = np.log10(posterior_samples1['luminosity_distance'].to_numpy(copy=True) / \
                                                        posterior_samples2['luminosity_distance'].iloc[choose_idx].to_numpy(copy=True))
Out['posterior']['log10_sqrt_mu_rel0'] = lsmr0
Out['posterior']['log10_sqrt_mu_rel1'] = lsmr1
Out['posterior']['weights0'] = w0
Out['posterior']['weights1'] = w1
Out['posterior_keys'] = list(Out['posterior'].keys())
Out['posterior'] = np.array(Out['posterior'])

Out['time'] = time.time() - tst
np.savez(outfile, **Out)
if verbose: print('Calculation complete', Out)

##### Note that the final posterior is not equally weighted #####
# It needs to be reweighted according to the weights stored in columns weights0 and weights0 for each Morse factor like so
'''
D = dict(np.load(outfile))
mur_naive = D['posterior'][:,-5]
mur0 = D['posterior'][:,-4]
mur1 = D['posterior'][:,-3]
w0 = np.clip(D['posterior'][:,-2] / D['posterior'][:,-2].sum(), 0, None)
w1 = np.clip(D['posterior'][:,-1] / D['posterior'][:,-1].sum(), 0, None)

mur_arr = np.linspace(-1,1,300)
P_mur_naive = gaussian_kde(mur_naive)(mur_arr)
P_mur0 = gaussian_kde(mur0, weights=w0)(mur_arr)
P_mur1 = gaussian_kde(mur1, weights=w1)(mur_arr)
morse_weight = 10**D['log10_B10']
P_mur = (P_mur0 / morse_weight + P_mur1 * morse_weight) / (morse_weight + 1. / morse_weight)
'''
