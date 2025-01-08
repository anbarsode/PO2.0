import numpy as np
rng = np.random.default_rng()
from scipy.stats import gaussian_kde as KDE
from scipy.spatial import Delaunay
import pandas as pd

class QuickChecks:
    """
    Author: ankur.barsode
    This class contains a collection of functions that quickly check whether
    a given pair of posterior samples for equal parameters have any overlap or not
    
    posterior_samples1: pandas dataframe for posterior samples of event 1,
        shape: (Nsampl, Nparam), should include equal params, biased params
    posterior_samples2: pandas dataframe for posterior samples of event 2,
        shape: (Nsampl, Nparam), should include equal params, biased params
    eaual_params: list of names of equal params. example: ['mass_1', 'mass_2']
    """
    
    def mass_prior_range_overlap(priordict1, priordict2):
        """
        checks overlap between ranges of Mc, m1, m2 priors of the individual PE runs
        priordict: a bilby.gw.prior.BBHPriorDict
        returns False if there is no overlap
        """
        
        min1 = priordict1['chirp_mass'].minimum
        min2 = priordict2['chirp_mass'].minimum
        max1 = priordict1['chirp_mass'].maximum
        max2 = priordict2['chirp_mass'].maximum
        if min1 <= min2 and max1 <= min2: return False
        if min2 <= min1 and max2 <= min1: return False
        
        min1 = priordict1['mass_1'].minimum
        min2 = priordict2['mass_1'].minimum
        max1 = priordict1['mass_1'].maximum
        max2 = priordict2['mass_1'].maximum
        if min1 <= min2 and max1 <= min2: return False
        if min2 <= min1 and max2 <= min1: return False
    
        min1 = priordict1['mass_2'].minimum
        min2 = priordict2['mass_2'].minimum
        max1 = priordict1['mass_2'].maximum
        max2 = priordict2['mass_2'].maximum
        if min1 <= min2 and max1 <= min2: return False
        if min2 <= min1 and max2 <= min1: return False
    
        return True
    
    def range_overlap(posterior_samples1, posterior_samples2, equal_params):
        """
        checks overlap between ranges of marginalized posteriors of equal parameters
        equal_params: list of param names that are equal in the two posteriors
        returns False if there is no overlap
        """
        
        d1 = posterior_samples1[equal_params]
        d2 = posterior_samples2[equal_params]
        
        min1 = d1.min(axis=0)
        min2 = d2.min(axis=0)
        max1 = d1.max(axis=0)
        max2 = d2.max(axis=0)
        
        ro = not bool(np.sum((max2 < min1) | (max1 < min2)))
        return ro, min1, max1, min2, max2
        
    def histogram1d_overlap(posterior_samples1, posterior_samples2, equal_params):
        """
        checks overlap between 1D histograms of marginalized posteriors of equal parameters
        equal_params: list of param names that are equal in the two posteriors
        returns False if there is no overlap
        
        Rice rule is used to set the number of bins
        """
        
        ro, min1, max1, min2, max2 = QuickChecks.range_overlap(posterior_samples1, posterior_samples2, equal_params)
        lolim = np.minimum(min1, min2)
        uplim = np.maximum(max1, max2)
        if not ro: return ro, lolim, uplim
        
        nbins = int(np.ceil(2.0 * np.min([posterior_samples1.shape[0], posterior_samples2.shape[0]])**(1.0/3.0))) # Rice rule
        for i,ep in enumerate(equal_params):
            bins = np.linspace(lolim[i], uplim[i], nbins)
            h1,_ = np.histogram(posterior_samples1[ep], bins, density=True)
            h2,_ = np.histogram(posterior_samples2[ep], bins, density=True)
            if np.sum(h1 * h2) == 0: return False, lolim, uplim
            else: continue
        return True, lolim, uplim

    def histogram2d_overlap(posterior_samples1, posterior_samples2, equal_params):
        """
        checks overlap between 2D histograms of marginalized posteriors of equal parameters
        equal_params: list of param names that are equal in the two posteriors
        returns False if there is no overlap
        
        Square root rule is used to set the number of bins
        """
        
        h1o, lolim, uplim = QuickChecks.histogram1d_overlap(posterior_samples1, posterior_samples2, equal_params)
        if not h1o: return h1o
        
        # nbins = int(np.ceil(np.min([posterior_samples1.shape[0], posterior_samples2.shape[0]])**0.5)) # Square root rule
        nbins = int(np.ceil(2.0 * np.min([posterior_samples1.shape[0], posterior_samples2.shape[0]])**(1.0/3.0))) # Rice rule
        for i, ep1 in enumerate(equal_params):
            for j, ep2 in enumerate(equal_params[i+1:]):
                binsi = np.linspace(lolim[i], uplim[i], nbins)
                binsj = np.linspace(lolim[i+1+j], uplim[i+1+j], nbins)
                h1,_,_ = np.histogram2d(posterior_samples1[ep1], posterior_samples1[ep2], [binsi, binsj], density=True)
                h2,_,_ = np.histogram2d(posterior_samples2[ep1], posterior_samples2[ep2], [binsi, binsj], density=True)
                if np.sum(h1 * h2) == 0: return False
                else: continue
        return True
    
    def hull2d_overlap(posterior_samples1, posterior_samples2, equal_params, check_ro=True):
        """
        checks overlap between convex hulls of pairwise joint posterior samples of equal parameters
        returns False if there is no overlap
        Must not be used for banana like posteriors
        """
        
        if check_ro:
            ro = QuickChecks.range_overlap(posterior_samples1, posterior_samples2, equal_params)[0]
            if not ro: return ro
        
        Poisson_thresh1 = 1.0 / np.sqrt(posterior_samples1.shape[0])
        Poisson_thresh2 = 1.0 / np.sqrt(posterior_samples2.shape[0])
        
        for i, ep1 in enumerate(equal_params):
            for ep2 in equal_params[i+1:]:
                d1 = posterior_samples1[[ep1, ep2]]
                d2 = posterior_samples2[[ep1, ep2]]
                of1 = np.mean(Delaunay(d2).find_simplex(d1)>=0) # fraction of points in d1 that overlap with convex hull of d2
                of2 = np.mean(Delaunay(d1).find_simplex(d2)>=0)
                if of1 < Poisson_thresh1 or of2 < Poisson_thresh2: return False
        return True
    
    def perform_quickchecks(posterior_samples1, posterior_samples2, theta_unbiased, prior1, prior2, quickcheck):
        if quickcheck == 1: 
            if not QuickChecks.range_overlap(posterior_samples1, posterior_samples2, theta_unbiased)[0]:
                return 0, 1
        if quickcheck == 2:
            if not QuickChecks.histogram1d_overlap(posterior_samples1, posterior_samples2, theta_unbiased)[0]:
                return 0, 2
        if quickcheck == 3:
            if not QuickChecks.histogram2d_overlap(posterior_samples1, posterior_samples2, theta_unbiased):
                return 0, 3
        if quickcheck == 4:
            if not QuickChecks.hull2d_overlap(posterior_samples1, posterior_samples2, theta_unbiased):
                return 0, 4
        if quickcheck == 5:
            if not QuickChecks.mass_prior_range_overlap(prior1, prior2):
                return 0, 5
        if isinstance(quickcheck, list):
            for qc in quickcheck:
                ov, cr = QuickChecks.perform_quickchecks(posterior_samples1, posterior_samples2, theta_unbiased, prior1, prior2, qc)
                if ov == 0: return ov, cr
        return 1, 0

def calc_lensed_evidence(posterior_samples1, posterior_samples2, prior1, prior2, samples_P_HL, theta_unbiased, \
                         morse_factor=0, use_distance=True, deltaT=None, quickcheck=[5,1,2,3], verbose=False, \
                         bw_fac_dist=1, bw_fac_kde2=1, bw_fac_P_HL=1, noPE=False, dL_PE_prior=None):
    ov, cr = QuickChecks.perform_quickchecks(posterior_samples1, posterior_samples2, theta_unbiased, prior1, prior2, quickcheck)
    if ov == 0: return ov, ov, cr
    if verbose: print('Passed quickchecks', flush=True)
    
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
        
    if verbose: print('Created samples2_from1, samples_theta_deltatheta', flush=True)
        
    # begin integration
    if noPE: denom = np.ones((samples1.shape[0]))
    else: denom = prior1.prob(samples1, axis=0) * prior2.prob(samples2_from1, axis=0)
    if dL_PE_prior is not None and use_distance:
        denom = denom * dL_PE_prior(samples1['luminosity_distance']) * \
                 dL_PE_prior(samples2_from1['luminosity_distance']) * \
                 P_samples_sqrt_mu_rel
    nnz_idx = np.where(denom > 0)[0]
    if len(nnz_idx) == 0: return 0, 0, 0
    if verbose: print('PE priors evaluated', flush=True)
    
    num = denom * 0
    kde = KDE(samples2.T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac_kde2)
    num[nnz_idx] = kde(samples2_from1.iloc[nnz_idx].T)
    nnz_idx = np.where(num > 0)[0]
    if len(nnz_idx) == 0: return 0, 0, 0
    if verbose: print('KDE2 evaluated', flush=True)
    
    kde = KDE(samples_P_HL[samples_theta_deltatheta.keys()].T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac_P_HL)
    num[nnz_idx] = num[nnz_idx] * kde(samples_theta_deltatheta.iloc[nnz_idx].T)
    nnz_idx = np.where(num > 0)[0]
    if len(nnz_idx) == 0: return 0, 0, 0
    if verbose: print('P_HL evaluated', flush=True)
    return np.sum(num[nnz_idx] / denom[nnz_idx]) / Nsamples1, np.sum(num[nnz_idx]) / Nsamples1, 0


def calc_unlensed_evidence(posterior_samples, prior_PE, samples_P_HU, theta_unbiased, \
                           morse_factor=0, use_distance=True, bw_fac=1, noPE=False, dL_PE_prior=None):
    theta_posterior = [p for p in theta_unbiased]
    if morse_factor is not None: theta_posterior = theta_posterior + ['phase']
    if use_distance: theta_posterior = theta_posterior + ['luminosity_distance']
    
    Nsamples = posterior_samples.shape[0]
    if noPE: denom = np.ones((Nsamples))
    else: denom = prior_PE.prob(posterior_samples[theta_posterior], axis=0)
    if dL_PE_prior is not None and use_distance:
        denom = denom * dL_PE_prior(posterior_samples['luminosity_distance'])
    nnz_idx = np.where(denom > 0)[0]
    num = denom * 0
    kde = KDE(samples_P_HU[theta_posterior].T)
    kde.set_bandwidth(bw_method=kde.factor * bw_fac)
    num[nnz_idx] = kde(posterior_samples[theta_posterior].iloc[nnz_idx].T)
    nnz_idx = np.where(num > 0)[0]
    return np.sum(num[nnz_idx] / denom[nnz_idx]) / Nsamples, np.sum(num[nnz_idx]) / Nsamples


def calc_RLU(deltaT, Tobs, samples_P_log10_deltaT_HL):
    P_log10_deltaT_lensed = KDE(samples_P_log10_deltaT_HL)(np.log10(deltaT))[0]
    P_log10_deltaT_unlensed = 2.0 * (Tobs - deltaT) * deltaT / Tobs**2.0 * np.log(10.0)
    return P_log10_deltaT_lensed / P_log10_deltaT_unlensed

def calc_histogram2D_overlap(d11, d12, d21, d22, p1=None, p2=None, nbins1=None, nbins2=None):
    if p1 is None or p2 is None:
        if nbins1 is None: nbins1 = int(np.sqrt(np.min([d11.shape[0], d21.shape[0]])))
        if nbins2 is None: nbins2 = int(np.sqrt(np.min([d11.shape[0], d21.shape[0]])))
        nbins2 = int(np.sqrt(np.min([d11.shape[0], d21.shape[0]])))
        bins1 = np.linspace(np.min([np.min(d11), np.min(d21)]), np.max([np.max(d11), np.max(d21)]), nbins1)
        bins2 = np.linspace(np.min([np.min(d12), np.min(d22)]), np.max([np.max(d12), np.max(d22)]), nbins2)
    else:
        if nbins1 is None: nbins1 = int(np.sqrt(np.min([d11.shape[0], d21.shape[0], p1.shape[0]])))
        if nbins2 is None: nbins2 = int(np.sqrt(np.min([d11.shape[0], d21.shape[0], p1.shape[0]])))
        bins1 = np.linspace(np.min([np.min(d11), np.min(d21), np.min(p1)]), np.max([np.max(d11), np.max(d21), np.max(p1)]), nbins1)
        bins2 = np.linspace(np.min([np.min(d12), np.min(d22), np.min(p2)]), np.max([np.max(d12), np.max(d22), np.max(p2)]), nbins2)
    h1,_,_ = np.histogram2d(d11, d12, [bins1, bins2], density=True)
    h2,_,_ = np.histogram2d(d21, d22, [bins1, bins2], density=True)
    if p1 is None or p2 is None:
        return np.sum(h1 * h2) * (bins1[1] - bins1[0]) * (bins2[1] - bins2[0])
    else:
        hp,_,_ = np.histogram2d(p1, p2, [bins1, bins2], density=True)
        return np.sum(h1 * h2 * hp) * (bins1[1] - bins1[0]) * (bins2[1] - bins2[0])
