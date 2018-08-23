import numpy as np
import healpy as hp
from astropy import units, constants
import numpy.linalg as la
import glob
import numpy.ma as ma
import re
from astropy.io import fits
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kB = 1.38065e-23 # Boltzman coefficients
C = 2.99792e8 # speed of light
h = 6.62607e-34 # planck const
T = 2.725 # CMB temp

#kB = constants.k_B
#C = constants.c
#h = constants.h
#T = 2.725 * units.K
hoverk = h / kB


def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / C)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * C / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return  K_CMB * conversion_factor * 1e20

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / C)**2 * kB
    return  K_RJ * conversion_factor * 1e20


def read_monopole(address):
    """
    Read Monopole data + pre-computed model

    Parameter
    address [str] : file location
    """
    file = glob.glob(address+'*_monopole.npz')
    monopole = np.load(monopole)
    C0, M0, spec = monopole['C0'], monopole['M0'], monopole['spec']
    return C0, M0, spec

def read_nmffiles(addresses):
    """
    Read NMF templates 
    
    address [list] : strings of full filepaths (make sure all files are saved in data)
    """
    sorted(addresses)
    nmfbasis = []
    nmfmcoeff = []
    
    nfreq = len(addresses)
    dir_index = (re.search("/", addresses[0])).start()
    num_index = (re.search("\d", addresses[0])).start()
    
    loc = addresses[0][:dir_index+1]
    file_name = addresses[0][dir_index+1:num_index]
    #ext = addresses[0][(num_index+1):]
    for i in range(nfreq):
        file = np.load(loc+file_name+'%d.npz' %i)
        nmfbasis.append(file['basis'])
        nmfmcoeff.append(file['mapm'])
    return nmfbasis, nmfmcoeff


def nmfoptimal_skymodel(Cmat, Mmat, peak_evidence):
    """
    Compute the optimal sky model based on the peak evidence expectation
    
    Parameters
    Cmat [matrix] : NMF Spectral Template with a shape (nfreq, ncomp)
    Mmat [matrix] : NMF Spatial Templates with a shape (ncomp, npix)
    peak_evidence [array] : Peak Evidence template computed in a given model
    """    
    npix = len(peak_evidence)
    mincomp = int(np.nanmin(peak_evidence))
    maxcomp = int(np.nanmax(peak_evidence))
    try:
        nfreq = len(Cmat[0])
        opt_map = np.zeros((nfreq, npix))

    except TypeError:
        nfreq = 1
        opt_map = np.zeros(npix)
    
    for i in range(mincomp, maxcomp+1):
        loc = np.where(peak_evidence == i)[0]
        fitmap = np.dot(Cmat[i-1], Mmat[i-1]) ## for 0 represent 1st components
        if opt_map.ndim ==1:
            opt_map[loc] = fitmap[loc]
        else:
            opt_map[:,loc] = fitmap[:,loc]

    return opt_map


def denormalize(data, model):
    """
    De-normalize the spectra and remove the artificial offset placed for NMF

    Parameters
    data [matrix] : raw data with no offset
    model [matrix] : final model

    Returns
    model with no offset
    """
    nfreq = data.shape[0]
    offset = np.abs(np.nanmin(data)*1.2)
    norms = np.zeros(nfreq)
    for i in range(nfreq):
        norms[i] = la.norm(data[i][~np.isnan(data[i])])
    denormalized_fit = (norms * model.T).T
    recovered = denormalized_fit - offset
    return recovered


def get_norms(data):
    """
    Compute the normalization factors 

    Parameter
    data [matrix] : input data
    """
    nfreq = data.shape[0]
    norms = np.zeros(nfreq)
    for i in range(nfreq):
        norms[i] = la.norm(data[i][~np.isnan(data[i])])
    return norms


def smooth_maps(maps, fwhm):
    """
    Smooth maps into the common resolution
    """
    if (maps.ndim > 1):
        nfreqs = maps.shape[0]
        smoothed_maps = np.zeros_like(maps)
        for i in range(nfreqs):
            smoothed_maps[i] = hp.smoothing(maps[i], fwhm, verbose=False)
    else:
        smoothed_maps = hp.smoothing(maps, fwhm, verbose=False)
    return smoothed_maps


def pick_overlap(data):
    """
    Find the Common coverage in data sets

    Parameter
    data [matirx] : input data
    """
    npix = data.shape[1]
    complete = []
    for i in range(npix):
        if all(~np.isnan(data[:,i])):
            complete.append(data[:,i])
    return np.array(complete).T


def find_principalmap(Cmat, data):
    """
    Compute the row-by-row least-sq solutions

    Parameter
    Cmat [matrix] : Spectral Template with a shape (nfreq, ncomp)
    data [matrix] : data with a shape (nfreq, npix)
    """
    principal_map =[]
    try:
        npix = data.shape[1]
        for j in range(npix):
            pixel = data[:,j]
            pixel_rm = pixel[~np.isnan(pixel)]
            comp = ma.masked_invalid(np.delete(Cmat, np.argwhere(np.isnan(pixel)),0))
            Ninv = np.eye(len(comp))
            a_i = ma.dot(la.inv(np.dot(np.dot(comp.T, Ninv), comp)).dot(np.dot(comp.T, Ninv)), pixel_rm)
            principal_map.append(a_i)
    except IndexError:
        npix = 1
        pixel = data
        pixel_rm = pixel[~np.isnan(pixel)]
        comp = ma.masked_invalid(np.delete(Cmat, np.argwhere(np.isnan(pixel)),0))
        Ninv = np.eye(len(comp))
        a_i = ma.dot(la.inv(np.dot(np.dot(comp.T, Ninv), comp)).dot(np.dot(comp.T, Ninv)), pixel_rm)
        principal_map.append(a_i)

    return np.array(principal_map)



def ratio_array(data, array):
    """
    Setting the smallest number on matrix comp to be 1, produce ratios; 

    output: if array generate diagonal
    """
    minimum = np.nanmin(array[~np.isnan(data)])
    ratio = array.copy() / minimum
    return ratio


def NMFpreprocess(data):
    """
    Normalize and offset the data to prepare for NMF algorithm
    """
    offset = np.abs(np.nanmin(data)*1.2)
    nan_loc = np.where(np.isnan(data))
    offset_all = data + offset
    offset_all[nan_loc] = np.NaN

    data_normalized =[offset_all[i]/la.norm(offset_all[i][~np.isnan(offset_all[i])]) for i in range(len(offset_all))]

    return np.array(data_normalized)


def interpolate_neighbors(x_target, x, y):
    """
    Mini linear interpolation between two nearby neighboring points

    Parameter
    x_target [float or int]: a target point to be interpolated
    x [arrays] 
    y [arrays] : f(x)
    """
    x = x.flatten()
    try:
        if len(x_target) == 1:
            x_target = x_target[0]
    except TypeError:
        x_target = x_target
        
    smaller_neighbor = min(x, key=lambda x:abs(x-x_target))
    try:
        bigger_neighbor = x[int(np.argwhere(x == smaller_neighbor))+1]
    except IndexError:
        bigger_neighbor = np.max(x)
    if bigger_neighbor < x_target:
        raise RuntimeError("Extrapolation is not allowed")
    else:
        pass
    
    try: 
        ydim = y.shape[1]
        interpolated = np.array([np.interp(x_target, x, y[:,i]) for i in range(ydim)])
    except IndexError:
        y = y.flatten()
        interpolated = np.interp(x_target, x, y)
        
    return interpolated


def gp_interpolate_spectrum(targetfreqs, freqs, spectra, logy=False, error=None, nsamp=1000, fudge_fac=1, option='regular'):
    """
    Gaussian Interpolate something frequency dependent
    Returns to interpolated target value of stuff;
    
    Parameters
    targetfreqs [float, list, array]: frequencies to be interpolated
    freqs [array] : input frequencies
    spectra [matrix] : C matrix with a shape (nfreq, ncomp)
    error [matrix] : error bars of C matrix (use monte carlo)
    nsamp [int] : number of data points to be initially interpolated
    fudge_fac [float] : a multiplicative factor that is customized for the Gaussian Kernel
    option [list] : 'regular' or 'monopole'
    regular is for the main model spectra and monopole is for the monopole spectra

    Returns
    Gaussian Process Interpolated C matrix with a shape (#(targetfreqs), ncomp)
    """
    logfreqs = np.log10(freqs).reshape(-1,1)
    xmin = np.min(logfreqs)
    xmax = np.max(logfreqs)

    if (spectra.shape[1] == 1) or (spectra.ndim == 1):
        ncomp = 1
    else:
        ncomp = spectra.shape[1]
    nfreq = len(logfreqs)
    yfac =spectra.copy() * fudge_fac
    if logy is True:
        ys = np.log10(yfac)
    else:
        ys = yfac

    if option is 'monopole':
        kernel = ConstantKernel() + Matern(length_scale=1) + WhiteKernel()
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    elif option is 'regular':
        kernel = ConstantKernel() + Matern(length_scale=1e-2) #+ WhiteKernel(noise_level=1)
        if error is None:
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        else:
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=error**2)
    else:
        raise RuntimeError("Specify which spectra to be interpolated")
    
    logx_pred = np.linspace(xmin, xmax, nsamp).reshape(-1,1)
    predictions = np.zeros((nsamp, ncomp))
    sigmay = np.zeros((nsamp, ncomp))
    
    for i in range(ncomp):
        y = ys[:,i]
        gp.fit(logfreqs, y)
        predictions[:,i], sigmay[:,i] = gp.predict(logx_pred, return_std=True)
      
    try:    
        n_targets = len(targetfreqs)
        target_pred = np.zeros((n_targets, ncomp))
        for j in range(n_targets):
            target_pred[j] = interpolate_neighbors(targetfreqs[j], 10**(logx_pred), predictions)

    except TypeError:
        target_pred = interpolate_neighbors(targetfreqs, 10**(logx_pred), predictions) 
        
    if logy is True:
        return 10**(target_pred)
    else:
        return target_pred


def gp_interpolator(targetx, x, y, yerr=None, logy=False, option='monopole'):
    """
    GP interpolate 1D array 

    Parameter
    targetx [array or a float]: target x
    option [str]: 'monopole' and 'norm'
    """
    logx = np.log10(x).reshape(-1,1)

    try:
        logtargets = np.log10(targetx).reshape(-1,1)
    except AttributeError:
        logtargets = np.log10(targetx)

    if logy is True:
        yeval = np.log10(y.copy())
        if yerr is not None:
            yerror = yerr ## think more about kernel normalization
            #yerror = yerr / y 
    else:
        yeval = y.copy()
        yerror = yerr.copy()

    if option is 'monopole':
        kernel = ConstantKernel() + Matern(length_scale=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=yerror**2)

    elif option is 'norm':
        kernel = ConstantKernel() + Matern(length_scale=1e-2)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)

    gp.fit(logx, yeval)
    prediction = gp.predict(logtargets)
    if logy is True:
        return 10**prediction
    else:
        return prediction


def linear_interpolator(targetx, x, y, logx=True):
    """
    Linear interpolator
    
    Parameters
    targetx [arrays or a float] : target x
    """
    if logx == True:
        testx = np.log10(x)
        target = np.log10(targetx)
    else:
        testx = x.copy()
        target = targetx
    
    try:
        n_targets = len(targetx)
    except TypeError:
        n_targets = 1

    if y.ndim == 1:
        ydim = 1
        interpy = np.interp(target, testx, y) 
    else:
        ydim = y.shape[1]
        if n_targets == 1:
            interpy = np.array([np.interp(target, testx, y[:,i]) for i in range(ydim)])
        else:
            interpy = np.zeros((n_targets, ydim))
            for i in range(ydim):
                interpy[:,i] = np.interp(target, testx, y[:,i])
    return interpy


def linear_interpolate_spectrum(targetfreqs, freqs, spectra):
    """
    Linear interpolate the spectra

    Parameter:
    targetfreqs [float, list, array]: frequencies to be interpolated
    freqs [array] : input frequencies
    spectra [matrix] : C matrix with a shape (nfreq, ncomp)

    Returns
    Linearly Interpolated C matrix with a shape (#(targetfreqs), ncomp)
    """
    logspec = np.log10(freqs)
    ncomp = spectra.shape[1]
    interpolated = np.zeros(ncomp)
    for i in range(ncomp):
        interpolated[i] = np.interp(np.log10(targetfreqs), freqs, spectra[:,i])
    return interpolated


def get_smoothed_errors(data, fit, error, fwhm=0.04):
    """
    Compute the overall error (fit error + survey error) then smooth

    Parameter
    data [matrix] : input data with a shape (nfreq, npix)
    fit [matrix] : model with a shape (nfreq, npix)
    error [matrix] : spatial error template of the input data
    fwhm [float] : beam size to be convolved (make this larger than convolution beam)

    Returns
    total error template (sum of survey error and fit error) smoothed with a beam size of fwhm
    """

    fiterror = np.sqrt((data-fit)**2)
    nfreq = data.shape[0]
    errorcom = fiterror + error
    errorcom_masked = nan_masking(errorcom)
    errorcom_smoothed = np.zeros_like(data)
    mask0 = boolmask_generator(data)
    for i in range(nfreq):
        errorcom_smoothed[i] = hp.smoothing(errorcom_masked[i], fwhm=fwhm, verbose=False)*mask0[i]
    error_fin = np.abs(errorcom_smoothed)

    return error_fin


def generate_Nls(survey_resol, model_resol, final_nside=128):
    """
    Generate N_ells based on survey resolution and final model resolution

    survey_resol [arrays] : survey resolution
    model_resol [float] : final resolution of the model (usually lowest survey resolution)
    final_nside [int] : final nside of the model
    """
    ell = final_nside*3
    Nls = np.zeros((len(survey_resol), ell))

    for i in range(len(survey_resol)):
        lrange = np.arange(ell)
        Nls[i] = np.exp(lrange*(lrange+1)*(survey_resol[i]**2 - model_resol**2)/(2.35**2))
    return Nls


def normfactor_Nl(Nl, nside=128):
    """
    Compute the normalization factor for given N_ell
    """
    lmax = len(Nl)
    mmax = lmax
    l_index, m_index = hp.sphtfunc.Alm.getlm(lmax-1)
    alm_length = int((mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1))
    alm_norm = np.zeros(alm_length, dtype=np.complex)
    for l in range(lmax):
        lmodes = np.where(l_index==l)
        alm_norm[lmodes] = (2*l+1) * Nl[l]

    norm = np.real(1/(np.sum(alm_norm) / (4*np.pi)))
    return norm


def normalize(data):
    """
    Normalize the data to whiten out the survey discrepancies;
    data has a shape of (n_freq, n_pix) 
    """
    data_normalized =[data[i]/la.norm(data[i][~np.isnan(data[i])]) for i in range(len(data))]
    return np.array(data_normalized)



def nan_masking(data, constant=0):
    """
    Convert NaN to a constant

    Parameter
    data [matrix or array] : input data
    constant [float] : any float to replace the NaN area
    """
    copy = data.copy()
    nan_loc = np.isnan(copy)
    copy[nan_loc] = constant
    return copy

def inf_mask(data, constant=0):
    """
    Mask both NaN and inf and fill it with constant

    Parameter
    data [matrix or array] : input data
    constant [float] : any float to replace the NaN or inf area
    """
    copy = data.copy()
    inf_loc = ~np.isfinite(copy)
    nan_loc = np.isnan(copy)
    copy[inf_loc] = constant
    copy[nan_loc] = constant
    return copy


def boolmask_generator(data):
    """
    Generates boolean mask to take care of NaN
    """
    copy = data.copy()
    nan_loc = ~np.isnan(data)
    return nan_loc


def proj(u, v):
    """
    Normalized projection
    Note: Assumes denominator is not zero
    """
    return u * np.dot(v,u) / np.dot(u,u)


def gram_schmidt(x):
    """
    Orthogonal projection of x vector
    """
    q0 = x[:,0]
    basis = []
    basis.append(q0)
    for k in range(1,len(x[0])):
        q = x[:,k] - np.sum(proj(b,x[:,k]) for b in basis)
        basis.append(q/la.norm(q))
    return basis


def perturb_data(data, error, Cmat, Mmat, peak_evidence, theta, model_resol=0.087):
    """
    Perturb the data with the random noise generated based on the power spectrum 

    Parameters
    data [Matrix] : Sky data with a shape (nfreq, npix)
    error [Matrix] : Spatial Error templates with a shape (nfreq, npix)
    Cmat [Matrix] : NMF Spectral Template with a shape (nfreq, ncomp)
    Mmat [Matrix] : NMF Spatial Templates with a shape (ncomp, npix)
    peak_evidence [array] : Peak Evidence template computed in a given model
    theta [array] : Surveyal Resolutions
    model_resol [float] : Resolution of the Model

    Returns 
    Random noise added data 

    """
    nfreq = len(data)
    random_noise = np.zeros_like(data)
    fiducial = nmfoptimal_skymodel(Cmat, Mmat, peak_evidence)
    error_scale = get_smoothed_errors(data, fiducial, error, model_resol*1.2)
    Nls = generate_Nls(theta, model_resol)

    for j in range(nfreq):
        normalization_fac = normfactor_Nl(Nls[j])
        random_noise = hp.synfast(Nls[j], 128, verbose=False) * error_scale[j] * normalization_fac

    perturbed_data = data + random_noise
    return perturbed_data


def add_offset(data):
    """
    Add offsets to the data set excluding NaN region
    """
    offset = np.abs(np.nanmin(data)*1.2) ## 1.2 determined arbitrarily
    nan_loc = np.where(np.isnan(data))
    offset_all = data + offset
    offset_all[nan_loc] = np.NaN
    return offset_all


def normalize_usingother(one, other):
    """
    Normalize 'one' with respect to 'other'
    """
    normalized = np.zeros_like(one)
    for i in range(len(one)):
        anorm = la.norm(other[i][~np.isnan(other[i])])
        normalized[i] = one[i] / anorm
    return normalized
