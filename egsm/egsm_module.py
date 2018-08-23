import numpy as np
import pylab as plt
import healpy as hp
import optparse, sys, os
import numpy.linalg as la
import warnings as wn
import egsm_Helper as h
from NonnegMFPy import nmf 
from astropy import units, constants
import glob
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

EGSM_lowCM = glob.glob('data/low_basismapm*.npz')
EGSM_highCM = glob.glob('data/high_basismapm*.npz')

#EGSM_MONOPOLE = 'data/egsm_monopole.npz'
EGSM_MONOPOLE = 'data/egsm_monopole_mjyr.npz'
EGSM_LOW_DATASET = {'parameters': 'data/egsm_lowparameters.npz', 'nmf_templates': EGSM_lowCM, 'evidence':'data/egsm_lowevidence.npz'}
EGSM_HIGH_DATASET = {'parameters': 'data/egsm_highparameters.npz', 'nmf_templates': EGSM_highCM, 'evidence':'data/egsm_highevidence.npz'}
TEMP_UNITS = ['MJysr', 'TCMB', 'TRJ']
FREQ_UNITS = ['Hz','MHz', 'GHz']
FREQ_CUTOFF = 22.8 # GHz


def default_resolution(freqs):
    try: 
        if len(freqs) > 1:
            freqs = np.array(freqs)
            if all(freqs < FREQ_CUTOFF): return 0.08
            else: return 0.015

    except TypeError:
        if freqs < FREQ_CUTOFF: return 0.08
        else: return 0.015 


def format_newdata(data, error, freqs, theta, filename=None):
    """
    Format the new input data fit for egsm

    Parameters
    data [matrix] : Sky data with a shape (nfreq, npix)
    error [matrix] : Spatial error information of the sky (nfreq, npix)
    freqs [array] : arrays of frequencies in GHz
    theta [array] : Survey resolutions for different frequencies in Radian
    """
    outpath = 'data/'
    nfreq = data.shape[0]
    npix = data.shape[1]
    fbound = np.min(freqs), np.max(freqs)

    assert(error.shape == data.shape)
    assert(len(freqs) == len(theta))
    assert(nfreq == len(freqs))

    if filename is None:
        filename = 'newparameters_%dGHzto%dGHz.npz' %(fbound)
    np.savez(outpath+filename, idata=data, error=error, freqs=freqs, theta=theta)
    print ("Your file is ready for eGSM_core Module")   


class eGSM_map(object):
    """
    Extended Global Sky model class to generate sky model 
    """

    def __init__(self, datafile, freq_unit='MHz', map_unit='TCMB'):
        """
        Extended Global Sky Model (eGSM) class for generation of sky models
        Given pre-computed M matrix and C matrix, computes the model expectation with a correct monopole signal

        Parameters
        dataset : choice between EGSM_LOW_DATASET and EGSM_HIGH_DATASET; if not indicated, will determined based on frequency
        freq_unit [str] : 'Hz', 'MHz', 'GHz' (Default:'MHz')
        map_unit [str] : 'MJysr', 'TCMB', 'TRJ' (Unit of output map; Default:'TCMB')
        resolution [undecided] : Resolution of output map in unit of radian (if smaller than the default value of 0.02 rad, it will be smoothed)
        """

        assert(map_unit in TEMP_UNITS)
        assert(freq_unit in FREQ_UNITS)

        self.freq_unit = freq_unit
        self.map_unit = map_unit
        #self.resolution = resolution

        #self.dataset = os.path.join(DATA_PATH, "egsmodule_test.npz")
            # np.load(file_address+'egsm_data.npz') # should contain monopole+egsm data low and high
        
        self.datafile = datafile
        self.dataset = np.load(self.datafile['parameters'])
        self.data = self.dataset['idata']
        self.norm = h.get_norms(self.data)
        self.offset = np.abs(np.nanmin(self.data)*1.2)
        self.spec = self.dataset['freqs'] ## frequencies of eGSM models

        self.Cmat, self.Mmat = h.read_nmffiles(self.datafile['nmf_templates']) # change ; think about making them dictionary in a file instead
        self.peak_evidence = np.load(self.datafile['evidence'])['peak']   
        self.monofile = np.load(EGSM_MONOPOLE)
        #self.Cmat_mono, self.Mmat_mono, self.monospec = h.read_monopole(file_address) ## monopole C&M&freq ; fix
        self.monospec = self.monofile['freqs']
        self.monopole = self.monofile['mean']
        self.monoerror = self.monofile['error']
        
        #self.Cerror ## MC error
        
        self.nfreq = len(self.Cmat)
        self.generated_map = None
        self.generated_freqs = None


    def generate_map(self, freqs, resol='default', GP=True):
        
        """
        Generate a sky model at a given frequency of User's choice
        
        Parameters
        freqs [float or arrays of float] : a frequency or frequencies of your choice
        resol [float] : angular resolution of the produced map in unit of radian
        GP [bool] : if True, Gaussian Process will be used in interpolation step, if False the old GSM's linear will be implemented.

        Returns
        Output : sky model in healpix format of nside=128(default) ring scheme
        """


        if self.freq_unit == 'GHz':
            freqs = np.array(freqs) * units.Unit(self.freq_unit)
            freqs_ghz = np.array(freqs)
        else:
            freqs = np.array(freqs) * units.Unit(self.freq_unit)
            freqs_ghz = freqs.to('GHz').value

        try:
            assert np.min(freqs_ghz) >= 0.022
            assert np.max(freqs_ghz) <= 3000

        except AssertionError:
            raise RuntimeError('Frequency that you requested is outside of eGSM Model')

        #self.datafile = select_data_file(freqs_ghz)
        if resol is'default':
            resol = default_resolution(freqs_ghz)

        try:
            if len(freqs_ghz) > 1:
                if not(all(freqs_ghz >= FREQ_CUTOFF) or all(freqs_ghz < FREQ_CUTOFF)):
                    raise RuntimeError('Do separate run for low and high frequencies model')
        except TypeError:
            pass
        

        ## Interpolation step

        target_Cmat = []
        target_Csigma = []

        #low_index = np.min(self.peak_evidence) -> with nmfopt skymodel might be tricky
        #high_index = np.max(self.peak_evidence)

        if GP == True:
            for i in range(self.nfreq):
                #Cintp, sigma_intp = h.gp_interpolate_spectrum(freqs_ghz, self.spec, self.Cmat, self.Cerror) 
                Cintp = h.gp_interpolate_spectrum(freqs_ghz, self.spec, self.Cmat[i]) 
                target_Cmat.append(Cintp)
                #target_Csigma.append(sigma_intp)
            interp_norm = h.gp_interpolator(freqs_ghz, self.spec,self.norm, logy=True, option='norm') #use same kernel to ensure this
        else:
            for i in range(self.nfreq):
                Cintp =  h.linear_interpolator(freqs_ghz, self.spec, self.Cmat[i], logx=True)
                target_Cmat.append(Cintp)
            interp_norm = h.linear_interpolator(freqs_ghz, self.spec, self.norm)
        
        sky_model = h.nmfoptimal_skymodel(target_Cmat, self.Mmat, self.peak_evidence)
        target_fit = sky_model.T
        
        target_fit *= interp_norm
        target_fit += self.offset ## return to MJy/sr unit
        

        ### Monopole correction
        monopole_pred = h.gp_interpolator(freqs_ghz, self.monospec, self.monopole, self.monoerror, logy=True, option='monopole')
        #monopole_pred = h.gp_interpolate_monopole(freqs_ghz, self.monospec, self.monopole, self.monoerror) ## change with GP
        output_trans = target_fit - np.nanmean(target_fit) + monopole_pred  ### monopole calibrated result

        if self.map_unit == 'TCMB':
            conversion = 1. / h.K_CMB2MJysr(1., 1e9 * freqs_ghz)
        elif self.map_unit == 'TRJ':
            conversion = 1. / h.K_RJ2MJysr(1., 1e9 * freqs_ghz)
        else:
            conversion = 1.

        output_trans *= conversion
        output = output_trans.T
        self.generated_freqs = freqs

        try:
            if len(self.generated_freqs) > 1:
                if any(freqs_ghz < 22.8) and resol < 0.08:
                    raise RuntimeError('Requested resolution is above eGSM capacity')
                elif all(freqs_ghz < 22.8) and resol >= 0.08:
                    self.generated_map = h.smooth_maps(output, fwhm=resol)
                elif any(freqs_ghz >= 22.8) and resol < 0.015:
                    raise RuntimeError('Requested resolution is above eGSM capacity')  
                elif all(freqs_ghz >= 22.8) and resol >= 0.015:
                    self.generated_map = h.smooth_maps(output, fwhm=resol)
                else:
                    raise RuntimeError('You might be mixing frequencies')
        except TypeError:
            if freqs_ghz < 22.8 and resol < 0.08:
                raise RuntimeError('Requested resolution is above eGSM capacity')
            elif freqs_ghz < 22.8 and resol >= 0.08:
                self.generated_map = h.smooth_maps(output, fwhm=resol)
            elif freqs_ghz >= 22.8 and resol < 0.015:
                raise RuntimeError('Requested resolution is above eGSM capacity')  
            elif freqs_ghz >= 22.8 and resol >= 0.015:
                self.generated_map = h.smooth_maps(output, fwhm=resol)
                
        return self.generated_map

             
    
    def show_map(self, index=0, multiview=False, norm_opt='hist'):
        """
        View the mollweide projection of the generated map

        Parameters
        index [int] : index of map to see. Only required to see one particular map in your generated maps
        multiview [bool] : True if want to see all of generated maps (Default:False)
        norm : 'hist', 'log' color normalization scheme (Default:'None')
        """
        try:
            nfreqs = len(self.generated_freqs)
        except TypeError:
            nfreqs = 1
        
        if self.generated_map is None:
            raise RuntimeError("You haven't generate the map yet")

        if multiview == False:
            if self.generated_map.ndim == 2:
                skymap = self.generated_map[index]
                skyfreq = self.generated_freqs[index]
            else:
                skymap = self.generated_map
                skyfreq = self.generated_freqs

            hp.mollview(skymap, coord='G', norm=norm_opt, title='eGSM at %s %s' %(str(skyfreq), self.map_unit))
            plt.show()

        else:
            if nfreqs == 1:
                skymap = self.generated_map
                skyfreq = self.generated_freqs 
                hp.mollview(skymap, coord='G', norm=norm_opt, title='eGSM at %s %s' %(str(skyfreq), self.map_unit))
                plt.show()
            else:
                for i in range(nfreqs):
                    skymap = self.generated_map[i]
                    skyfreq = self.generated_freqs[i]
                    hp.mollview(skymap, coord='G', norm=norm_opt, title='eGSM at %s %s' %(str(skyfreq), self.map_unit))
                    plt.show() ## think whether plt.show will process after the mollview        

    
    def save_map(self, filename=None, file_ext='npz'):
        """
        Write maps
        Multiple maps will be saved as npz by default
        A single map will be saved as npz if fits option is False

        Parameter:
        filepath [str] : a full file path to save output file
                        (default: None) will save all files on the egsm output directory
        file_ext [str] : save file format
        """
        try:
            nfreqs = len((self.generated_freqs).value)
        except TypeError:
            nfreqs = 1

        if  nfreqs> 1:
            minf = np.min((self.generated_freqs).value)
            maxf = np.max((self.generated_freqs).value)             
            np.savez('output/egsm_multimaps%dto%d%s_%s.npz' %(minf, maxf, self.freq_unit, self.map_unit), freqs=(self.generated_freqs).value, maps=self.generated_map)
        elif nfreqs == 1:
            if filename is None:
                file = 'output/egsm_%d%s_%s' %((self.generated_freqs).value, self.freq_unit, self.map_unit)
            else:
                file = 'output/'+filename
        
            if file_ext == 'npz':
                np.savez(file+'.'+file_ext, idata=self.generated_map)
            elif file_ext == 'fits':
                hp.write_map(file+'.'+file_ext, self.generated_map, column_units=self.map_unit)
            elif file_ext == 'txt':
                np.savetxt(file+'.'+file_ext, self.generated_map)
            else:
                raise RuntimeError("%s format is not supported" %file_ext)

        return print('Map[s] Saved!')



class egsm_CORE(object):
    """
    Extended Global Sky model class for an advanced user
    """

    def __init__(self, datafile, peak_evidence, freq_unit='MHz', map_unit='MJysr', nside=128):
        
        """
        Extended Global Sky Model (eGSM) 
        Given pre-computed M matrix and C matrix, computes the model expectation with a correct monopole signal

        Parameters
        dataset : choice of LOW or HIGH dataset
        evidenc
        freq_unit : 'Hz', 'MHz', 'GHz' (Default:'MHz')
        map_unit : 'MJysr', 'TCMB', 'TRJ' (Unit of output map; Default:'TCMB')
        resolution : Resolution of output map in unit of radian (if smaller than the default value of 0.02 rad, it will be smoothed)
        """
        
        self.datafile = datafile
        self.freq_unit = freq_unit
        self.map_unit = map_unit

        file = np.load(self.datafile)
        self.rawdata = file['idata']
        self.rawerror = file['errors']
        self.spec = file['freqs']
        self.theta = file['theta'] ## fwhm of all claimed survey beams
        
        self.resol = np.nanmin(self.theta)
        self.offset = np.abs(np.nanmin(self.rawdata)*1.2) # arbitary offset
        offsetted_data = self.rawdata + self.offset

        self.data = h.NMFpreprocess(self.rawdata.copy()) ## ready for NMF
        self.norm = h.get_norms(offsetted_data)
        self.error = h.normalize_usingother(self.rawerror, offsetted_data)
        self.Cmat = None
        self.Mmat = None

        self.boolmask = h.boolmask_generator(self.data)
        #self.option = option ## choose between PCA and NMF   

        self.monofile = np.load(EGSM_MONOPOLE)
        #self.Cmat_mono, self.Mmat_mono, self.monospec = h.read_monopole(file_address) ## monopole C&M&freq ; fix
        self.monospec = self.monofile['freqs']
        self.monodata = self.monofile['idata'] #monopole-trusted sky maps
        self.monoerror = self.monofile['error']
        self.monomean = self.monofile['mean']
        
        self.Cmat_mono = None
        self.Mmat_mono = None
        self.monopole = None #monopole signal

        self.peak_evidence = peak_evidence

        self.nside = nside # set
        self.npix = 12*(self.nside**2)
        self.nfreq = len(self.spec)

        self.skypred = None 
        self.generated_map = None
        self.generated_freqs = None


    def compute_NMFtemplates(self, save=False, filename=None):
        """
        Compute the spatial and spectral templates using sequential NMF method
        """

        masked_data = h.nan_masking(self.data)

        zeroth = nmf.NMF(masked_data, V=1/self.error**2, M=self.boolmask, n_components=1)
        zeroth_chi2, zeroth_time = zeroth.SolveNMF()

        C0, M0 = zeroth.W, zeroth.H
        if filename is None:
            outname = 'eGSM_'+'nmftemplates'
        else:
            outname = filename
        if save == True:
            np.savez('output/'+outname+'0.npz', basis=C0, spatial=M0)
            print ('first NMF C&M saved!')
        else:
            pass

        bases = [C0]
        spatial_coeff = [M0]
        chi2 = np.hstack((zeroth_chi2, np.zeros(self.nfreq - 1)))

        for i in range(1, self.nfreq):
            random_throw = np.random.rand(self.nfreq).reshape(self.nfreq, 1)
            previous_basis = np.array(bases[i-1])
            basis_now = np.hstack((previous_basis, random_throw)) 

            module = nmf.NMF(masked_data, V=1/self.error**2, W=basis_now, M=self.boolmask, n_components=i+1)
            module_chi2, module_time = module.SolveNMF()
            chi2[i] = module_chi2
            bases.append(module.W)
            spatial_coeff.append(module.H)
            if save == True:
                np.savez('output/'+outname+'%s.npz' %i, basis=module.W, spatial=module.H)
                print (str(i)+'th C&M saved!')
            else:
                pass

        self.Cmat, self.Mmat = bases, spatial_coeff

        return bases, spatial_coeff


    def compute_monotemplates(self, nmode=6):
        """
        Construct a monopole model using PCA; returns to C&M matrices of the Monopole
        """
        overlap = h.pick_overlap(self.monodata)
        cov = np.dot(overlap, overlap.T)
        ev, basis = la.eig(cov)

        basis_eval = basis[:,0:nmode]
        spatial_coeff = h.find_principalmap(basis_eval, self.monodata)

        self.Cmat_mono = basis_eval.copy()
        self.Mmat_mono = spatial_coeff.copy()

        return basis_eval, spatial_coeff


    def model_monopole(self):
        """
        Compute the monopole
        """
        monobases = h.gp_interpolate_spectrum(self.spec, self.monospec, self.Cmat_mono, option='monopole')
        monopred = np.dot(monobases, (self.Mmat_mono).T)
        regional_monopoles = self.boolmask * monopred
        self.monopole = np.nanmean(regional_monopoles, axis=1)
        return self.monopole


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
    except TypeError:
        nfreq = 1
        
    if nfreq is 1:
        opt_map = np.zeros(npix)
    else:
        opt_map = np.zeros((nfreq, npix))
    
    for i in range(mincomp, maxcomp+1):
        loc = np.where(peak_evidence == i)[0]
        fitmap = np.dot(Cmat[i-1], Mmat[i-1]) ## for 0 represent 1st components
        
        if opt_map.ndim ==1:
            opt_map[loc] = fitmap[loc]
        else:
            opt_map[:,loc] = fitmap[:,loc]

    return opt_map


    def interpolate_sky(self, freqs, resol='default', GP=True):
        """
        Acquire the sky prediction at requested frequencies 
        """

        if self.freq_unit == 'GHz':
            freqs = np.array(freqs) * units.Unit(self.freq_unit)
            freqs_ghz = np.array(freqs)
        else:
            freqs = np.array(freqs) * units.Unit(self.freq_unit)
            freqs_ghz = freqs.to('GHz').value

        try:
            assert np.min(freqs_ghz) >= np.min(self.spec)
            assert np.max(freqs_ghz) <= np.max(self.spec)

        except AssertionError:
            raise RuntimeError('Frequency that you requested is outside of input frequency ranges')

        #self.datafile = select_data_file(freqs_ghz)
        if resol is'default':
            resol = self.resol
        else:
            pass
        ## Interpolation step
        target_Cmat = []
        target_Csigma = []

        if GP == True:
            for i in range(self.nfreq):
                #Cintp, sigma_intp = h.gp_interpolate_spectrum(freqs_ghz, self.spec, self.Cmat, self.Cerror) 
                Cintp = h.gp_interpolate_spectrum(freqs_ghz, self.spec, self.Cmat[i]) 
                target_Cmat.append(Cintp)
                #target_Csigma.append(sigma_intp)
            interp_norm = h.gp_interpolator(freqs_ghz, self.spec,self.norm, logy=True, option='norm') #use same kernel to ensure this
        else:
            for i in range(self.nfreq):
                Cintp =  h.linear_interpolator(freqs_ghz, self.spec, self.Cmat[i], logx=True)
                target_Cmat.append(Cintp)
            interp_norm = h.linear_interpolator(freqs_ghz, self.spec, self.norm)
        
        sky_model = h.nmfoptimal_skymodel(target_Cmat, self.Mmat, self.peak_evidence)
        target_fit = sky_model.T
        
        target_fit *= interp_norm
        target_fit += self.offset ## return to MJy/sr unit
        

        ### Monopole correction
        monopole_pred = h.gp_interpolator(freqs_ghz, self.monospec, self.monomean, self.monoerror, logy=True, option='monopole')
        output_trans = target_fit - np.nanmean(target_fit) + monopole_pred  ### monopole calibrated result

        if self.map_unit == 'TCMB':
            conversion = 1. / h.K_CMB2MJysr(1., 1e9 * freqs_ghz)
        elif self.map_unit == 'TRJ':
            conversion = 1. / h.K_RJ2MJysr(1., 1e9 * freqs_ghz)
        else:
            conversion = 1.

        output_trans *= conversion
        output = output_trans.T
        self.generated_freqs = freqs
            
        if resol < self.resol:
            raise RuntimeError('You requested resolution higher than the convolved resolution')
        else:
            self.generated_map = h.smooth_maps(output, fwhm=resol)

        return self.generated_map


    def generate_MCtemplates(self, seed=None, save=False, outpath=None, filename=None):
        """
        Generate monte-carlo samples using the noise added data
        basis and mapm are arrays of different orders
        fwhm is the final resolution
        """
        if seed is None:
            seed = np.random.random_integers(low=0, high=1e6)
        else:
            pass
        np.random.seed(seed)

        chi2 = []
        bases = []
        spatial_coeff = []

        perturbed = h.perturb_data(self.data, self.error, self.Cmat, self.Mmat, self.peak_evidence, self.theta, self.resol)
        masked_perturbed = h.nan_masking(perturbed)


        for i in range(self.nfreq):
            if i == 0:
                basis_eval = self.Cmat[i].reshape(self.nfreq, 1) ## starting point is fiducial model
            else:
                basis_eval = self.Cmat[i].copy()

            module = nmf.NMF(masked_perturbed, V=1/self.error**2, W=basis_eval, M=self.boolmask, n_components=i+1)
            module_chi2, module_time = module.SolveNMF()
            chi2.append(module_chi2)
            bases.append(module.W)
            spatial_coeff.append(module.H)
            if save == True:    
                if filename is None:
                    outname = 'eGSM_'+'_mctemplates'
                else:
                    outname = filename
                np.savez('output/'+outname+'%s_%d.npz' %(seed, i), basis=module.W, mapm=module.H)
            else:
                pass

        return bases, spatial_coeff


    def get_skyprediction(self):
        """
        Return to interpolated sky and its corresponding frequencies
        """
        return self.generated_map, self.generated_freqs

    def get_Mmat(self, mode):
        return self.Mmat[mode]

    def get_Cmat(self, mode):
        return self.Cmat[mode]


