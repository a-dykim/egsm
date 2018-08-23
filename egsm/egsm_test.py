"""
Test for eGSM 
"""

import nose.tools as nt
import os
import numpy as np
import optparse
import sys
import egsm_evidence as evid
import egsm_module as egsm
import egsm_Helper as h
import warnings as wn
import glob


EGSM_lowCM = glob.glob('data/low_basismapm*.npz')
EGSM_highCM = glob.glob('data/high_basismapm*.npz')

#EGSM_MONOPOLE = 'data/egsm_monopole.npz'
EGSM_MONOPOLE = 'data/egsm_monopole_mjyr.npz'
EGSM_LOW_DATASET = {'parameters': 'data/egsm_lowparameters.npz', 'nmf_templates': EGSM_lowCM, 'evidence':'data/egsm_lowevidence.npz'}
EGSM_HIGH_DATASET = {'parameters': 'data/egsm_highparameters.npz', 'nmf_templates': EGSM_highCM, 'evidence':'data/egsm_highevidence.npz'}
TEMP_UNITS = ['MJysr', 'TCMB', 'TRJ']
FREQ_UNITS = ['Hz','MHz', 'GHz']
FREQ_CUTOFF = 22.8e3 # MHz


class Test_eGSM_map:

    @classmethod
    def setUpClass(self):

        self.low_dataset = EGSM_LOW_DATASET
        self.high_dataset = EGSM_HIGH_DATASET

        lowfile = np.load(self.low_dataset['parameters'])
        highfile = np.load(self.high_dataset['parameters'])
        monofile = np.load(EGSM_MONOPOLE)

        self.lowdata = lowfile['idata']
        self.lownorm = h.get_norms(self.lowdata)
        self.lowoffset = np.abs(np.nanmin(self.lowdata)*1.2)
        self.lowspec = lowfile['freqs']
        self.lowerror = lowfile['errors']
        self.lowtheta = lowfile['theta']
        self.lowC, self.lowM = h.read_nmffiles(self.low_dataset['nmf_templates'])
        self.lowpeak_evidence = np.load(self.low_dataset['evidence'])['peak']

        self.highdata = highfile['idata']
        self.highnorm = h.get_norms(self.highdata)
        self.highoffset = np.abs(np.nanmin(self.highdata)*1.2)
        self.highspec = highfile['freqs']
        self.higherror = highfile['errors']
        self.hightheta = highfile['theta']
        self.highC, self.highM = h.read_nmffiles(self.high_dataset['nmf_templates'])
        self.highpeak_evidence = np.load(self.high_dataset['evidence'])['peak']

        self.monospec = monofile['freqs']
        self.monopole = monofile['idata']
        self.monoerror = monofile['error']

        self.low_nfreq = (self.lowdata).shape[0]
        self.high_nfreq = (self.highdata).shape[0]
        self.mono_nfreq = (self.monopole).shape[0]
        self.npix = (self.highdata).shape[1]

        self.generated_map = None
        self.generated_freqs = None 

    def test_data_shapes(self):
        """
        Test the shape of input data
        """
        nt.assert_equal((self.lowdata).shape[1], (self.highdata).shape[1]) 

        findex = np.random.random_integers(low=2, high=16)
        nt.assert_equal((self.lowC[findex]).shape[0], self.low_nfreq)
        nt.assert_equal((self.lowM[findex]).shape[1], self.npix)
        nt.assert_equal((self.lowC[findex]).shape[1], (self.lowM[findex]).shape[0])
        nt.assert_equal((self.lowC[findex]).shape[0], len(self.lowspec))
        nt.assert_equal(len(self.lowpeak_evidence), self.npix)
        nt.assert_equal((self.lowerror).shape[0], self.low_nfreq)
        nt.assert_equal((self.lowerror).shape[1], self.npix)

        nt.assert_equal((self.highC[findex]).shape[0], self.high_nfreq)
        nt.assert_equal((self.highM[findex]).shape[1], self.npix)
        nt.assert_equal((self.highC[findex]).shape[1], (self.highM[findex]).shape[0])
        nt.assert_equal((self.highC[findex]).shape[0], len(self.highspec))
        nt.assert_equal(len(self.highpeak_evidence), self.npix)
        nt.assert_equal((self.higherror).shape[0], self.high_nfreq)
        nt.assert_equal((self.higherror).shape[1], self.npix)

        nt.assert_equal((self.monodata).shape[0], len(self.monospec))
        #nt.assert_equal((self.monoC).shape[1], (self.monoM).shape[0])

    def test_dataform(self):
        """
        Miscellaneous tests on input 
        """
        ## errorformat
        nt.assert_false(all(self.lowerror == 0))
        nt.assert_false(all(self.higherror == 0))


        ## maximum/minimum check
        HIGH_THRESH = 1e20
        LOW_THRESH = -1e20
        maxout_low = np.where(self.lowdata > HIGH_THRESH)[0] # number based on experience
        maxout_high = np.where(self.highdata > HIGH_THRESH)[0]
        nt.assert_true(not maxout_low) # list is empty?
        nt.assert_true(not maxout_high)

        minout_low = np.where(self.lowdata < LOW_THRESH)[0]
        minout_high = np.where(self.highdata < LOW_THRESH)[0]
        nt.assert_true(not minout_low) 
        nt.assert_true(not minout_high)


    def test_nanANDinf(self):
        """
        NaN is not allowed in some places
        """

        nt.assert_false(all(np.isnan(self.lowerror)))
        nt.assert_false(all(np.isnan(self.lowpeak_evidence)))
        nt.assert_false(all(np.isnan(self.lownorm)))
        nt.assert_false(all(np.isnan(self.lowoffset)))

        nt.assert_false(all(np.isinf(self.lowpeak_evidence)))
        nt.assert_false(all(np.isinf(self.lownorm)))
        nt.assert_false(all(np.isinf(self.lowoffset)))

        nt.assert_false(all(np.isnan(self.higherror)))
        nt.assert_false(all(np.isnan(self.highpeak_evidence)))
        nt.assert_false(all(np.isnan(self.highnorm)))
        nt.assert_false(all(np.isnan(self.highoffset)))

        nt.assert_false(all(np.isinf(self.highpeak_evidence)))
        nt.assert_false(all(np.isinf(self.highnorm)))
        nt.assert_false(all(np.isinf(self.highoffset)))


        for i in range(self.low_nfreq):
            lowfit = np.dot(self.lowC[i], self.lowM[i])
            nt.assert_false(all(np.isnan(lowfit)))
            nt.assert_false(all(np.isinf(lowfit)))

        for j in range(self.high_nfreq):
            highfit = np.dot(self.highC[j], self.highM[j])
            nt.assert_false(all(np.isnan(highfit)))
            nt.asert_false(all(np.isinf(highfit)))


        #monofit = np.dot(self.Cmat_mono, self.Mmat_mono)
        #nt.assert_false(all(np.isnan(monofit)))
        #nt.assert_false(all(np.isinf(monofit)))

    def test_model(self):
        lowest_lowf = np.min(self.lowspec)
        highest_highf = np.max(self.highspec)
        FREQ_CUTOFF = 22.8

        lowf_sample = round(np.random.uniform(lowest_lowf, FREQ_CUTOFF), 2)
        highf_samples = np.random.uniform(FREQ_CUTOFF, highest_highf, size=3)

        lowf = egsm.eGSM_map(self.low_dataset, freq_unit='GHz', map_unit='MJysr')
        highf = egsm.eGSM_map(self.high_dataset, freq_unit='GHz', map_unit='MJysr')

        lowf_output = lowf.generate_map(lowf_sample, GP=False)
        highf_output = highf.generate_map(highf_sample, GP=True)

        nt.assert_true(all(lowf_output)>0)
        nt.assert_true(all(highf_output)>0)




class Test_eGSM_CORE:

    @classmethod
    def setUpClass(self, newdata_filename=None):

        if newdata_filename is None:
            self.file = np.load(EGSM_HIGH_DATASET['parameters'])
        else:
            self.file = np.load('data/'+newdata_filename)

        #self.dataset = os.path.join(DATA_PATH, "actualdata.npz")
        
        self.rawdata =  self.file['idata']
        self.rawerror = self.file['error']
        self.offset = np.abs(np.nanmin(self.data)*1.2)
        self.spec = self.file['freqs'] 
        self.theta = self.file['theta']
        offsetted_data = self.rawdata + self.offset

        self.data = h.NMFpreprocess(self.rawdata.copy()) ## ready for NMF
        self.norm = h.get_norms(offsetted_data)
        self.error = h.normalize_usingother(self.rawerror, offsetted_data)

        self.Cmat, self.Mmat = compute_NMFtemplates()
        self.Cmat_mono, self.Mmat_mono, self.monospec = h.read_monopole()

        self.nfreq = (self.data).shape[0]
        self.npix = (self.data).shape[1]


    def test_data_shapes(self):
        findex = np.random.random_integers(low=2, high=15, size=1)
        nt.assert_equal((self.Cmat[findex]).shape[0], self.nfreq)
        nt.assert_equal((self.Mmat[findex]).shape[0], self.npix)
        nt.assert_equal((self.Cmat[findex]).shape[1], (self.Mmat[findex]).shape[0])
        nt.assert_equal((self.Cmat[findex]).shape[0], len(self.spec))
        nt.assert_equal(len(self.peak_evidence), self.npix)
        nt.assert_equal((self.error).shape[0], self.nfreq)
        nt.assert_equal((self.error).shape[1], self.npix)
        
        nt.assert_equal((self.Cmat_mono).shape[0], len(self.monospec))
        #nt.assert_equal((self.Cmat_mono).shape[1], (self.Mmat_mono).shape[0])
        #nt.assert_equal((self.Mmat_mono).shape[1])

    def test_dataform(self):
        """
        Miscellaneous tests on input 
        """
        ## errorformat
        nt.assert_false(any(self.error == 0))

        ## common coverage 
        common = h.pick_overlap(self.data)
        nt.assert_true(len(common)>0) ## how to not pass if there is none

        ## maximum/minimum check
        HIGH_THRESH = 1e20
        LOW_THRESH = -1e20

        maxout = np.where(self.data > HIGH_THRESH)[0] 
        minout = np.where(self.data < LOW_THRESH)[0]

        nt.assert_true(not maxout)
        nt.assert_true(not minout) 


    def test_nanANDinf(self):
        """
        #NaN is not allowed in some places
        """
        nt.assert_false(all(np.isnan(self.error)))
        nt.assert_false(all(np.isnan(self.peak_evidence)))
        nt.assert_false(all(np.isnan(self.norm)))
        nt.assert_false(all(np.isnan(self.offset)))

        nt.assert_false(all(np.isinf(self.peak_evidence)))
        nt.assert_false(all(np.isinf(self.norm)))
        nt.assert_false(all(np.isinf(self.offset)))


        for j in range(self.nfreq):
            fit = np.dot(self.Cmat[j], self.Mmat[j])
            nt.assert_false(all(np.isnan(fit)))
            nt.asert_false(all(np.isinf(fit)))

        monofit = np.dot(self.Cmat_mono, self.Mmat_mono)
        nt.assert_false(all(np.isnan(monofit)))
        nt.assert_false(all(np.isinf(monofit)))


    def test_masking(self):
        masked = h.nan_masking(self.data)
        infmasked = h.inf_mask(self.error)
        nt.assert_false(all(np.isnan(masked)))
        nt.assert_false(all(np.isinf(infmasked)))


    def test_skymodel(self):
        """
        Test the final sky model
        """ 
        monomodel = egsm_CORE.model_monopole()
        skymodel = egsm_CORE.model_optimalsky()

        if not all(skymodel) > 0:
            wn.warning("Some negative flux in your model is detected", DeprecationWarning)
        else:
            pass

        freq_sample = round(np.random.uniform(np.min(self.spec), np.max(self.spec)), 2)
        sky_interp = egsm.eGSM_CORE.interpolate_sky(freq_sample, resol=np.min(self.theta))

        if not all(sky_interp) > 0:
            wn.warning("Some negative flux is detected in the interpolated sky model", DeprecationWarning)
        else:
            pass



    def test_mcsample(self):
        """
        Check Monte Carlo Samples are actually perturbed
        ## check synfast dependency on random seed
        """

        Nls = f.generate_Nls(theta, model_resol)
        normalization_fac = [f.normfactor_Nl(Nls[j]) for j in range(self.freqs)]
        noise_realization = [hp.synfast(Nls[j], 128, verbose=False) * normalization_fac[j] for j in range(self.freqs)]
        nt.assert_true(all(noise_realization<1))
        nt.assert_true(all(noise_realization>-1))

        perturbed = h.perturb_data(self.data, self.error, self.Cmat, self.Mmat, self.peak_evidence, self.theta, self.resol)
        nt.assert_not_equal(self.data, perturbed)



    def test_hyperparam(self):
        """
        #Check hyperparameters of Bayseian Evidence are behaving as expected
        """
        nsamp = 50
        randompix = np.random.random_integers(0, self.npix, size=nsamp)
        
        alphas = np.zeros((nsamp, self.nfreq))
        #betas = np.zeros((nsamp, self.nfreq))
        for i in range(nsamp):
            alphas[i] = [evid.compute_alpha(self.basis[j], randompix[i]) for j in range(self.nfreq)]## prone to change
            #betas[i] = [evid.compute_beta(self.basis, randompix[i], 1/self.error[randompix[i]]**2, alpha[j,i]) for j in range(self.nfreq)]

        if not all(np.isinf(alphas)):
            infalp = np.where(np.isinf(alphas))
            wn.warning("hyper-parameter alpha is infinite at "+str(infalp), DeprecationWarning)




