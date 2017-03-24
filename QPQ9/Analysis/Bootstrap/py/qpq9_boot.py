#Module for QPQ9 boot-strapping
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import glob, os, sys, copy
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from astropy.io import ascii, fits
from astropy import units as u
from astropy.modeling import models,fitting
sys.path.append(os.path.abspath("../../../../py"))
from enigma.qpq import stacks as qpqk

#  Bootstrap one transition
def boot_trans(wrest=1334.5323*u.AA,full_sample=True,
               outfil=None,nboot=10000,
               cranges=None,
               stack_tup0=None,
               median=False,
               debug=False):
    ''' Bootstrap on a single transition

    Parameters:
    ----------
    wrest: Quantity, optional
      Rest wavelength [CII 1334]
    nboot: int, optional
      Number of bootstraps [10000]

    Returns:
    --------
    Outputs an image of the bootstraps
    '''

    if outfil is None:
        outfil = 'Output/boot_{:d}.fits'.format(int(wrest.value))

    # For convenience            
    fin_velo, stck_img0, stck_msk0, all_dict0 = stack_tup0
    stck_msk = np.zeros(len(fin_velo))
    stck_img = np.zeros(len(fin_velo))
    all_dict = []
    for ii,idict in enumerate(all_dict0):
        if idict is None:
            continue
        if np.sum(stck_msk0[ii,:]) == 0:
            continue
        stck_msk = np.vstack([stck_msk,stck_msk0[ii,:]])
        stck_img = np.vstack([stck_img,stck_img0[ii,:]])
        all_dict.append(idict)
    stck_msk = stck_msk[1:,:]
    stck_img = stck_img[1:,:]
    stack_tup = fin_velo,stck_img,stck_msk,all_dict

    boot_list = [item for item in stack_tup]

    # Generate bootstrap image
    boot_img = np.zeros((nboot,len(fin_velo)))
    sz = stck_img.shape
    cen = np.zeros(nboot)
    disp = np.zeros(nboot)
    frac = np.zeros(nboot)

    # Loop away (make parallel with map!)
    for qq in range(nboot):
        # Random image
        ran_i = np.random.randint(sz[0], size=sz[0])
        ran_img = stck_img[ran_i,:]
        ran_msk = stck_msk[ran_i,:]

        # Stack
        boot_list[1] = ran_img
        boot_list[2] = ran_msk
        if median is True:
            velo,fx,tdict = qpqk.stack_med(boot_list)
        else:
            velo,fx,tdict = qpqk.stack_avg(boot_list)
        boot_img[qq,:] = fx

        # Gaussian model
        c_init = models.Const1D(amplitude=0.98)
        if wrest == 1334.5323*u.AA:
            g_amp = 0.2
            g_mean = 170.
            g_stddev = 388.
            g_init = models.GaussianAbsorption1D(amplitude=g_amp,mean=g_mean,stddev=g_stddev)
            model_init = c_init*g_init
            model_init.amplitude_1.bounds = [0.,1.]
        if wrest == 1548.195*u.AA:
            g1548_amp = 0.1
            g1548_mean = 60.
            g1548_stddev = 350.
            g1550_amp = g1548_amp*0.5
            g1550_mean = g1548_mean+498.
            g1550_stddev = g1548_stddev
            g1548_init = models.GaussianAbsorption1D(amplitude=g1548_amp,mean=g1548_mean,stddev=g1548_stddev)
            g1550_init = models.GaussianAbsorption1D(amplitude=g1550_amp,mean=g1550_mean,stddev=g1550_stddev)
            model_init = c_init*g1548_init*g1550_init
            model_init.amplitude_1.bounds = [0.,1.]
            def tie_mean2(model):
                mean_2 = model.mean_1 + 498.
                return mean_2
            def tie_stddev2(model):
                stddev_2 = model.stddev_1
                return stddev_2
            def tie_amplitude2(model):
                amplitude_2 = model.amplitude_1
                return amplitude_2
            model_init.mean_2.tied = tie_mean2
            model_init.stddev_2.tied = tie_stddev2
            model_init.amplitude_2.tied = tie_amplitude2
        if wrest == 2796.354*u.AA:
            g2796_amp = 0.1
            g2796_mean = 250.
            g2796_stddev = 250.
            g2803_amp = g2796_amp*0.5
            g2803_mean = g2796_mean+769.
            g2803_stddev = g2796_stddev
            g2796_init = models.GaussianAbsorption1D(amplitude=g2796_amp,mean=g2796_mean,stddev=g2796_stddev)
            g2803_init = models.GaussianAbsorption1D(amplitude=g2803_amp,mean=g2803_mean,stddev=g2803_stddev)
            model_init = c_init*g2796_init*g2803_init
            model_init.amplitude_1.bounds = [0.,1.]
            def tie_mean2(model):
                mean_2 = model.mean_1 + 769.
                return mean_2
            def tie_stddev2(model):
                stddev_2 = model.stddev_1
                return stddev_2
            model_init.mean_2.tied = tie_mean2
            model_init.stddev_2.tied = tie_stddev2
        fit = fitting.LevMarLSQFitter()
        model_final = fit(model_init,velo.value,fx)
        cen[qq] = model_final.mean_1.value
        disp[qq] = model_final.stddev_1.value

        # Pseudo-EW
        ## Continuum ranges
        if cranges is None:
            cranges = [(-3000.,-1300.)*u.km/u.s,(1300.,3000.)*u.km/u.s]
        EW_range = (cranges[0][1],cranges[1][0])
        ## Continuum "fit"
        pix = np.where(((velo >= cranges[0][0]) & (velo <= cranges[0][1])) |
                       ((velo >= cranges[1][0]) & (velo <= cranges[1][1])))[0]
        fit = np.polyfit(velo[pix].value, fx[pix], 0)
        pv = np.poly1d(fit)
        conti = pv(velo.value)
        ## Normalize
        cfx = fx/conti
        EWpix = np.where( (velo > EW_range[0]) & (velo<EW_range[1]))[0]
        EW = np.sum(1. - cfx[EWpix])
        ## Now left/right
        EWl_pix = np.where( (velo >= EW_range[0]) & (velo<0.*u.km/u.s))[0]
        EWr_pix = np.where( (velo < EW_range[1]) & (velo>=0.*u.km/u.s))[0]
        EWl = np.sum(1. - cfx[EWl_pix])
        EWr = np.sum(1. - cfx[EWr_pix])
        ## Equivalent width sknewness
        frac[qq] = (EWr-EWl) / EW

    # Stats
    mu_frac = np.mean(frac)
    med_frac = np.median(frac)
    std_frac = np.std(frac)
    print('Equivalent width skewness: Mean={:g}, Median={:g}, std={:g} for Ntrials={:d}'
          .format(mu_frac,med_frac,std_frac,nboot))
    mu_cen = np.mean(cen)
    mu_disp = np.mean(disp)
    std_cen = np.std(cen)
    std_disp = np.std(disp)
    print('Centroid: Mean = {:g}, scatter={:g}'.format(mu_cen,std_cen))
    print('Dispersion and scatter of it: ',mu_disp,std_disp)
    print('max and min centroid: ',np.max(cen),np.min(cen))

    # Write
    prihdu = fits.PrimaryHDU()
    imghdu = fits.ImageHDU(boot_img)
    frachdu = fits.ImageHDU(frac)
    taucenhdu = fits.ImageHDU(cen)
    taudisphdu = fits.ImageHDU(disp)
    hdu = fits.HDUList([prihdu, imghdu, frachdu, taucenhdu, taudisphdu])
    hdu.writeto(outfil, clobber=True)
    print('Wrote {:s}'.format(outfil))