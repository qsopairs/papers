#Module for QPQ9 boot-strapping
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import glob, os, sys, copy, pdb
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from astropy.io import ascii, fits
from astropy import units as u
from astropy.modeling import models,fitting
sys.path.append(os.path.abspath("../../../../py"))
from enigma.qpq import stacks as qpqk

#  Bootstrap one transition
def boot_trans(outfil=None,nboot=10000,
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

    # Can only do this for CII1334, not doublets.
    wrest = 1334.5323*u.AA
    if outfil is None:
        outfil = 'Output/boot2_{:d}.fits'.format(int(wrest.value))

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
        g_amp = 0.2
        g_mean = 170.
        g_stddev = 388.
        g_init = models.GaussianAbsorption1D(amplitude=g_amp,mean=g_mean,stddev=g_stddev)
        model_init = c_init*g_init
        model_init.amplitude_1.bounds = [0.,1.]
        fit = fitting.LevMarLSQFitter()
        model_final = fit(model_init,velo.value,fx)
        cen[qq] = model_final.mean_1.value
        '''
        ## Continuum ranges
        # cranges = [(-3000.,-1300.)*u.km/u.s,(1300.,3000.)*u.km/u.s]
        cranges = [(-3000.,model_final.mean_1.value-1300.)*u.km/u.s,(model_final.mean_1.value+1300.,3000)*u.km/u.s]
        ## Continuum "fit"
        pix = np.where(((velo >= cranges[0][0]) & (velo <= cranges[0][1])) |
                       ((velo >= cranges[1][0]) & (velo <= cranges[1][1])))[0]
        fit = np.polyfit(velo[pix].value, fx[pix], 0)
        pv = np.poly1d(fit)
        conti = pv(velo.value)
        '''
        ## Normalize
#        bad = np.where(fx > 1.)[0]
#        fx[bad] = 1.
        conti = model_final.amplitude_0.value
        cfx = fx/conti
        ## centroid and dispersion
#        bad = np.where(cfx > 1.)[0]
#        cfx[bad] = 1.
        velo = velo.value
        pix = np.where((velo > (model_final.mean_1.value-1200)) & (velo < (model_final.mean_1.value+1200)))[0]
#        pix = np.where((velo > -1300) & (velo < +1300))[0]
        this_disp = np.sqrt(np.sum((1.-cfx[pix])*(velo[pix]-model_final.mean_1.value)**2.)/np.sum(1.-cfx[pix]))
        disp[qq] = this_disp

    disp = disp[~np.isnan(disp)]
    mu_cen = np.mean(cen)
    mu_disp = np.mean(disp)
    std_cen = np.std(cen)
    std_disp = np.std(disp)
    print('Centroid: Mean = {:g}, scatter={:g}'.format(mu_cen,std_cen))
    print('Dispersion and scatter of it: ',mu_disp,std_disp)
    print('max and min centroid: ',np.max(cen),np.min(cen))

    # Write
    '''
    prihdu = fits.PrimaryHDU()
    imghdu = fits.ImageHDU(boot_img)
    frachdu = fits.ImageHDU(frac)
    taucenhdu = fits.ImageHDU(cen)
    taudisphdu = fits.ImageHDU(disp)
    hdu = fits.HDUList([prihdu, imghdu, frachdu, taucenhdu, taudisphdu])
    hdu.writeto(outfil, clobber=True)
    print('Wrote {:s}'.format(outfil))
    '''