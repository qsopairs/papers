#Module for QPQ9 boot-strapping
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import glob, os, sys, copy
from scipy import stats as scistats
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import QTable
from astropy.io import ascii, fits
from astropy import units as u
from xastropy.xutils import xdebug as xdb
sys.path.append(os.path.abspath("../../../../py"))
from enigma.qpq import stacks as qpqk

#  Bootstrap one transition
def boot_trans(wrest=None,outfil=None,nboot=10000,
    cranges=None,
    stack_tup=None,
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

    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA

    # Continuum ranges
    if cranges is None:
        cranges = [(-3000.,-1300.)*u.km/u.s,(1500.,3000.)*u.km/u.s]
    EW_range = (cranges[0][1],cranges[1][0])

    if outfil is None:
        outfil = 'Output/boot_{:d}.fits'.format(int(wrest.value))

    # For convenience            
    fin_velo, stck_img, stck_msk, all_dict = stack_tup
    boot_list = [item for item in stack_tup]

    # Generate bootstrap image
    boot_img = np.zeros((nboot,len(fin_velo)))
    sz = stck_img.shape
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
        velo,fx,tdict = qpqk.stack_avg(boot_list)
        boot_img[qq,:] = fx

        # Continuum "fit"
        pix = np.where( (velo > cranges[0][0]) & (velo<cranges[0][1]) |
            (velo > cranges[1][0]) & (velo<cranges[1][1]))[0]
        fit = np.polyfit(velo[pix].value, fx[pix], 1)
        pv = np.poly1d(fit)
        conti = pv(velo.value)
        #xdb.xplot(velo, fx, conti)

        # Normalize
        cfx = fx / conti

        # Pseudo-EW
        EWpix = np.where( (velo > EW_range[0]) & (velo<EW_range[1]))[0]
        EW = np.sum(1. - cfx[EWpix])

        # Now left/right
        EWl_pix = np.where( (velo >= EW_range[0]) & (velo<0.*u.km/u.s))[0]
        EWr_pix = np.where( (velo < EW_range[1]) & (velo>=0.*u.km/u.s))[0]
        EWl = np.sum(1. - cfx[EWl_pix])
        EWr = np.sum(1. - cfx[EWr_pix])

        # Frac
        frac[qq] = (EWr-EWl) / EW

    # Check
    if debug:
        xdb.xhist(frac)

    # Stats
    mu = np.mean(frac)
    med = np.median(frac)
    rms = np.std(frac)
    print('Stats:  Mean = {:g}, RMS={:g} for Ntrials={:d}'.format(mu,rms,nboot))

    # Write
    prihdu = fits.PrimaryHDU()
    imghdu = fits.ImageHDU(boot_img)
    frachdu = fits.ImageHDU(frac)
    hdu = fits.HDUList([prihdu, imghdu, frachdu])
    hdu.writeto(outfil, clobber=True)
    print('Wrote {:s}'.format(outfil))