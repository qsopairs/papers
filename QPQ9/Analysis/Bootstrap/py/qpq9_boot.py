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
from astropy import constants as const

from xastropy.igm.abs_sys import abssys_utils as abssys
from xastropy import spec as xspec
from xastropy.spec.lines_utils import AbsLine
from xastropy.plotting import utils as xputils
from xastropy.xutils import xdebug as xdb
from xastropy.obs import radec as xor
from xastropy.atomic import ionization as xai

# Local
sys.path.append(os.path.abspath("../py"))
import qpq9_analy as qpq9a
import qpq_spec as qpqs

####
#  Bootstrap one transition
def boot_trans(wrest=None, outfil=None, nboot=10000,
    vmnx = (-3000., 3000.)*u.km/u.s, stack_tup=None,
    passback=False, debug=False, qpq9=None):
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

    # Todo
    #   Have symbol size indicate near-IR for redshift
    reload(qpq9a)

    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA

    # Continuum ranges
    cranges = [ (-2400., -1600)*u.km/u.s, 
        (1600., 2500)*u.km/u.s]
    EW_range = (-1600., 1600)*u.km/u.s

    # Load QPQ9
    if qpq9 is None:
        qpq9 = qpq9a.load_qpq(wrest)

    if outfil is None:
        outfil = 'Output/boot_{:d}.fits'.format(int(wrest.value))

    # Load the stack image
    if stack_tup is None:
        stack_tup = qpq9a.load_stack_img(qpq9, wrest, vmnx=vmnx)
        if passback:
            return stack_tup

    # For convenience            
    fin_velo, stck_img, stck_msk, all_dict = stack_tup
    boot_list = [item for item in stack_tup]

    # Generate bootstrap image
    boot_img = np.zeros( (nboot, len(fin_velo)) )
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
        velo,fx,tdict = qpq9a.stack_avg(boot_list)
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



#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(1)] ))
    else:
        flg_fig = int(flg_fig)

    # Simple bootstrap of CII 1334, restricted to OIII
    if (flg_fig % 2**1) >= 2**0:
        boot_trans() 

    # Simple bootstrap of CII 1334, with any vsig<150 km/s
    if (flg_fig % 2**2) >= 2**1:
        wrest = 1334.5323*u.AA
        qpq9 = qpq9a.load_qpq(wrest, msk_flg=6, vsig_cut=150*u.km/u.s)
        print('QPQ9 has {:d} entries'.format(len(qpq9)))
        boot_trans(qpq9=qpq9, outfil='Output/boot_1334_v150.fits', 
            nboot=1000, debug=True)

    # Simple bootstrap of CIV 1548
    #if (flg_fig % 2**1) >= 2**0:
    #    boot_trans() # CII 1334

    # All done
    print('All done!')


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_fig = 0 
        #flg_fig += 1     # Simple 1334
        flg_fig += 2**1   # Simple 1334, with vsig<150km/s cut
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
