""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import numpy as np
import pdb

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

from linetools.spectra import utils as lspu

from specdb.specdb import IgmSpec
from specdb import cat_utils

this_file = __file__


def tpe_stack_boss(dv=100*u.km/u.s):
    """ Testing stacks with BOSS
    """
    # Load sample
    ipos = this_file.rfind('/')
    if ipos == -1:
        path = './'
    else:
        path = this_file[0:ipos]
    tpe = Table.read(path+'/../TPE_DR12_31.2_spec.fits')
    # Load spectra
    igmsp = IgmSpec()
    # Coordiantes
    b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    f_coords = SkyCoord(ra=tpe['FG_RA'], dec=tpe['FG_DEC'], unit='deg')

    # Cut on impact parameter and BOSS
    kpc_amin = cosmo.kpc_comoving_per_arcmin(tpe['FG_Z'])  # kpc per arcmin
    ang_seps = b_coords.separation(f_coords)
    rho = ang_seps.to('arcmin') * kpc_amin / (1+tpe['FG_Z'])

    cut_Rboss = (rho.to('Mpc').value < 4) & (tpe['BG_LYA_INSTRUMENT'] == 'BOSS') & (
        tpe['FG_Z'] > 2.) # Some of these have too low z (just barely)

    # Cut
    gd_b_coords = b_coords[cut_Rboss]
    gd_f_coords = f_coords[cut_Rboss]
    gd_tpe = tpe[cut_Rboss]

    # Grab these spectra from igmsp
    #   For boss, we are ok taking the first entry of each
    #   The returned set is aligned with the input coords
    spec,meta = igmsp.coords_to_spectra(gd_b_coords, 'BOSS_DR12', all_spec=False)

    # Check for continua
    has_co = np.array([True]*spec.nspec)
    for ii in range(spec.nspec):
        # Select
        spec.select = ii
        # Match to lya
        lya = (1+gd_tpe['FG_Z'][ii]) * 1215.67 * u.AA
        iwave = np.argmin(np.abs(spec.wavelength-lya))
        # Check for co
        #coval = spec.co[iwave]
        #print('spec: {:d} with co={:g}'.format(ii, coval))
        if np.isclose(spec.co[iwave], 0.) or np.isclose(spec.co[iwave],1.):
            has_co[ii] = False

    # Slice to good co
    print("{:d} BOSS spectra with a continuum".format(np.sum(has_co)))
    co_spec = spec.slice(has_co)
    co_spec.normed = True  # Apply continuum

    # NEED TO ZERO OUT REGIONS WITHOUT CONTINUUM
    #  May also wish to isolate in wavelength to avoid rejected pixels
    for ii in range(co_spec.nspec):
        co_spec.select = ii
        co = co_spec.co.value
        bad_pix = np.any([(co == 0.),(co == 1.)], axis=0)
        co_spec.add_to_mask(bad_pix, compressed=True)

    # Rebin to rest
    zarr = gd_tpe['FG_Z'][has_co]
    rebin_spec = lspu.rebin_to_rest(co_spec, zarr, dv)

    # Stack
    stack = lspu.smash_spectra(rebin_spec)

    # Plot
    plot_stack(stack, 'BOSS_stack.pdf')

    return stack


def tpe_stack_lris(dv=100*u.km/u.s):
    """ Testing stacks with LRIS
    """
    # Load sample
    ipos = this_file.rfind('/')
    if ipos == -1:
        path = './'
    else:
        path = this_file[0:ipos]
    tpe = Table.read(path+'/../TPE_DR12_31.2_spec.fits')
    # Load spectra
    # Coordiantes
    b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    f_coords = SkyCoord(ra=tpe['FG_RA'], dec=tpe['FG_DEC'], unit='deg')

    # Cut on impact parameter and BOSS
    kpc_amin = cosmo.kpc_comoving_per_arcmin(tpe['FG_Z'])  # kpc per arcmin
    ang_seps = b_coords.separation(f_coords)
    rho = ang_seps.to('arcmin') * kpc_amin / (1+tpe['FG_Z'])

    cut_Rlris = (rho.to('Mpc').value < 4) & (tpe['BG_LYA_INSTRUMENT'] == 'LRIS')# & (
        #tpe['FG_Z'] > 2.) # Some of these have too low z (just barely)

    # Cut
    gd_b_coords = b_coords[cut_Rlris]
    gd_tpe = tpe[cut_Rlris]

    # Grab these spectra from QPQ
    #   For boss, we are ok taking the first entry of each
    #   The returned set is aligned with the input coords
    qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)

    IDs = qpq.qcat.match_coord(gd_b_coords, group='LRIS')
    meta = qpq['LRIS'].meta
    gcut = meta['GRATING'] == '1200/3400'  # There is one with B400
    B1200 = np.in1d(IDs, meta['PRIV_ID'][gcut])
    print("There are {:d} sources without B1200".format(np.sum(~B1200)))
    # Cut again
    gd_b_coords = gd_b_coords[B1200]
    gd_tpe = gd_tpe[B1200]
    gd_IDs = IDs[B1200]

    # Find the rows
    idx = cat_utils.match_ids(gd_IDs, meta['PRIV_ID'])
    rows = meta['GROUP_ID'][idx]  # Will change to GROUP_ID
    pdb.set_trace()

    spec,meta = qpq.coords_to_spectra(gd_b_coords, 'LRIS', all_spec=False)

    # Check for continua
    has_co = np.array([True]*spec.nspec)
    for ii in range(spec.nspec):
        # Select
        spec.select = ii
        # Match to lya
        lya = (1+gd_tpe['FG_Z'][ii]) * 1215.67 * u.AA
        iwave = np.argmin(np.abs(spec.wavelength-lya))
        # Check for co
        #coval = spec.co[iwave]
        #print('spec: {:d} with co={:g}'.format(ii, coval))
        if np.isclose(spec.co[iwave], 0.) or np.isclose(spec.co[iwave],1.):
            has_co[ii] = False

    # Slice to good co
    print("{:d} BOSS spectra with a continuum".format(np.sum(has_co)))
    co_spec = spec.slice(has_co)
    co_spec.normed = True  # Apply continuum

    # NEED TO ZERO OUT REGIONS WITHOUT CONTINUUM
    #  May also wish to isolate in wavelength to avoid rejected pixels
    for ii in range(co_spec.nspec):
        co_spec.select = ii
        co = co_spec.co.value
        bad_pix = np.any([(co == 0.),(co == 1.)], axis=0)
        co_spec.add_to_mask(bad_pix, compressed=True)

    # Rebin to rest
    zarr = gd_tpe['FG_Z'][has_co]
    rebin_spec = lspu.rebin_to_rest(co_spec, zarr, dv)

    # Stack
    stack = lspu.smash_spectra(rebin_spec)

    # Plot
    plot_stack(stack, 'BOSS_stack.pdf')

    return stack


def plot_stack(stack, outfil):
    # Quick Plot
    plt.clf()
    ax = plt.gca()
    ax.plot(stack.wavelength, stack.flux, 'k', drawstyle='steps-mid')
    ax.set_xlabel('Rest wavelength')
    ax.set_ylabel('Normalized flux')
    ax.set_xlim(1170., 1270.)
    ax.set_ylim(0.35, 1.0)
    # Lya
    ax.plot([1215.67]*2, [0., 2.], 'g--')
    plt.savefig(outfil)

# Command line execution (Run above py!!)
if __name__ == '__main__':

    flg_stack = 0
    #flg_stack += 2**0  # BOSS test
    flg_stack += 2**1  # LRIS

    if (flg_stack % 2**1) >= 2**0:
        tpe_stack_boss()

    if (flg_stack % 2**2) >= 2**1:
        tpe_stack_lris()

