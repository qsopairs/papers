""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import numpy as np
import pdb

from scipy.io import readsav

from astropy.coordinates import SkyCoord
from astropy.table import Table

from specdb.specdb import IgmSpec


def tpe_table(qso_fg, qso_bg):
    """ Generate a TPE table given f/g and b/g tables
    Parameters
    ----------
    qso_fg : Table
    qso_bg : Table

    Returns
    -------
    tpe_tbl : Table

    """
    tpe_tbl = Table()
    # Add in f/g
    fg_keys = ['Z_TOT', 'LOGLV', 'G_UV', 'RA', 'DEC']
    for key in fg_keys:
        tpe_tbl['FG_'+key] = qso_fg[key]
    # Add in b/g
    fg_keys = ['Z_TOT', 'RA', 'DEC']
    for key in fg_keys:
        tpe_tbl['BG_'+key] = qso_fg[key]
    # Rename a few
    orig = ['FG_Z_TOT', 'BG_Z_TOT']
    new = ['FG_Z', 'BG_Z']
    for jj,iorig in enumerate(orig):
        tpe_tbl.rename_column(iorig, new[jj])
    # Return
    return tpe_tbl


def make_sample(min_logLV, outfil=None, tpe_sav=None):
    """
    Parameters
    ----------
    min_logLV : float
    outfil : str

    Returns
    -------
    tpe_sample : Table
      Unified table of QSO pairs for TPE
    """
    # Read SAV file
    if tpe_sav is None:
        svfile = os.getenv('DROPBOX_DIR')+'QSOPairs/TPE_DR12/TPE_DR12_Mon-May-16-18:00:17-2016_concat.sav'
        print("Loading save file {:s}".format(svfile))
        print("Be patient....")
        tpe_sav = readsav(svfile)
    # Cosmology dict
    cdict = {}
    for key in ['omega_m', 'omega_v', 'w', 'lit_h']:
        cdict[key] = tpe_sav[key]

    # Tables for convenience
    qso_fg = Table(tpe_sav['qso_fg'])
    qso_bg = Table(tpe_sav['qso_bg'])

    # Luminosity cut
    Lcut = qso_fg['LOGLV'] > min_logLV
    print("{:d} pairs satisfy the Lcut of {:g}".format(np.sum(Lcut), min_logLV))
    qso_fg = qso_fg[Lcut]
    qso_bg = qso_bg[Lcut]

    # Load IgmSpec and QPQ
    igmsp = IgmSpec()
    qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_optical.hdf5'
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)

    # Insist on existing spectra
    b_coords = SkyCoord(ra=qso_bg['RA'], dec=qso_bg['DEC'], unit='deg')
    f_coords = SkyCoord(ra=qso_fg['RA'], dec=qso_fg['DEC'], unit='deg')

    bin_igmsp = igmsp.qcat.match_coord(b_coords) >= 0
    bin_qpq = qpq.qcat.match_coord(b_coords) >= 0
    fin_igmsp = igmsp.qcat.match_coord(f_coords) >= 0
    fin_qpq = qpq.qcat.match_coord(f_coords) >= 0

    fgd = np.any([fin_igmsp, fin_qpq],axis=0)
    print("{:d} f/g quasars in IgmSpec or QPQspec".format(np.sum(fgd)))
    bgd = np.any([bin_igmsp, bin_qpq],axis=0)
    print("{:d} b/g quasars in IgmSpec or QPQspec".format(np.sum(bgd)))

    gd_pairs = fgd & bgd
    print("{:d} good quasar pairs".format(np.sum(gd_pairs)))

    # Generate TPE Table
    tpe_tbl = tpe_table(qso_fg[gd_pairs], qso_bg[gd_pairs])

    # Write
    if outfil is not None:
        print("Writing {:s}".format(outfil))
        tpe_tbl.write(outfil)
    # Return
    return tpe_tbl


# Command line execution
if __name__ == '__main__':

    flg_fig = 0
    flg_fig += 2**0  # Preferred cut

    if (flg_fig % 2**1) >= 2**0:
        #_ = make_sample(31.2, outfil='TPE_DR12_31.2_spec.fits')
        _ = make_sample(31.0, outfil='TPE_DR12_31.0_spec.fits')
