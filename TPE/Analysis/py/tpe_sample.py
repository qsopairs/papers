""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import numpy as np
import pdb

from scipy.io import readsav

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.io import fits

from specdb.specdb import IgmSpec

from enigma.qpq import qpq_query


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
    bg_keys = ['Z_TOT', 'RA', 'DEC']
    for key in bg_keys:
        tpe_tbl['BG_'+key] = qso_bg[key]
    # Deal with b/g 'objects'
    bg_keys = ['LYA_INSTRUMENT', 'LYA_FILE']
    for key in bg_keys:
        # Convert to str array
        tmp = []
        for obj in qso_bg[key]:
            tmp.append(str(obj))
        tpe_tbl['BG_'+key] = np.array(tmp)
    # Rename a few
    orig = ['FG_Z_TOT', 'BG_Z_TOT']
    new = ['FG_Z', 'BG_Z']
    for jj,iorig in enumerate(orig):
        tpe_tbl.rename_column(iorig, new[jj])
    # Return
    return tpe_tbl


def run_wide_query(outfil, R_MAX=15.*u.Mpc):
    # 15 cMpc, no vetting
    qso_fg, qso_bg = qpq_query.run_query(Z_MIN=1.715, Z_MAX=6.0,
                                         R_MIN=0.0*u.Mpc, R_MAX=R_MAX,  # co-moving
                                         VEL_MIN=2000.0*u.km/u.s)
    # Write
    prihdu = fits.PrimaryHDU()
    fhdu = fits.table_to_hdu(qso_fg)
    bhdu = fits.table_to_hdu(qso_bg)
    thdulist = fits.HDUList([prihdu,fhdu,bhdu])
    thdulist.writeto(outfil, overwrite=True)



def make_new_sample(min_logLV, qpq_query_file, outfil=None):
    # Run wide qpq_query
    #
    qso_fg, qso_bg = qpq_query.run_query(Z_MIN=1.715, Z_MAX=6.0,
                                         R_MIN=0.0*u.Mpc, R_MAX=0.10*u.Mpc,  # co-moving
                                         VEL_MIN=2000.0*u.km/u.s)


def make_old_sample(min_logLV, outfil=None, tpe_sav=None):
    """ Generate TPE sample
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
    qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'
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
    print("{:d} good quasar pairs (ie. in both)".format(np.sum(gd_pairs)))

    # Generate TPE Table
    tpe_tbl = tpe_table(qso_fg[gd_pairs], qso_bg[gd_pairs])

    # Write
    if outfil is not None:
        print("Writing {:s}".format(outfil))
        tpe_tbl.write(outfil, overwrite=True)
    # Return
    return tpe_tbl


def tpe_chk_spec(tpe_file):
    """ Check spectrum exists and then look for continuum

    Parameters
    ----------
    tpe_file

    Returns
    -------

    """
    from astropy import units as u

    # Load spectral datasets
    igmsp = IgmSpec()
    qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)
    # Load TPE table
    tpe = Table.read(tpe_file)
    b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    uni_instr = np.unique(tpe['BG_LYA_INSTRUMENT'])

    # Instrument dict
    inst_dict = {}
    for instr in uni_instr:
        inst_dict[instr] = {}
        inst_dict[instr]['NO_CO'] = 0
    inst_dict['LRIS']['GRATING'] = '1200/3400'
    inst_dict['BOSS']['GROUP'] = 'BOSS_DR12'
    inst_dict['SDSS']['GROUP'] = 'SDSS_DR7'

    # Standard process
    # -- Find all b/g spectra covering Lya (igmspec, QPQ): Generate list (INSTR,GRATING)
    # -- Order by: UVES/HIRES/MIKE,
    #              MagE/ESI/XShooter,
    #              LRIS+B1200,  B600?, B400?
    #              MODS?/GMOS+B600?
    #              BOSS/SDSS

    # Query catalogs
    igm_cat = igmsp.qcat.query_coords(b_coords)

    # Scan
    for instr in uni_instr:
        # Parse
        idx = np.where(tpe['BG_LYA_INSTRUMENT'] == instr)[0]
        inst_dict[instr]['NSPEC'] = len(idx)
        gd_b_coords = b_coords[idx]
        # Load
        if instr in ['BOSS', 'SDSS']:
            spec, meta = igmsp.coords_to_spectra(gd_b_coords, inst_dict[instr]['GROUP'], all_spec=False)
        else:
            continue
        # Checks
        for ii,iidx in enumerate(idx):
            lya = (1+tpe['FG_Z'][iidx]) * 1215.67 * u.AA
            spec.select = ii
            iwave = np.argmin(np.abs(spec.wavelength-lya))
            if np.isclose(spec.co[iwave], 0.) or np.isclose(spec.co[iwave],1.):
                print("BG source {} at z={} has no continuum at Lya".format(b_coords[iidx],
                                                                            tpe['FG_Z'][iidx]))
                inst_dict[instr]['NO_CO'] += 1
    pdb.set_trace()

# Command line execution
if __name__ == '__main__':

    flg_fig = 0
    #flg_fig += 2**0  # Preferred cuts
    #flg_fig += 2**1  # Check spectra of TPE sample
    flg_fig += 2**2  # Generate pair tables (15 cMpc)

    if (flg_fig % 2**1) >= 2**0:
        # Load for speed
        svfile = os.getenv('DROPBOX_DIR')+'QSOPairs/TPE_DR12/TPE_DR12_Mon-May-16-18:00:17-2016_concat.sav'
        print("Loading save file {:s}".format(svfile))
        print("Be patient....")
        tpe_sav = readsav(svfile)
        # Generate
        _ = make_old_sample(31.2, outfil='TPE_DR12_31.2_spec.fits', tpe_sav=tpe_sav)
        _ = make_old_sample(31.0, outfil='TPE_DR12_31.0_spec.fits', tpe_sav=tpe_sav)

    if (flg_fig % 2**2) >= 2**1:
        tpe_chk_spec('TPE_DR12_31.2_spec.fits')

    if flg_fig & (2**2):
        run_wide_query('QPQ_v2000_R15_novette.fits', R_MAX=0.1*u.Mpc)
