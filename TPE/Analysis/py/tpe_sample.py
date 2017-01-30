""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os, sys
import numpy as np
import pdb

from scipy.io import readsav

from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo

from specdb.specdb import IgmSpec

from enigma.qpq import qpq_query
qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'

sys.path.append(os.path.abspath("./py"))
import tpe_stack as tstack
import tpe_plots as tplots


def tpe_table(qso_fg, qso_bg, old=False):
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
    if old:
        fg_keys = ['Z_TOT', 'LOGLV', 'G_UV', 'RA', 'DEC']
        bg_keys = ['Z_TOT', 'RA', 'DEC']
    else:
        fg_keys = ['ZEM', 'LOGLV', 'RA_ALLQ', 'DEC_ALLQ', 'IGM_ID', 'QPQ_ID', 'MYERS_ZEM_SOURCE']
        bg_keys = ['ZEM', 'RA_ALLQ', 'DEC_ALLQ', 'IGM_ID', 'QPQ_ID']
    for key in fg_keys:
        tpe_tbl['FG_'+key] = qso_fg[key]
    # Add in b/g
    for key in bg_keys:
        tpe_tbl['BG_'+key] = qso_bg[key]
    # Deal with b/g 'objects'
    if old:
        bg_keys = ['LYA_INSTRUMENT', 'LYA_FILE']
        for key in bg_keys:
            # Convert to str array
            tmp = []
            for obj in qso_bg[key]:
                tmp.append(str(obj))
            tpe_tbl['BG_'+key] = np.array(tmp)
    # Rename a few
    if old:
        orig = ['FG_Z_TOT', 'BG_Z_TOT']
        new = ['FG_Z', 'BG_Z']
    else:
        orig = ['FG_ZEM', 'BG_ZEM', 'FG_RA_ALLQ', 'FG_DEC_ALLQ', 'BG_RA_ALLQ', 'BG_DEC_ALLQ']
        new = ['FG_Z', 'BG_Z', 'FG_RA', 'FG_DEC', 'BG_RA', 'BG_DEC']
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


def make_tpe(min_logLV, R, qso_fg=None, qso_bg=None, outfil=None, qpqquery_file=None):
    """ Generate TPE sample from QPQ query
    Parameters
    ----------
    min_logLV : float
    R : float
      Cut in physical Mpc
    outfil : str

    Returns
    -------
    tpe_sample : Table
      Unified table of QSO pairs for TPE
    """
    if qso_fg is None:
        if 'sav' in qpqquery_file: # Read SAV file
            svfile = os.getenv('DROPBOX_DIR')+'QSOPairs/'+qpqquery_file #TPE_DR12/TPE_DR12_Mon-May-16-18:00:17-2016_concat.sav'
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
        else:
            qso_fg = Table.read(qpqquery_file, hdu=1)
            qso_bg = Table.read(qpqquery_file, hdu=2)

    # Luminosity cut
    if isinstance(min_logLV, float):
        Lcut = qso_fg['LOGLV'] > min_logLV
        print("{:d} pairs satisfy the Lcut of {:g}".format(np.sum(Lcut), min_logLV))
    elif isinstance(min_logLV, (list,tuple)):
        Lcut = (qso_fg['LOGLV'] > min_logLV[0]) & (qso_fg['LOGLV'] < min_logLV[1])
        print("{:d} pairs satisfy the Lcut of [{:g},{:g}]".format(np.sum(Lcut),
                                                                  min_logLV[0], min_logLV[1]))
    qso_fg = qso_fg[Lcut]
    qso_bg = qso_bg[Lcut]

    # Load IgmSpec and QPQ
    igmsp = IgmSpec()
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)

    # Insist on existing spectra
    b_coords = SkyCoord(ra=qso_bg['RA_ALLQ'], dec=qso_bg['DEC_ALLQ'], unit='deg')
    f_coords = SkyCoord(ra=qso_fg['RA_ALLQ'], dec=qso_fg['DEC_ALLQ'], unit='deg')

    bin_igmsp, _, bin_igmID = igmsp.qcat.query_coords(b_coords)
    bin_qpq, _, bin_qpqID = qpq.qcat.query_coords(b_coords)
    fin_igmsp, _, fin_igmID = igmsp.qcat.query_coords(f_coords)
    fin_qpq, _, fin_qpqID = qpq.qcat.query_coords(f_coords)

    # Add IDs
    qso_bg['IGM_ID'] = bin_igmID
    qso_bg['QPQ_ID'] = bin_qpqID
    qso_fg['IGM_ID'] = fin_igmID
    qso_fg['QPQ_ID'] = fin_qpqID

    fgd = np.any([fin_igmsp, fin_qpq],axis=0)
    print("{:d} f/g quasars in IgmSpec or QPQspec".format(np.sum(fgd)))
    bgd = np.any([bin_igmsp, bin_qpq],axis=0)
    print("{:d} b/g quasars in IgmSpec or QPQspec".format(np.sum(bgd)))

    gd_pairs = fgd & bgd
    print("{:d} quasar pairs where both has a spec)".format(np.sum(gd_pairs)))

    # Generate TPE Table
    print("Generating TPE table from those with a good b/g spec".format(np.sum(gd_pairs)))
    tpe = tpe_table(qso_fg[bgd], qso_bg[bgd])

    # Eliminate close pair junk
    c_fg = SkyCoord(ra=tpe['FG_RA'], dec=tpe['FG_DEC'], unit='deg')
    c_bg = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    sep = c_fg.separation(c_bg)
    close = sep < 5*u.arcsec
    junk = close & (tpe['FG_MYERS_ZEM_SOURCE'] == 'VCV')
    cut_tpe = tpe[~junk]
    print("{:d} pairs after cutting close junk".format(len(cut_tpe)))

    # Build meta spec table and cut on those with good b/g spectra
    spec_tbl = get_spec_meta(cut_tpe)
    gd_spec = spec_tbl['nok'] > 0
    cut_tpe = hstack([cut_tpe, spec_tbl])
    spec_tpe = cut_tpe[gd_spec]
    print("{:d} pairs after cutting on good b/g spectra".format(len(spec_tpe)))

    # LV cut
    cut_LV = spec_tpe['FG_LOGLV'] < 37.
    LV_tpe = spec_tpe[cut_LV]
    print("{:d} pairs after cutting LV > 37".format(len(LV_tpe)))

    # R cut
    b_coords = SkyCoord(ra=LV_tpe['BG_RA'], dec=LV_tpe['BG_DEC'], unit='deg')
    f_coords = SkyCoord(ra=LV_tpe['FG_RA'], dec=LV_tpe['FG_DEC'], unit='deg')
    kpc_amin = cosmo.kpc_comoving_per_arcmin(LV_tpe['FG_Z'])  # kpc per arcmin
    ang_seps = b_coords.separation(f_coords)
    rho_phys = ang_seps.to('arcmin') * kpc_amin / (1+LV_tpe['FG_Z']) # Physical

    LV_tpe['R_phys'] = rho_phys
    cut_rho = rho_phys < R*u.Mpc
    rho_tpe = LV_tpe[cut_rho]
    print("{:d} pairs after R < {:g} pMpc".format(len(rho_tpe),R))

    # VCV without f/g spectrum
    no_fg = np.all([rho_tpe['FG_IGM_ID'] < 0, rho_tpe['FG_QPQ_ID'] < 0],axis=0)
    final_tpe = rho_tpe[~no_fg]
    print("{:d} pairs after requiring f/g spectrum".format(len(final_tpe)))

    # Update VCV redshifts
    vcv = np.where(final_tpe['FG_MYERS_ZEM_SOURCE'] == 'VCV')[0]
    for idx in vcv:
        mt = igmsp.cat['IGM_ID'] == final_tpe['FG_IGM_ID'][idx]
        assert np.sum(mt) == 1
        final_tpe['FG_Z'][idx] = igmsp.cat['zem'][mt][0]

    # Write
    if outfil is not None:
        print("Writing {:s}".format(outfil))
        final_tpe.write(outfil, overwrite=True)
    # Return
    return final_tpe



def get_spec_meta(tpe, outfil=None):
    """ Given a TPE table, generate a table describing available spectra
    and the preferred choice.
    Parameters
    ----------
    tpe : Table
    outfil : str, optional

    Returns
    -------
    spec_tbl : Table
      Table describing the spectra; aligned with input TPE Table
        specm -- str, describing all available spectra
        nspec -- int, number of available spectra
        best_spec -- str, describes the best one given priority dict
        nok -- number good enough for TPE (i.e. in instr_priority dict)
        ibest, best_row -- int, uninteresting indices
    """
    # Load spectral sets
    igmsp = IgmSpec()
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)#, verbose=True)
    #
    b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    # Query the spectral catalogs
    #igm_cat_match, igm_cat, igm_ID = igmsp.qcat.query_coords(b_coords)
    #qpq_cat_match, qpq_cat, qpq_ID = qpq.qcat.query_coords(b_coords)
    # Generate lists of meta tables
    igm_meta_match, igm_meta_list, igm_meta_stack = igmsp.meta_from_coords(b_coords, first=False)
    qpq_meta_match, qpq_meta_list, qpq_meta_stack = qpq.meta_from_coords(b_coords, first=False)
    # Identify best instrument/grating combo
    instr_pri_dict = tstack.instr_priority()
    spec_dict = dict(specm='', best_spec='', nspec=0, ibest=-1, nok=0, best_row=-1)
    spec_meta = [spec_dict.copy() for i in range(len(tpe))]
    print('Looping on pairs')
    for qq,pair in enumerate(tpe):
        # igmspec
        if igm_meta_match[qq]:
            # Add
            add_to_specmeta('igmsp', igm_meta_list, igm_meta_stack, qq, spec_meta, pair['FG_Z'], instr_pri_dict)
        # QPQ
        if qpq_meta_match[qq]:
            # Meta + add
            add_to_specmeta('qpq', qpq_meta_list, qpq_meta_stack, qq, spec_meta, pair['FG_Z'], instr_pri_dict)
        #if (qq % 500) == 0:
        #    print("Done with {:d}".format(qq))
    # Convert to Table
    spec_tbl = Table()
    for key in spec_dict.keys():
        clm = [sdict[key] for sdict in spec_meta]
        spec_tbl[key] = clm
    # Add Group, Group_ID
    dbase, group, group_id, specfile = [], [], [], []
    for kk,row in enumerate(spec_tbl):
        if row['best_spec'][0:4] == 'igms':
            dbase.append('igmspec')
            iidx = np.where(igm_meta_list[kk])[0]
            group.append(igm_meta_stack['GROUP'][iidx][row['best_row']])
            group_id.append(igm_meta_stack['GROUP_ID'][iidx][row['best_row']])
            specfile.append(igm_meta_stack['SPEC_FILE'][iidx][row['best_row']])
        elif row['best_spec'][0:3] == 'qpq':
            dbase.append('qpq')
            iidx = np.where(qpq_meta_list[kk])[0]
            group.append(qpq_meta_stack['GROUP'][iidx][row['best_row']])
            group_id.append(qpq_meta_stack['GROUP_ID'][iidx][row['best_row']])
            specfile.append(qpq_meta_stack['SPEC_FILE'][iidx][row['best_row']])
        else:
            dbase.append('none')
            group.append('none')
            group_id.append(-1)
            specfile.append('N/A')
    spec_tbl['DBASE'] = dbase
    spec_tbl['GROUP'] = group
    spec_tbl['GROUP_ID'] = group_id
    spec_tbl['SPEC_FILE'] = specfile
    # Write?
    if outfil is not None:
        spec_tbl.write(outfil, overwrite=True)
        print("Writing spec table: {:s}".format(outfil))
    # Return
    return spec_tbl


def add_to_specmeta(dbase, meta_list, meta_stack, qq, spec_meta, zabs, instr_prior, buff=50.):
    idx = np.where(meta_list[qq])[0]
    for ss,row in enumerate(idx):
        wv_lya = (1+zabs) * 1215.67
        if np.any([(meta_stack['WV_MIN'][row] > wv_lya-buff),(meta_stack['WV_MAX'][row] < wv_lya+buff)]):
            continue
        #
        if spec_meta[qq]['nspec'] > 0:
            spec_meta[qq]['specm'] += ';'
        spec_meta[qq]['specm'] += ','.join([dbase,meta_stack['GROUP'][row], meta_stack['INSTR'][row], meta_stack['DISPERSER'][row]])
        spec_meta[qq]['nspec'] += 1
        # Priority
        try:
            aok = instr_prior[meta_stack['INSTR'][row]] in meta_stack['DISPERSER'][row]
        except KeyError:
            if meta_stack['INSTR'][row] != '2dF':
                print('Instr = {:s} not in Priority dict with disperser={:s}'.format(meta_stack['INSTR'][row], meta_stack['DISPERSER'][row]))
        else:
            if aok:  # Better choice?
                spec_meta[qq]['nok'] += 1
                pri = instr_prior.keys().index(meta_stack['INSTR'][row])
                if pri > spec_meta[qq]['ibest']:
                    spec_meta[qq]['best_row'] = ss
                    spec_meta[qq]['ibest'] = pri
                    spec_meta[qq]['best_spec'] = ','.join([dbase,meta_stack['GROUP'][row], meta_stack['INSTR'][row], meta_stack['DISPERSER'][row]])


def tpe_chk_spec(tpe_file):
    """ Check spectrum exists and then look for continuum
    NO LONGER USED

    Parameters
    ----------
    tpe_file

    Returns
    -------

    """
    pdb.set_trace()
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
    #flg_fig += 2**2  # Generate pair tables (15 cMpc)
    flg_fig += 2**3  # Build TPE table for >31.5
    #flg_fig += 2**4  # Build TPE table for 31.2-31.5

    if (flg_fig % 2**1) >= 2**0:
        # Load for speed
        svfile = os.getenv('DROPBOX_DIR')+'QSOPairs/TPE_DR12/TPE_DR12_Mon-May-16-18:00:17-2016_concat.sav'
        print("Loading save file {:s}".format(svfile))
        print("Be patient....")
        tpe_sav = readsav(svfile)
        # Generate
        _ = make_tpe(31.2, outfil='TPE_DR12_31.2_spec.fits', tpe_sav=tpe_sav)
        _ = make_tpe(31.0, outfil='TPE_DR12_31.0_spec.fits', tpe_sav=tpe_sav)

    if (flg_fig % 2**2) >= 2**1:
        tpe_chk_spec('TPE_DR12_31.2_spec.fits')

    if flg_fig & (2**2):
        #run_wide_query('QPQ_v2000_R15_novette.fits', R_MAX=0.1*u.Mpc) # TEST
        run_wide_query('QPQ_v2000_R15_novette.fits', R_MAX=15.*u.Mpc) # TEST

    if flg_fig & (2**3):
        make_tpe(31.5, 4., qpqquery_file='QPQ_v2000_R15_novette.fits', outfil='TPE_31.5_4pMpc.fits')

    if flg_fig & (2**4):
        make_tpe([31.2,31.5], 4., qpqquery_file='QPQ_v2000_R15_novette.fits', outfil='TPE_31.2_31.5_4pMpc.fits')
