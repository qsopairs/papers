""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import numpy as np
import pdb
import warnings
import sys

from collections import OrderedDict
import h5py

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

from linetools.spectra import utils as lspu
from linetools.spectra import io as lsio

from specdb.specdb import IgmSpec
from specdb import cat_utils

from xastropy.xutils import xdebug as xdb

this_file = __file__
qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'

sys.path.append(os.path.abspath("./py"))
import tpe_plots as tpep

def instr_priority():
    """ Generate a dict that gives the instrument priority for the TPE analysis
    NOTE: Priority is in reverse order

    Returns
    -------
    instr_prior : dict
      Priority dict;  key/item = instr/grating
        '' for grating means use any

    """
    instr_prior = OrderedDict()
    instr_prior['SDSS'] = ''
    instr_prior['BOSS'] = ''
    instr_prior['GMOS-N'] = 'B600'
    instr_prior['LRISb'] = '1200/3400'
    instr_prior['MagE'] = ''
    instr_prior['ESI'] = 'ECH'
    instr_prior['UVES'] = ''
    instr_prior['HIRES'] = ''
    # Return
    return instr_prior


def add_to_specmeta(dbase, meta, qq, spec_meta, zabs, instr_prior, buff=50.):
    for ss,row in enumerate(meta):
        wv_lya = (1+zabs) * 1215.67
        if np.any([(row['WV_MIN'] > wv_lya-buff),(row['WV_MAX'] < wv_lya+buff)]):
            continue
        #
        if spec_meta[qq]['nspec'] > 0:
            spec_meta[qq]['specm'] += ';'
        spec_meta[qq]['specm'] += ','.join([dbase,row['GROUP'], row['INSTR'], row['DISPERSER']])
        spec_meta[qq]['nspec'] += 1
        # Priority
        try:
            aok = instr_prior[row['INSTR']] in row['DISPERSER']
        except KeyError:
            print('Instr = {:s} not in Priority dict'.format(row['INSTR']))
        else:
            if aok:  # Better choice?
                spec_meta[qq]['nok'] += 1
                pri = instr_prior.keys().index(row['INSTR'])
                if pri > spec_meta[qq]['ibest']:
                    spec_meta[qq]['best_row'] = ss
                    spec_meta[qq]['ibest'] = pri
                    spec_meta[qq]['best_spec'] = ','.join([dbase,row['GROUP'], row['INSTR'], row['DISPERSER']])


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
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)
    #
    b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
    # Query the spectral catalogs
    #igm_cat_match, igm_cat, igm_ID = igmsp.qcat.query_coords(b_coords)
    #qpq_cat_match, qpq_cat, qpq_ID = qpq.qcat.query_coords(b_coords)
    # Generate lists of meta tables
    igm_meta_match, igm_meta_list = igmsp.meta_from_coords(b_coords, first=False)
    qpq_meta_match, qpq_meta_list = qpq.meta_from_coords(b_coords, first=False)
    # Identify best instrument/grating combo
    instr_pri_dict = instr_priority()
    spec_dict = dict(specm='', best_spec='', nspec=0, ibest=-1, nok=0, best_row=-1)
    spec_meta = [spec_dict.copy() for i in range(len(tpe))]
    for qq,pair in enumerate(tpe):
        # igmspec
        if igm_meta_match[qq]:
            # Add
            add_to_specmeta('igmsp', igm_meta_list[qq], qq, spec_meta, pair['FG_Z'], instr_pri_dict)
        # QPQ
        if qpq_meta_match[qq]:
            # Meta + add
            add_to_specmeta('qpq', qpq_meta_list[qq], qq, spec_meta, pair['FG_Z'], instr_pri_dict)
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
            group.append(igm_meta_list[kk]['GROUP'][row['best_row']])
            group_id.append(igm_meta_list[kk]['GROUP_ID'][row['best_row']])
            specfile.append(igm_meta_list[kk]['SPEC_FILE'][row['best_row']])
        elif row['best_spec'][0:3] == 'qpq':
            dbase.append('qpq')
            group.append(qpq_meta_list[kk]['GROUP'][row['best_row']])
            group_id.append(qpq_meta_list[kk]['GROUP_ID'][row['best_row']])
            specfile.append(qpq_meta_list[kk]['SPEC_FILE'][row['best_row']])
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

def chk_continua(spec, fg_z):
    # Check for continua
    has_co = np.array([True]*spec.nspec)
    for ii in range(spec.nspec):
        # Select
        spec.select = ii
        # Match to lya
        lya = (1+fg_z[ii]) * 1215.67
        iwave = np.argmin(np.abs(spec.wavelength.value-lya))
        # Check for co
        if np.isclose(spec.co[iwave], 0.) or np.isclose(spec.co[iwave],1.):
            has_co[ii] = False
    # Return
    return has_co

def build_spectra(tpe, spec_tbl=None, outfil=None):
    """ Generate an XSpectrum1D object of TPE spectra
    Parameters
    ----------
    tpe : Table
    spec_tbl : str or Table, optional
    outfil : str, optional

    Returns
    -------
    spec : XSpectrum1D

    """
    from specdb.build import utils as spbu
    from linetools.spectra import utils as ltspu
    # Grab spectra table -- might read from disk eventually
    if spec_tbl is not None:
        if isinstance(spec_tbl, Table):
            pass
        elif isinstance(spec_tbl, basestring):
            spec_tbl = Table.read(spec_tbl)
    else:
        spec_tbl = get_spec_meta(tpe)
    assert len(tpe) == len(spec_tbl)
    # Load spectral sets
    igmsp = IgmSpec()
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)
    # Grab igmspec spectra
    iigms = spec_tbl['DBASE'] == 'igmspec'
    sub_meta = spec_tbl[['GROUP', 'GROUP_ID']][iigms]
    igm_spec = igmsp.spectra_from_meta(sub_meta)
    # Grab QPQ
    iqpq = spec_tbl['DBASE'] == 'qpq'
    sub_meta = spec_tbl[['GROUP', 'GROUP_ID']][iqpq]
    qpq_spec = qpq.spectra_from_meta(sub_meta)
    # Cut TPE
    gdtpe = spec_tbl['GROUP_ID'] >= 0
    cut_tpe = tpe[gdtpe]
    cut_stbl = spec_tbl[gdtpe]
    if np.sum(~gdtpe) > 0:
        print("These pairs had no good b/g spectrum")
        print(tpe[['BG_RA','BG_DEC']][~gdtpe])
    # Collate
    coll_spec = ltspu.collate([igm_spec, qpq_spec])
    # Reorder to match cut_tpe
    idxi = np.where(iigms)[0]
    idxq = np.where(iqpq)[0]
    alli = np.concatenate([idxi,idxq])
    isrt = np.argsort(alli)
    fin_spec = coll_spec[isrt]
    # Check continua
    has_co = chk_continua(fin_spec, cut_tpe['FG_Z'])
    cut_stbl['HAS_CO'] = has_co
    if np.sum(~has_co) > 0:
        print("These spectra need a continuum")
        print(cut_stbl[['SPEC_FILE']][~has_co])
    # Write
    if outfil is not None:
        hdf = h5py.File(outfil, 'w')
        fin_spec.write_to_hdf5('dumb', hdf5=hdf)
        # Add Tables
        spbu.clean_table_for_hdf(cut_tpe)
        hdf['TPE'] = cut_tpe
        spbu.clean_table_for_hdf(cut_stbl)
        hdf['SPEC_TBL'] = cut_stbl
        # Close
        hdf.close()
        print("Wrote: {:s}".format(outfil))
    return cut_tpe, cut_stbl, fin_spec

def load_spec(spec_file):
    """
    Parameters
    ----------
    spec_file : str

    Returns
    -------
    xspec : XSpectrum1D
    tpe : Table
    spec_tbl : Table

    """
    # Load
    print("Loading spectra from: {:s}".format(spec_file))
    hdf = h5py.File(spec_file,'r')
    xspec = lsio.parse_hdf5(hdf, close=False)
    # Tables
    tpe = Table(hdf['TPE'].value)
    spec_tbl = Table(hdf['SPEC_TBL'].value)
    # Close
    hdf.close()
    # Return
    return xspec, tpe, spec_tbl


def stack_spec(spec_file, dv=100*u.km/u.s, cut_on_rho=4.):
    # Load
    xspec, tpe, spec_tbl = load_spec(spec_file)
    # Cut on separation (should do this much earlier in the process)
    if cut_on_rho is not None:
        warnings.warn("Cutting on rho in stack.  Should do this earlier")
        b_coords = SkyCoord(ra=tpe['BG_RA'], dec=tpe['BG_DEC'], unit='deg')
        f_coords = SkyCoord(ra=tpe['FG_RA'], dec=tpe['FG_DEC'], unit='deg')
        kpc_amin = cosmo.kpc_comoving_per_arcmin(tpe['FG_Z'])  # kpc per arcmin
        ang_seps = b_coords.separation(f_coords)
        rho = ang_seps.to('arcmin') * kpc_amin / (1+tpe['FG_Z'])
        cut_rho = rho.to('Mpc').value < cut_on_rho
        print("We have {:d} spectra after the cut.".format(np.sum(cut_rho)))
        xspec = xspec[cut_rho]
        tpe = tpe[cut_rho]
        spec_tbl = spec_tbl[cut_rho]
    # Remove those without continua
    has_co = spec_tbl['HAS_CO'].data
    co_spec = xspec[has_co]
    co_spec.normed = True  # Apply continuum
    #  May also wish to isolate in wavelength to avoid rejected pixels
    for ii in range(co_spec.nspec):
        co_spec.select = ii
        co = co_spec.co.value
        sig = co_spec.sig.value
        bad_pix = np.any([(co == 0.),(co == 1.),(sig <= 0.)], axis=0)
        co_spec.add_to_mask(bad_pix, compressed=True)

    # Rebin to rest
    zarr = tpe['FG_Z'][has_co]
    rebin_spec = lspu.rebin_to_rest(co_spec, zarr, dv)

    # Stack
    stack = lspu.smash_spectra(rebin_spec)

    # Plot
    tpep.plot_stack(stack, 'all_stack.pdf')
    tpep.plot_spec_img(rebin_spec, 'spec_img.pdf')
    pdb.set_trace()
    # Return
    return


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
    co_spec = spec[has_co]
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

    # Check 2D
    check_td = True
    if check_td:
        fx = rebin_spec.data['flux']
        sig = rebin_spec.data['sig']
        gds = sig > 0.
        fx[~gds] = 0.
        xdb.set_trace() # xdb.ximshow(fx)

    # Stack
    stack = lspu.smash_spectra(rebin_spec)
    # Plot
    plot_stack(stack, 'BOSS_stack.pdf')
    print('Wrote')

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
    rows = meta['GROUP_ID'][idx]
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
    co_spec = spec[has_co]
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
    plot_stack(stack, 'LRIS_stack.pdf')

    return stack


# Command line execution (Run above py!!)
if __name__ == '__main__':

    flg_stack = 0
    #flg_stack += 2**0  # BOSS test
    #flg_stack += 2**1  # LRIS
    #flg_stack += 2**2  # Generate spec meta from tpe Table (test)
    #flg_stack += 2**3  # Generate spectra
    flg_stack += 2**4  # Stack

    if (flg_stack % 2**1) >= 2**0:
        tpe_stack_boss()

    if (flg_stack % 2**2) >= 2**1:
        tpe_stack_lris()

    if flg_stack & (2**2):
        tpe = Table.read('TPE_DR12_31.2_spec.fits')
        _ = get_spec_meta(tpe, outfil='tmp_spec_tbl.fits')

    if flg_stack & (2**3):
        tpe = Table.read('TPE_DR12_31.2_spec.fits')
        _, _, _ = build_spectra(tpe, spec_tbl='tmp_spec_tbl.fits',
                                outfil='TPE_DR12_31.2_spec.hdf5')
    if flg_stack & (2**4):
        stack_spec('TPE_DR12_31.2_spec.hdf5')
