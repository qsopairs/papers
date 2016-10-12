"""
#;+ 
#; NAME:
#; qpq_spec
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for QPQ redshifts
#;   May-2015 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

# imports
import numpy as np
import glob, os, sys, copy
import multiprocessing

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.stats import sigma_clip
from astropy.nddata import StdDevUncertainty
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.table import QTable, Table, Column 

from linetools.spectra import io as lsio
from linetools.spectra.xspectrum1d import XSpectrum1D

from xastropy.xutils import fits as xxf
from xastropy.stats import basic as xsb

from xastropy.xutils import xdebug as xdb

from enigma.qso import hewittwildz as hw10z

def parse_o2_zstruct(xx=None):
    ''' Parse the output file from JFH
    Parameters:
    -------------
    xx: str, optional
      Line to focus on (MgII, Ha, Hb)

    Returns:
    --------------
    qsos 
    zline
    '''
    # Read O2 file
    qsozout = os.getenv('DROPBOX_DIR') + '/O2_zsys/qsos_o2_zstruct.fits'
    qsos = Table.read(qsozout)
    # Cut on good [OII]
    good_o2 = ((qsos['ZSYS_ZSYS'] > 0.0)&# AND $   ;; systemic redshift not zero
        (qsos['ZSYS_OII_QUAL'] == 1)&# AND $ ;; received
        (qsos['M_I_Z2'] <= -23.0)&# AND $   ;; bright enough to be a QSO and not Seyfert
        (qsos['OII_SN'] >= 30.0)& # AND $     ;; OII S/N ratio cut (S/N in a 500 km/s window)
        (qsos['ZSYS_OII_PEAK2CONT'] >= 0.5)) # ;; OII peak/continuum > 0.5
    #
    # Good MgII
    if xx == 'MgII':
        good = good_o2 & (qsos['MGII_MODE'] > 0.0)
        lambda_line = 2798.75
        z_line = qsos['MGII_MODE']/lambda_line - 1.0
    elif xx == 'Ha':
        good = good_o2 & (qsos['HA_MODE'] > 0.0)
        lambda_line = 6564.61
        z_line = qsos['HA_MODE']/lambda_line - 1.0
    elif xx == 'Hb':
        good = good_o2 & (qsos['HB_MODE'] > 0.0)
        lambda_line = 4862.68
        z_line = qsos['HB_MODE']/lambda_line - 1.0
    elif xx is None:
        good = good_o2 
        z_line = None
    else:
        raise ValueError('Not prepared for this')

    # Keep good ones
    ikeep = np.where(good)
    gqsos = qsos[ikeep]
    if z_line is not None:
        z_line = z_line[ikeep]

    # [OII] redshift
    lambda_o2 = 3728.30  # ;; Boroson or van den Berk??
    z_O2 = gqsos['ZSYS_OII_FPEAK']/lambda_o2 - 1.0

    # Return
    return gqsos, z_O2, z_line

# XX rms vs [OII]
def xx_o2_joe(xx, nsclip=3., outfil=None):
    '''Stats on XX vs [OII] from Joe's analysis alone
    Parameters:
    -----------
    nsclip: float, optional
      Number of sigma for clipping [3.]
    '''
    # TODO
    #  Check the clipping

    # Init
    if outfil is None:
        outfil=xx+'_vs_o2_jfh.pdf'

    gqsos, z_O2, z_line = parse_o2_zstruct(xx)

    # Velocity differences
    all_vdiff = const.c.to('km/s').value*(z_O2-z_line)/(1+z_O2)
    print('RMS={:g} before'.format(np.std(all_vdiff)))

    # Clip
    clip_vdiff = sigma_clip(all_vdiff,sigma=nsclip)
    nclip = np.sum(clip_vdiff.mask==True)

    # Stats
    vmean = np.mean(clip_vdiff)
    vmedian = np.mean(clip_vdiff)
    rms = np.std(clip_vdiff)
    binsz = rms/5.
    xper = xsb.perc(all_vdiff,per=0.95)

    # Plot
    fig = plt.figure(figsize=(7, 5.0))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    # Binning
    minv, maxv = np.amin(clip_vdiff), np.amax(clip_vdiff)
    i0 = int( minv / binsz) - 1
    i1 = int( maxv / binsz) + 1
    rng = tuple( binsz*np.array([i0,i1]) )
    nbin = i1-i0

    # Histogram
    hist, edges = np.histogram(clip_vdiff, range=rng, bins=nbin)
    ax.bar(edges[:-1], hist, width=binsz)
    ax.set_xlim(-3000., 3000)

    # Label
    ax.set_xlabel(r'$\delta v$ (km/s)')
    ax.set_ylabel('Number')
    xlbl = 0.05
    ax.text(xlbl, 0.9, r'$z_{\rm '+xx+r'}-z_{\rm O2}$ (JFH)',
        transform=ax.transAxes, ha='left')
    ax.text(xlbl, 0.8, r'$<\delta v>=$'+'{:.1f} km/s'.format(vmean),
        transform=ax.transAxes, ha='left')
    ax.text(xlbl, 0.7, r'$\sigma(\delta v)=$'+'{:.1f} km/s'.format(rms),
        transform=ax.transAxes, ha='left')
    ax.text(xlbl, 0.6, 'nclip={:d} for {:.1f}sigma'.format(nclip,nsclip),
        transform=ax.transAxes, ha='left')
    ax.text(xlbl, 0.5, '95% limit=({:.1f},{:.1f})'.format(xper[0],xper[1]),
        transform=ax.transAxes, ha='left')

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    print('Wrote: {:s}'.format(outfil))


# Globals for parallization
g_mgii_qsos, g_z_O2, g_mgii_z_line = parse_o2_zstruct('MgII')
drop_dir  = os.getenv('DROPBOX_DIR')

# HW10 on MgII (with parallelization)
def zmgii_from_hw10(parallel=True,nproc=8, debug=False):
    # Setup qso files for execution including spectra (for multi-process)

    print('Loading..')
    gdq = np.arange(len(g_mgii_qsos))
    if True:
        gdq = gdq[0:1000]
    if parallel:
        pool = multiprocessing.Pool(nproc) # initialize thread pool N threads
        mgii_o2_spec = pool.map(load_mgii_o2_qso,gdq)
    else:
        xdb.set_trace()
        #hw_qsos = map(load_qso,gdhw)

    print('Done loading..')

    # Run HW10
    #parallel=False
    if parallel:
        pool = multiprocessing.Pool(nproc) # initialize thread pool N threads
        all_HWz = pool.map(run_mgii_hw10,mgii_o2_spec)
    else:
        xdb.set_trace() # Next line needs updating
        #all_HWz = map(run_hw10,gdtst)

    # Generate final table
    mgii_o2_tab = g_mgii_qsos[gdq]
    z_O2 = Column(g_z_O2[gdq],name='O2_z')
    JFH_z = Column(g_mgii_z_line[gdq],name='JFH_z')
    HW_z = Column(np.array([hwz.z_dict['zfin'] for hwz in all_HWz]),name='HW_z')
    cc_max = Column(np.array([hwz.z_dict['cc_max'] for hwz in all_HWz]),name='cc_max')
    mgii_o2_tab.add_columns([z_O2,JFH_z,HW_z,cc_max])
    # Write
    xxf.table_to_fits(mgii_o2_tab,'MgII_O2_HW.fits')
    print('All done..')
    #xdb.set_trace()

# For mapping
def load_mgii_o2_qso(ii):
    '''Load QSO object and fill spectrum
    '''
    # Generate XSpectrum1D
    spec_file = drop_dir+'/O2_zsys/sdss_boss_o2/spectra/'+g_mgii_qsos[ii]['SPEC_FILE']
    spec = XSpectrum1D.from_file(spec_file)
    spec.z = g_mgii_qsos[ii]['Z']
    # Return
    return spec

# For mapping
def run_mgii_hw10(spec):
    ''' Run HW10 algorithm
    '''
    hwz = hw10z.HewittWildz(spec)
    hwz.run(msk_for_mgii=True)
    # Return
    return hwz

# XX rms vs [OII]
def compare_mgii(nsclip=3., outfil=None):
    '''Stats on MgII vs [OII] and HW vs JFH 
    Parameters:
    -----------
    nsclip: float, optional
      Number of sigma for clipping [3.]
    '''
    # TODO
    #  Check the clipping

    # Init
    if outfil is None:
        outfil='compare_mgii.pdf'

    mgii_o2_tab = Table.read('MgII_O2_HW.fits')


    # Plot
    fig = plt.figure(figsize=(7, 5.0))
    gs = gridspec.GridSpec(1,2)

    for ss in range(2):
        ax = plt.subplot(gs[ss])

        if ss == 0:
            # Velocity differences
            all_vdiff = const.c.to('km/s').value*(mgii_o2_tab['O2_z']-
                mgii_o2_tab['HW_z'])/(1+mgii_o2_tab['O2_z'])
            lbl = r'$z_{\rm MgII}-z_{\rm O2}$ (HW)'
        elif ss==1:
            # Velocity differences
            all_vdiff = const.c.to('km/s').value*(mgii_o2_tab['JFH_z']-
                mgii_o2_tab['HW_z'])/(1+mgii_o2_tab['O2_z'])
            lbl = r'$z_{\rm MgII}-z_{\rm MgII}$ (HW/JFH)'

        # Clip
        clip_vdiff = sigma_clip(all_vdiff,sigma=nsclip)
        nclip = np.sum(clip_vdiff.mask==True)

        # Stats
        vmean = np.mean(clip_vdiff)
        vmedian = np.mean(clip_vdiff)
        rms = np.std(clip_vdiff)
        binsz = rms/5.
        xper = xsb.perc(all_vdiff,per=0.95)
        # Binning
        minv, maxv = np.amin(clip_vdiff), np.amax(clip_vdiff)
        i0 = int( minv / binsz) - 1
        i1 = int( maxv / binsz) + 1
        rng = tuple( binsz*np.array([i0,i1]) )
        nbin = i1-i0

        # Histogram
        hist, edges = np.histogram(clip_vdiff, range=rng, bins=nbin)
        ax.bar(edges[:-1], hist, width=binsz)
        ax.set_xlim(-3000., 3000)

        # Label
        ax.set_xlabel(r'$\delta v$ (km/s)')
        ax.set_ylabel('Number')
        xlbl = 0.05
        ax.text(xlbl, 0.9, lbl, transform=ax.transAxes, ha='left')
        ax.text(xlbl, 0.8, r'$<\delta v>=$'+'{:.1f} km/s'.format(vmean),
            transform=ax.transAxes, ha='left')
        ax.text(xlbl, 0.7, r'$\sigma(\delta v)=$'+'{:.1f} km/s'.format(rms),
            transform=ax.transAxes, ha='left')
        ax.text(xlbl, 0.6, 'nclip={:d} for {:.1f}sigma'.format(nclip,nsclip),
            transform=ax.transAxes, ha='left')
        ax.text(xlbl, 0.5, '95% limit=({:.1f},{:.1f})'.format(xper[0],xper[1]),
            transform=ax.transAxes, ha='left')

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.0)
    plt.savefig(outfil)
    print('Wrote: {:s}'.format(outfil))



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_fig = 0 
        #flg_fig += 2**0  # MgII (JFH)
        #flg_fig += 2**1  # Ha (JFH)
        #flg_fig += 2**2  # Hb (JFH)
        #flg_fig += 2**3  # Run HW10 on MgII
        flg_fig += 2**4  # Compare MgII results (HW, JFH, [OII])

    # MgII (JFH)
    if (flg_fig % 2**1) >= 2**0:
        xx_o2_joe('MgII', outfil='mgii_vs_o2_jfh.pdf')

    # Halpha (JFH)
    if (flg_fig % 2**2) >= 2**1:
        xx_o2_joe('Ha', outfil='Ha_vs_o2_jfh.pdf')

    # Hbeta (JFH)
    if (flg_fig % 2**3) >= 2**2:
        xx_o2_joe('Hb', outfil='Hb_vs_o2_jfh.pdf')

    # MgII analysis with HW10 template
    if (flg_fig % 2**4) >= 2**3:
        zmgii_from_hw10()

    # MgII comparison with HW10 template
    if (flg_fig % 2**5) >= 2**4:
        compare_mgii()


