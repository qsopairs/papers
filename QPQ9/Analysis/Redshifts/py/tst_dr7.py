"""
#;+ 
#; NAME:
#; tst_dr7
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for running our HW10 code on DR7
#;   Jul-2015 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

# imports
import numpy as np
import glob, os, sys, copy
import multiprocessing
import cPickle as pickle

from scipy import signal as sci_signal

from astropy.nddata import StdDevUncertainty
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from astropy.table import QTable, Column

from xastropy.obs import radec as xrad
from xastropy.xutils import xdebug as xdb
from xastropy.xutils import fits as xxf
from xastropy.sdss import quasars as sdssq
from xastropy.sdss import qso as sdssqso

sys.path.append(os.path.abspath("./py"))
import qpq_z as qpqz
import hewittwildz as hw10z

# Hewitt&Wild Prescription

# Generate globals for mapping

# Read HW10 table
fil = os.getenv('QPQ9')+'Analysis/Redshifts/hw10_tab4.fits.gz'
hw10_tab4 = QTable.read(fil)

# JXP DR7
sdss_dr7 = sdssq.SdssQuasars(verbose=False)

def mgii_in_hw10(parallel=True, nproc=8):
    '''Analyze quasars good for MgII in HW10 and DR7
    
    Parameters:
    ------------
    parallel: bool, optional
      Run code in parallel (with pool.map)
    nproc: int, optional
      Number of processors for parallel

    Returns:
    --------
    '''

    # Parse
    zmnx = [1.5,2.2]
    gdhw = np.where( (hw10_tab4['z']>zmnx[0]) & (hw10_tab4['z']<zmnx[1]))[0]
    print('We have {:d} DR7 quasars to try.'.format(len(gdhw)))

    testing = True
    if testing:
        idx = np.arange(5000)
        gdhw = gdhw[idx]

    # Setup qso files for execution including spectra (for multi-process)
    if parallel:
        pool = multiprocessing.Pool(nproc) # initialize thread pool N threads
        hw_qsos = pool.map(load_qso,gdhw)
    else:
        hw_qsos = map(load_qso,gdhw)
    print(hw_qsos[0:20])

    # Avoid non-overlapping
    gdtst = [idx for idx in range(len(gdhw)) if hw_qsos[idx] is not None]
    tst_qsos = [hw_qsos[idx] for idx in gdtst]

    # Run HW10
    #parallel=False
    if parallel:
        pool = multiprocessing.Pool(nproc) # initialize thread pool N threads
        all_HWz = pool.map(run_hw10,tst_qsos)
    else:
        xdb.set_trace() # Next line needs updating
        all_HWz = map(run_hw10,gdtst)

    # Generate final table
    dr7_tst = hw10_tab4[gdhw[gdtst]]
    dr7_tst.rename_column('z','HW_z')
    DR7_z = Column(np.array([qso.z for qso in tst_qsos]),name='DR7_z')
    JXP_z = Column(np.array([hwz.z_dict['zfin'] for hwz in all_HWz]),name='JXP_z')
    cc_max = Column(np.array([hwz.z_dict['cc_max'] for hwz in all_HWz]),name='cc_max')
    dr7_tst.add_columns([DR7_z,JXP_z,cc_max])
    xxf.table_to_fits(dr7_tst,'HW_DR7_test.fits')

    print('All done!')


# For mapping
def load_qso(ii):
    '''Load QSO object and fill spectrum
    '''
    pf = (hw10_tab4['Plate'][ii],hw10_tab4['Fiber'][ii])
    qso = sdss_dr7[pf[0],pf[1]]
    if qso is not None:
        qso.verbose=False
        qso.load_spec()
    # Return
    return qso

# For mapping
def run_hw10(qso):
    ''' Run HW10 algorithm
    '''
    qso.spec.z = qso.z
    hwz = hw10z.HewittWildz(qso.spec)
    hwz.run()
    # Flush memory
    # Return
    return hwz
    
################## PLOTS ########################
def plt_dr7_test(outfil='plt_dr7_test.pdf'):
    '''Plot offests between my code and HW10 results
    '''
    # Imports
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    # Read output
    dr7_tst = QTable.read('HW_DR7_test.fits')

    # Begin plot
    if outfil is not None:
        pp = PdfPages(outfil)

    fig = plt.figure(figsize=(6, 6))
    fig.clf()
    gs = gridspec.GridSpec(2, 2)

    for qq in range(3):
        ax = plt.subplot(gs[qq%2,qq//2])
        ax.set_ylabel(r'$z_{\rm JXP}-z_{\rm HW10}$')
        yval = dr7_tst['JXP_z']-dr7_tst['HW_z']
        ax.set_ylim(-5e-3, 5e-3)

        if qq == 0:
            ax.set_xlabel(r'$z_{\rm HW10}$')
            xval = dr7_tst['HW_z']
        elif qq == 1:
            ax.set_xlabel(r'$cc_{\rm max}$')
            xval = dr7_tst['cc_max']
        elif qq == 2:
            ax.set_xlabel(r'$(z_{\rm JXP} - z_{\rm DR7})/(1+z)$')
            xval = (dr7_tst['JXP_z'] - dr7_tst['DR7_z'])/(1+dr7_tst['JXP_z'])
            ax.set_xlim(-5e-3, 5e-3)
        # Plot
        ax.scatter(xval,yval,s=5.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.0)
    pp.savefig(bbox_inches='tight')
    pp.close()
    print('Generated {:s}'.format(outfil))



# ################
if __name__ == "__main__":

    import pdb
    flg_tst = 0 
    pdb.set_trace()
    #flg_tst += 2**0  # Test simple case
    flg_tst += 2**1  # Test with spectrum
    
    # 
    if (flg_tst % 2**1) >= 2**0:
        mgii_in_hw10()

    if (flg_tst % 2**2) >= 2**1:
        plt_dr7_test()
