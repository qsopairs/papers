# Module for QPQ9 structure

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, copy, os, sys

import sdsspy as spy

from astropy import units as u
from astropy.units import Unit
from astropy.io import ascii, fits
from astropy.table import QTable, Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.relativity import velocities as arv
from astropy import constants as const

from xastropy.xutils import fits as xxf
from xastropy.xutils import xdebug as xdb

sys.path.append(os.path.abspath(os.getenv('QPQ9')+"Analysis/Redshifts/py"))
import zsys_figs as qpqz

sys.path.append(os.path.abspath(os.getenv('QPQ9')+"Analysis/py"))
import qpq_sample as qpqsmp

##
def mk_qpq9(outfil=None):

    # Imports
    reload(qpqsmp)

    # Init
    if outfil is None:
        outfil = 'qpq9_final.fits'
        
    # Read
    bgfil = 'QPQ-CGM-IR_00.000-00.300_Sat-Apr-25-21:33:45-2015_qsobg.fits.gz'
    fgfil = 'QPQ-CGM-IR_00.000-00.300_Sat-Apr-25-21:33:45-2015_qsofg.fits.gz'
    ''' Getting too few Lya files in this one
    Not sure why
    bgfil = 'QPQ-CGM-IR_00.000-00.300_Tue-May-26-18:12:57-2015_qsobg.fits'
    fgfil = 'QPQ-CGM-IR_00.000-00.300_Tue-May-26-18:12:57-2015_qsofg.fits'
    '''
    qso_bg = xxf.bintab_to_table(os.getenv('DROPBOX_DIR')+'/QSOPairs/QSO-CGM-IR/'+bgfil)
    qso_fg = xxf.bintab_to_table(os.getenv('DROPBOX_DIR')+'/QSOPairs/QSO-CGM-IR/'+fgfil)

    # Collate
    qpq9 = copy.deepcopy(qso_bg)
    # Add/replce key columns from qso_fg
    oldc = ['Z_IR','SIGMA_ZIR', 'RA', 'DEC']
    newc = ['FG_ZIR', 'FG_SIG_ZIR', 'FG_RA', 'FG_DEC']
    for jj,ioldc in enumerate(oldc):
        tmpc = Column( qso_fg[ioldc], name=newc[jj])
        #print newc[jj]
        qpq9.add_column( tmpc )
    qpq9['FG_SIG_ZIR'].unit = u.km/u.s
    # 
    qpq9['G_UV'] = qso_fg['G_UV']
    qpq9.add_column( Column( qso_bg['Z_TOT'], name='BG_Z') )

    # Rphys
    # Cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.26)

    # Coordinates
    c_fg = SkyCoord(ra=qpq9['FG_RA']*u.degree, dec=qpq9['FG_DEC']*u.degree)
    c_bg = SkyCoord(ra=qpq9['RA']*u.degree, dec=qpq9['DEC']*u.degree)

    # Separation (Rphys)
    kpc_amin = cosmo.kpc_proper_per_arcmin( qpq9['FG_ZIR'] ) # kpc per arcmin
    ang_sep = c_fg.separation(c_bg).to('arcmin')
    Rphys = kpc_amin * ang_sep

    # Add
    qpq9.add_column(Column( Rphys, name='R_PHYS'))

    # Cuts
    # Lya beta
    zLya_beta = (qpq9['BG_Z']+1)*1025.7223 / 1215.6701 - 1
    dv_beta = arv.v_from_z(qpq9['FG_ZIR'], zLya_beta)
    gd_beta = dv_beta > (-500.*Unit('km/s'))
    #

    qpq9 = qpq9[gd_beta]
    c_fg = c_fg[gd_beta]
    c_bg = c_bg[gd_beta]
    print('We have {:d} pairs in QPQ9'.format(len(qpq9)))

    # Update zIR
    all_zlines = []
    for ss,qpq in enumerate(qpq9):
        # zem_fil
        try:
            zem_fil = glob.glob(os.getenv('QPQ9')+'/Analysis/Redshifts/zem/SDSSJ'+
                '{:s}'.format(c_fg[ss].ra.to_string(unit=u.hour,pad=True,sep='',precision=2))[:-1]
                +'*')[0]
        except IndexError:
            print('No zem file for {:s}'.format(qpq['NAME']))
            print('Skipping..')
            all_zlines.append('NULL')
            qpq9[ss]['FG_SIG_ZIR'] = int(9999.) * u.km/u.s
            continue
        # Line
        out_struct = QTable.read(zem_fil)
        zstruct = QTable.read(zem_fil,2)
        zsys_flag = out_struct['ZSYS_FLAG'][0]
        zline = qpqz.zsys_flag_str(zsys_flag)
        all_zlines.append(zline)
        # Redshift
        if zline != 'NULL':
            qpq9[ss]['FG_ZIR'] = out_struct['ZSYS_ZSYS'][0]
            err = out_struct['ZSYS_ERR'][0]
            qpq9[ss]['FG_SIG_ZIR'] = int(err) * u.km/u.s
        else:
            qpq9[ss]['FG_SIG_ZIR'] = int(9999.) * u.km/u.s
    qpq9.add_column(Column( all_zlines, name='ZFG_LINE'))

    # M_i
    M_i = np.zeros(len(qpq9))
    for jj,qpq in enumerate(qpq9):
        # Flux
        fi = spy.dered_fluxes(qpq['EXTINCTION'][3], qpq['PSFFLUX'][3])
        # Magnitude
        imag = qpqsmp.sdss_flux2mag(fi, 'i')
        # Absolute (standard cosmology assumed)
        M_i[jj] = qpqsmp.m_iz2_qso(imag, qpq['FG_ZIR']).value
    qpq9.add_column(Column( M_i*u.mag, name='M_i'))

    # Write
    print('Writing QPQ9 structure to {:s}'.format(outfil))
    xxf.table_to_fits(qpq9, outfil)

    # Return
    return qpq9


    
# ##################################################
# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg= 0
    flg+= 2**0  # Generate the Structure

    if flg% 2**1 >= 2**0:
        qpq9 = mk_qpq9()
        print(qpq9)
        
