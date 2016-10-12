#Module for Tables for QPQ9
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

from astropy.io import ascii, fits
from astropy import units as u
from astropy.table import QTable

from xastropy.plotting import utils as xputils
from xastropy.xutils import xdebug as xdb
from xastropy.atomic import ionization as xai

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import qpq9_analy as qpq9a
import qpq_spec as qpqs

# ##################### #####################
# ##################### #####################
# Summary table of all transitions measured
def mktab_summ(outfil='tab_summ.tex', vcut=None, all_tups=None,
    passback=False, use_oiii=True):
    '''Summary table
    Parameters:
    -----------
    use_oiii: bool, optional
      Require [OIII]?  [Default=True]
    '''

    # Todo
    #   Include NHI on the label
    # Imports
    reload(qpq9a)
    reload(qpqs)

    # Read
    qpq9 = QTable.read('../Analysis/qpq9_final.fits')
    qpq9_adict = qpq9a.qpq9_init()

    #lines = [1548.195]  * u.AA
    lines = [1215.6701, 1334.5323, 1548.195]  * u.AA
    # 
    summ_dicts = []
    for ii in range(len(lines)):
        summ_dicts.append(dict(Nqpq=0, Nhigh=0,avgz=0.,avgR=0.))
    measures = dict(Nqpq=r'$N$', Nhigh=r'$N_{\rm high}$',
        avgz=r'$<z>$',avgR='\\rphys')
    formats = dict(Nqpq='{:d}', Nhigh='{:d}', avgz='{:.2f}',
        avgR='{:.1f}')
    order = ['Nqpq', 'Nhigh', 'avgz', 'avgR']
    if all_tups is None:
        all_tups = []

    # Loop on lines
    for kk,wrest in enumerate(lines):
        summ_dict = {}
        # Mask        #
        msk_flg = 10 
        if np.abs(wrest.value-1215.6701) > 1e-3:
            msk_flg += 4
        if use_oiii:
            msk_flg += 1
        #
        msk = qpq9a.cut_sample(qpq9,wrest,msk_flg)
        iqpq9 = copy.deepcopy(qpq9[msk])

        # Load stack image
        if len(all_tups) < len(lines):
            stack_tup = qpq9a.load_stack_img(iqpq9, wrest, 
                vmnx=qpq9_adict['vmnx'])
            all_tups.append(stack_tup)
        else:
            stack_tup = all_tups[kk]

        # Unpack
        fin_velo, stck_img, stck_msk, all_dict = stack_tup

        for idict in all_dict:
            # 
            if idict is None:
                continue
            #
            summ_dicts[kk]['Nqpq'] += 1
            if idict['R'] > 4000.:
                summ_dicts[kk]['Nhigh'] += 1
            # <z>, Rphys
            summ_dicts[kk]['avgz'] += idict['qpq']['FG_ZIR']
            summ_dicts[kk]['avgR'] += idict['qpq']['R_PHYS']
        # Averages
        for key in ['avgz','avgR']:
            summ_dicts[kk][key] = summ_dicts[kk][key] / summ_dicts[kk]['Nqpq']

    if passback:
        return all_tups

    # Open
    tbfil = open(outfil, 'w')

    # Header
    tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{deluxetable}{lccc}\n')
    #tbfil.write('\\rotate\n')
    tbfil.write('\\tablewidth{0pc}\n')
    tbfil.write('\\tablecaption{QPQ9 SUMMARY\\label{tab:summary}}\n')
    tbfil.write('\\tabletypesize{\\tiny}\n')
    tbfil.write('\\tablehead{\\colhead{Measure} & \\colhead{\lya} \n')
    tbfil.write('& \\colhead{\\ion{C}{2}~1334} \n')
    tbfil.write('& \\colhead{\\ion{C}{4}~1548} \\\\ \n')
    tbfil.write('} \n')

    tbfil.write('\\startdata \n')
    
    # Write
    for iorder in order:
        lin = '{:s} '.format(measures[iorder])
        for kk,wrest in enumerate(lines):
            lin += '&'+formats[iorder].format(summ_dicts[kk][iorder])
        tbfil.write(lin)
        tbfil.write('\\\\ \n')
            

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\enddata \n')
    '''
    tbfil.write('\\tablecomments{Columns are as follows: \n')
    tbfil.write('(1) Quasar name; \n')
    tbfil.write('(2,3) RA/DEC; \n')
    tbfil.write('(4) Absorption redshift of LLS; \n')
    tbfil.write('(5) HI column Density; \n')
    tbfil.write('(6) Rest wavelength of transition; \n')
    tbfil.write('(7) Velocity limits (min/max) for integration relative to $z_{\\rm abs}$; \n')
    tbfil.write('(8) Flag on individual measurement: [0,1=standard measurement; 2,3=Lower limit; 4,5=Upper limit]; \n')
    tbfil.write('(9) $\\log_{10}$ column density; \n')
    tbfil.write('(10) Standard deviation on $\\log_{10} N$.  Limits are given a value of 99.99; \n')
    tbfil.write('(11) Ion [atomic number, ionization state]; \n')
    tbfil.write('(12) Flag for the ionic column density [1=standard measurement; 2=Lower limit; 3=Upper limit]; \n')
    tbfil.write('(13) $\\log_{10}$ column density for the ion; \n')
    tbfil.write('(14) Standard deviation on $\\log_{10} N_{\\rm ion}$.  Limits are given a value of 99.99; \n')
    tbfil.write('} \n')
  #printf, 1, '\tablenotetext{a}{Velocity interval for the AODM'
  #printf, 1, 'relative to $z='+string(zcen,format='(f8.6)')+'$.}'
    #  printf, 1, '\tablenotetext{b}{Rest equivalent width.}'
    if sub is True:
        tbfil.write('\\tablecomments{[The complete version of this table is in the \n')
        tbfil.write('electronic edition of the Journal.  \n') 
        tbfil.write('The printed edition contains only a sample.]} \n')
    # End
    '''
    tbfil.write('\\end{deluxetable} \n')
    tbfil.close()

    print('All done with {:s}'.format(outfil))

        

#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_tab = 0 
        flg_tab += 2**0  # Summary table
        flg_tab += 2**1  # Summary table for Ha/Hb
        #flg_fig += 2**6  # Nucleo
    else:
        flg_tab = sys.argv[1]


    # Summary table
    if (flg_tab % 2**1) >= 2**0:
        mktab_summ()

    if (flg_tab % 2**2) >= 2**1:
        mktab_summ(outfil='tab_summ_Hab.tex', use_oiii=False)
