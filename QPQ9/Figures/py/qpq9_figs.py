#Module for QPQ9 figures
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

from astropy.table import QTable, Table
from astropy.io import ascii
from astropy import units as u
from astropy import constants as const

from linetools.spectralline import AbsLine

from xastropy.igm.abs_sys import abssys_utils as abssys
from xastropy import spec as xspec
from xastropy.plotting import utils as xputils
from xastropy.xutils import xdebug as xdb
from xastropy.obs import radec as xor
from xastropy.atomic import ionization as xai

sys.path.append(os.path.abspath("../../../py"))
from enigma.qpq import utils as qpqutils
from enigma.qpq import spec as qpqs

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import qpq9_analy as qpq9a

####
#  Plot the Experiment
def experiment(outfil=None, qpq9=None, spec_files=None, spec_dicts=None):

    # Todo
    #   Have symbol size indicate near-IR for redshift

    # Imports
    reload(xai)

    if outfil is None:
        outfil = 'fig_experiment.pdf'

    # Read QPQ9
    if qpq9 is None:
        qpq9 = QTable.read('../Analysis/qpq9_final.fits')
    if spec_files is None:
        spec_files = qpqs.get_spec_files([(iqpq9['RA']*u.degree,iqpq9['DEC']*u.degree) for iqpq9 in qpq9]) 
    '''
    spec_files = []
    for iqpq9 in qpq9:
        i_spec_files = qpqs.get_spec_files((iqpq9['RA']*u.degree,iqpq9['DEC']*u.degree)) 
        spec_files.append(i_spec_files)
        # Missing?
        if len(i_spec_files) == 0:
            print('No spectra for b/g QSO {:s} with Lya File = {:s}'.format(iqpq9['NAME'], iqpq9['LYA_FILE']))
            ras, decs = xor.dtos1( (iqpq9['RA'], iqpq9['DEC']) )
            print('RA/DEC = {:s}, {:s}'.format(ras, decs))
            print('RA/DEC = {:g}, {:g}'.format(iqpq9['RA'], iqpq9['DEC']))
            xdb.set_trace()
        else:
            print('Got at least one spectrum for {:s}'.format(iqpq9['NAME']))
    '''
    # Generate the dicts
    if spec_dicts is None:
        spec_dicts = []
        for ispecfs in spec_files:
            # Generate the list
            ispec_dicts = [qpqs.load_spec(spec_file) for spec_file in ispecfs]
            # Append
            spec_dicts.append( ispec_dicts )

    # Setup
    lines = [1548.195, 1334.5323, 1215.6701] * u.AA
    zlim = (1.6, 3.9)
    Rlim = (0., 300.)*u.kpc
    guv_min = np.min(qpq9['G_UV'])
    guv_max = np.max(qpq9['G_UV'])
    xyc = range(int(guv_min), int(guv_max))
    psz = 35.
    
    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    fig = plt.figure(figsize=(5, 8))
    fig.clf()
    gs = gridspec.GridSpec(3, 1)

    #font_axes = FontProperties()

    # Looping
    for kk,line in enumerate(lines):

        # Get line info
        aline = AbsLine(line)
        

        # Axes
        ax = plt.subplot(gs[kk,0])
        #ax.xaxis.set_minor_locator(plt.MultipleLocator(100.))
        #ax.xaxis.set_major_locator(plt.MultipleLocator(200))
        ax.set_xlim(Rlim.value)
        ax.set_ylim(zlim)

        ## ####
        # Labels
        ax.set_ylabel(r'$z_{\rm f/g}$')
        if kk<2:
            ax.get_xaxis().set_ticks([])
        else:
            ax.set_xlabel(r'$R_\perp$ (kpc)')
        ax.text(10., 1.7, xai.ion_name(aline.atomic), size='large' )

        # Loop on QPQ9
        msk = qpq9['R_PHYS'] > qpq9['R_PHYS']
        for jj,iqpq9 in enumerate(qpq9):
            # Does a spectrum cover this line?
            for s_dict in spec_dicts[jj]:
                sflg=True
                try:
                    wrest_mnx = [s_dict['wvmnx'][ii]/(iqpq9['FG_ZIR']+1) for ii in range(2)] 
                except TypeError:
                    print('Skipping qso {:s}'.format(iqpq9['NAME']))
                    sflg=False
                # Coverage?
                if (line>wrest_mnx[0]) & (line<wrest_mnx[1]) & sflg:
                    # Line
                    if not msk[jj]:
                        ax.plot( [iqpq9['R_PHYS'].value]*2, zlim, '--', color='gray', alpha=0.5, lw=0.5)
                        msk[jj] = True
        # Scatter
        #xdb.set_trace()
        sc = ax.scatter( qpq9['R_PHYS'][msk].value, qpq9['FG_ZIR'][msk], s=psz,
                         c=np.log10(qpq9['G_UV'][msk]))#, edgecolor='none')

        # Font
        xputils.set_fontsize(ax,15.)

    # Color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.05, 0.07, 0.05, 0.9])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label(r'$\log_{10} \; g_{\rm UV}$')


    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('experiment: Wrote {:s}'.format(outfil))
    else: 
        plt.show()

####
#  Simple stack plot(s)
def simple_stack(outfil=None, all_stack=None, passback=False):

    # Imports
    reload(qpqs)
    reload(qpq9a)
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    # Load QPQ9
    qpq9_fil = os.getenv('QSO_DIR')+'/tex/QPQ9/Analysis/qpq9_final.fits'
    print('Reading {:s}'.format(qpq9_fil))
    qpq9 = QTable.read(qpq9_fil)
    qpq9_adict = qpq9a.qpq9_init()

    # Initialize
    if outfil is None:
        outfil = 'fig_simple_stack.pdf'
    avmnx = qpq9_adict['vmnx'] # Analysis velocity range
    pvmnx = (-2000., 2000.)*u.km/u.s # Plotting velocity range
    yrng = (0.,1.1)

    # Lines
    lines = [1215.6701, 1334.5323, 1548.195]*u.AA
    nlin = len(lines)

    if all_stack is None: 
        # Generate
        all_stack = {}
        for kk,wrest in enumerate(lines):
            # Mask
            if np.abs(wrest.value-1215.6701) < 1e-3:
                msk_flg = 3
            else: 
                msk_flg = 7 # stay outside Lya forest
            msk = qpq9a.cut_sample(qpq9,wrest,msk_flg)
            # Stack (average)
            vel, fx, spec_dict = qpq9a.stack_avg( (qpq9[msk], wrest), 
                vmnx=avmnx)
            # Save a dict
            tdict = {'velo': vel, 'fx': fx, 'spec_dict': spec_dict,
                'qpq': copy.deepcopy(qpq9[msk]) } 
            all_stack[int(wrest.value)] = tdict
            print('Finished stack for {:g}'.format(wrest))
    else:
        if len(all_stack) != nlin:
            raise ValueError('Uh oh')

    # Passback?
    if passback:
        return all_stack

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    fig = plt.figure(figsize=(5, 8))
    fig.clf()
    # Color map
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, 
        vmax=np.amax(all_stack[int(lines[0].value)]['qpq']['R_PHYS'].value))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #print scalarMap.get_clim()

    # Axes
    gs = gridspec.GridSpec(nlin, 1)

    # Looping
    for kk,wrest in enumerate(lines):

        # Grab the dict
        ldict = all_stack[int(wrest.value)]

        # Get line info
        aline = AbsLine(wrest)

        # Axes
        ax = plt.subplot(gs[kk,0])
        ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
        ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        ax.set_xlim(pvmnx.value)
        ax.set_ylim(yrng)

        ## ####
        # Labels
        ax.set_ylabel('Relative Flux')
        if kk+1<nlin:
            ax.get_xaxis().set_ticks([])
        else:
            ax.set_xlabel(r'$\delta v \rm km \, s^{-1}$')

        # Stack
        ax.plot(ldict['velo'], ldict['fx'], 'k', drawstyle='steps', 
            linewidth=2.)

        # Zero line
        ax.plot([0]*2, yrng, 'b--')
        ax.plot(pvmnx, [1]*2, ':', color='lightgreen')

        # Label
        ax.text(0.07, 0.10, xai.ion_name(aline.atomic)+' {:d}'.format(int(wrest.value)), 
            size='x-large', transform=ax.transAxes, ha='left', 
            bbox={'facecolor':'white', 'edgecolor':'white'})

        # All the data
        allv = []
        allf = []
        allc = []
        for jj,sdict in enumerate(ldict['spec_dict']):
            if sdict is None:
                continue
            # Convert to velocity then plot
            velo = sdict['spec'].relative_vel(
                wrest*(ldict['qpq']['FG_ZIR'][jj]+1))
            #xdb.set_trace()
            # Dots
            gdp = np.where((velo > pvmnx[0]) & (velo < pvmnx[1]))[0]
            ngdp = len(gdp)
            #xdb.set_trace()
            # Save
            allv.append(velo[gdp].value)
            allf.append(sdict['spec'].flux[gdp])
            allc.append([ldict['qpq']['R_PHYS'][jj].value]*ngdp)
            #sc = ax.scatter(velo[gdp], sdict['spec'].flux[gdp], 
            #    alpha=0.5, s=2., c=[ldict['qpq']['R_PHYS'][jj].value]*ngdp)
            # Lines
            colorVal = scalarMap.to_rgba(ldict['qpq']['R_PHYS'][jj].value)
            ax.plot(velo, sdict['spec'].flux, color=colorVal,
                drawstyle='steps', linewidth=0.3, alpha=0.7)

        #xdb.set_trace()
        #sc = ax.scatter(np.concatenate(allv), np.concatenate(allf), 
        #    alpha=0.5, s=2., c=np.concatenate(allc), edgecolor='none')
        # Font
        xputils.set_fontsize(ax,15.)

    # Color bar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([1.05, 0.07, 0.05, 0.9])
    #cb = fig.colorbar(sc, cax=cbar_ax)
    #cb.set_clim(20., 300)
    #cb.set_label(r'$R_\perp$')

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('simple_stack: Wrote {:s}'.format(outfil))
    else: 
        plt.show()
    plt.close()

####
#  Compare stacks
def stack_trials(wrest=None, outfil=None, stack_tup=None, passback=False):

    # Imports
    reload(qpqs)
    reload(qpq9a)

    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA
    aline = AbsLine(wrest)

    # Initialize
    if outfil is None:
        outfil = 'fig_stack_trials_{:s}.pdf'.format(xai.ion_name(aline.data))
    avmnx = (-3000., 3000.)*u.km/u.s # Analysis velocity range
    pvmnx = (-2500., 2500.)*u.km/u.s # Plotting velocity range
    yrng = (0.,1.1)

    # Load QPQ9
#    qpq9 = qpq9a.load_qpq(wrest)
    qpq9 = qpqutils.load_qpq(9,wrest=wrest)

    # Load stack image
    if stack_tup is None:
        stack_tup = qpq9a.load_stack_img(qpq9, wrest, vmnx=avmnx)
        # Passback?
        if passback:
            return stack_tup

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    fig = plt.figure(figsize=(5, 5))
    fig.clf()

    # Axes
    gs = gridspec.GridSpec(1, 1)


    # Axes
    ax = plt.subplot(gs[0,0])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.set_xlim(pvmnx.value)
    ax.set_ylim(yrng)

    ## ####
    # Labels
    ax.set_ylabel('Normalized Values')
    ax.set_xlabel('Relative Velocity (km/s)')

    # Zero line
    ax.plot([0]*2, yrng, 'b--')
    ax.plot(pvmnx, [1]*2, ':', color='pink')

    # Loop on stacking methods
    lbls = ['Avg', r'$1 + <\ln f>$', 'bspline']
    for kk in range(3):
        if kk == 0:
            velo,fx,tdict = qpq9a.stack_avg(stack_tup)
        elif kk == 1:
            velo,fx,tdict = qpq9a.stack_lnf(stack_tup)
        elif kk == 2:
            velo,fx,tdict = qpq9a.stack_bspline(stack_tup)

        # Stack
        ax.plot(velo, fx, drawstyle='steps', linewidth=2., label=lbls[kk])

        # Label
        ax.text(0.7, 0.10, xai.ion_name(aline.data)+' {:d}'.format(int(wrest.value)), 
            size='x-large', transform=ax.transAxes, ha='left', 
            bbox={'facecolor':'white', 'edgecolor':'white'})

    # Legend
    legend = plt.legend(loc='lower left', borderpad=0.3,
                        handletextpad=0.3)#, fontsize='small')

    #xputils.set_fontsize(ax,15.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('stack_trial: Wrote {:s}'.format(outfil))
    else: 
        plt.show()
    plt.close()

#######
def plt_trans(wrest=None):
    '''Plot individual transitions.
    Show EW and velocity window too
    '''
    reload(qpq9a)
    reload(qpqs)
    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA
    aline = AbsLine(wrest)
    outfil = 'fig_trans_{:s}.pdf'.format(xai.ion_name(aline.data))

    # Kin file
    kinfil = '../Analysis/QPQ9_kin_driver.dat' 
    kintab = Table.read(kinfil,format='ascii.fixed_width',
        delimiter='|')
    mtw = np.where(kintab['wrest']*u.AA == wrest)[0]
    nplt = len(mtw) # We can show the bad ones too

    # Load QPQ9
#    qpq9 = qpq9a.load_qpq(wrest)
    qpq9 = qpqutils.load_qpq(9,wrest=wrest)

    vmnx = [-1500., 1500] * u.km/u.s
    ymnx = (-0.1, 1.1)
    #if stack_tup is None:
    #    stack_tup = qpq9a.load_stack_img(qpq9, wrest, vmnx=vmnx)
    #fin_velo, stck_img, stck_msk, all_dict = stack_tup

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(5, 8))
    plt.clf()

    # Axes
    ncol = 3
    nrow = (nplt//ncol) + ((nplt % ncol) > 0)
    gs = gridspec.GridSpec(nrow, ncol)

    # Loop on systems
    for ss,qpq in enumerate(qpq9):
        # Load spectrum
        wvobs = wrest*(1+qpq['FG_ZIR'])
#        idict = qpqs.spec_qpq_wvobs(
        idict = qpqs.spec_wvobs(
            (qpq['RA']*u.deg,qpq['DEC']*u.deg), wvobs, 
            normalize=True, high_res=1)
        #
        ax = plt.subplot(gs[ss%nrow,ss//nrow])
        ax.set_xlim(vmnx.value)
        ax.set_ylim(ymnx)

        if ((ss+1)%nrow) == 0: 
            ax.set_xlabel('Relative Velocity (km/s)', size=9.)
        else:
            ax.get_xaxis().set_ticks([]) 
        #ax.set_ylabel('Normalized flux', size=9.)

        #  Generate stuff
        velo = idict['spec'].relative_vel(wvobs)

        # Plot original data
        #subp = idict['spec'].sub_pix
        ax.plot(velo, idict['spec'].flux, drawstyle='steps', color='black')

        # Plot processed
        #ax.plot(fin_velo, stck_img[ss,:], color='green', drawstyle='steps')

        # Label
        ax.text(-900., 0., '{:d}: {:s} {:g}'.format(ss,
            qpq['NAME'],qpq['FG_ZIR']),
            size=7., ha='left', bbox={'facecolor':'white', 'edgecolor':'white'})

        # Lines
        ax.plot(vmnx.value, [1.]*2, 'g:')
        ax.plot([0]*2, ymnx, 'b--')

        # Font
        xputils.set_fontsize(ax,7.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('lls_figs_ionic.fig_ionic_lls: Wrote {:s}'.format(outfil))
    else:
        plt.show()


#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(1)] ))
    else:
        flg_fig = int(flg_fig)

    # Experiment
    if (flg_fig % 2**1) >= 2**0:
        experiment()

    # Simple stack
    if (flg_fig % 2**2) >= 2**1:
        simple_stack()

    # Stack trials
    if (flg_fig % 2**3) >= 2**2:
        stack_trials()

    # Transitions
    if (flg_fig % 2**4) >= 2**3:
        plt_trans(wrest=1334.5323*u.AA)


    # All done
    print('All done!')


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_fig = 0 
        #flg_fig += 1     # Experiment
        #flg_fig += 2**1  # Simple stack
        #flg_fig += 2**2  # Stack trials
        flg_fig += 2**3  # CII 1334 Transitions
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
