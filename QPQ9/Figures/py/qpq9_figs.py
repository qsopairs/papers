#Module for QPQ9 figures
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import copy
import os
import sys

import matplotlib as mpl
import numpy as np
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from astropy.table import QTable, Table
from astropy import units as u

from linetools.spectralline import AbsLine

from xastropy.plotting import utils as xputils
from xastropy.atomic import ionization as xai

sys.path.append(os.path.abspath("../Analysis/Stacks/py"))
import qpq9_stacks as qpq9k

#  Plot the Experiment
def experiment(outfil=None,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999)):

    # Get line info
    aline = AbsLine(wrest)

    if outfil is None:
        outfil = 'fig_experiment_'+xai.ion_name(aline.data)+'.pdf'

    fontsize = 40

    # Load stack_tup
    stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=wrest,S2N_cut=S2N_cut,zfg_mnx=zfg_mnx)
    # Mask
    fin_velo, stck_img, stck_msk, all_dict = stack_tup0
    stck_mskN = copy.deepcopy(stck_msk)
    idx_mask = []
    for ii,idict in enumerate(all_dict):
        if idict is None:
            continue
        if 'J1508+3635' in idict['qpq']['NAME']: #DLA not excluded by forest cut, should be excluded now
            idx_mask.append(ii)
    for idx in idx_mask:
        stck_mskN[idx,:] = 0.

    sv_Rphys = []
    sv_zfg = []
    sv_gUV = []
    sv_symsize = []
    for ii,idict in enumerate(all_dict):
        if idict is None:
            continue
        if np.sum(stck_mskN[ii,:]) > 0.:
            sv_Rphys.append(idict['qpq']['R_PHYS'])
            sv_zfg.append(idict['qpq']['Z_FG'])
            sv_gUV.append(idict['qpq']['G_UV'])
            if idict['qpq']['ZFG_LINE'] == '[OIII]':
                sv_symsize.append(70)
            else:
                sv_symsize.append(140)
    
    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8,5))
    fig.clf()
    gs = gridspec.GridSpec(1,1)

    # Axes
    ax = plt.subplot(gs[0,0])
    ax.set_xlim(0,300)
    ax.set_ylim(np.min(sv_zfg)-0.1,np.max(sv_zfg)+0.1)
    ax.tick_params(labelsize=fontsize,length=5,width=1)
    # Labels
    ax.set_ylabel(r'$z_{\rm fg}$')
    ax.set_xlabel(r'$R_\perp$ (kpc)')
    ax.text((ax.get_xlim()[0])*0.10,(ax.get_ylim()[1])*0.93,xai.ion_name(aline.data),size=fontsize)

    # Scatter
    sc = ax.scatter(sv_Rphys,sv_zfg,s=sv_symsize,c=np.log10(sv_gUV),alpha=0.6,edgecolors='none')

    # Font
    xputils.set_fontsize(ax,fontsize)
    # Color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.05,0.07,0.05,0.9])
    cb = fig.colorbar(sc,cax=cbar_ax)
    tick_locator = MaxNLocator(nbins=7)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.tick_params(labelsize=fontsize,length=5,width=1)
    cb.set_label(r'$\log_{10} \; g_{\rm UV}$',size=fontsize)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('experiment: Wrote {:s}'.format(outfil))
    else: 
        plt.show()

#  Simple stack plot(s)
def simple_stack(outfil=None, all_stack=None, passback=False):
    print('inside simple_stack')

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
#    lines = [1215.6701, 1334.5323, 1548.195]*u.AA
    lines = [1334.5323]*u.AA
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
        ax.text(0.07, 0.10, xai.ion_name(aline.data)+' {:d}'.format(int(wrest.value)), 
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
    print('inside plt_trans')
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
        if (qpq['NAME']=='BOSSJ1006+4804'):
            continue
        if (qpq['NAME']=='APOJ1420+1603'):
            continue
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
