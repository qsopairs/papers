#Module for QPQ9 figures
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import copy,os,sys,imp

import matplotlib as mpl
import numpy as np
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.table import QTable, Table
from astropy.io import ascii,fits
from astropy import units as u
from astropy.utils.misc import isiterable
from astropy.modeling import models

from linetools.spectralline import AbsLine

from xastropy.plotting import utils as xputils
from xastropy.atomic import ionization as xai

sys.path.append(os.path.abspath("../Analysis/Stacks/py"))
import qpq9_stacks as qpq9k

#  Plot the Experiment
def experiment(outfil=None,wrest=[1334.5323*u.AA,1548.195*u.AA,2796.354*u.AA],S2N_cut=5.5/u.AA,
               zfg_mnx=(1.6,9999)):

    if outfil is None:
        outfil = 'fig_experiment.pdf'
    fontsize = 20

    # fixes
    if not isiterable(wrest):
        wrest = [wrest]

    # Read QPQ9, determine g_UV range
    QPQ9_fil = imp.find_module('enigma')[1] + '/data/qpq/qpq9_final.fits'
    QPQ9 = QTable.read(QPQ9_fil)
    good_z = np.where((QPQ9['Z_FG'] > zfg_mnx[0]) & (QPQ9['Z_FG'] < zfg_mnx[1]))
    logguv_min = np.min(np.log10(QPQ9[good_z]['G_UV']))
    logguv_max = np.max(np.log10(QPQ9[good_z]['G_UV']))

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(5,10))
    fig.clf()
    gs = gridspec.GridSpec(len(wrest),1)


    # Loop the lines
    for kk,line in enumerate(wrest):

        # Get line info
        aline = AbsLine(line)

        # Load stack_tup
        stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=line,S2N_cut=S2N_cut,zfg_mnx=zfg_mnx,plot_indiv=False)
        # Mask
        fin_velo, stck_img, stck_msk, all_dict = stack_tup0
        stck_mskN = copy.deepcopy(stck_msk)
        idx_mask = []
        for ii,idict in enumerate(all_dict):
            if idict is None:
                continue
            if xai.ion_name(aline.data) == 'CII':
                if 'J1508+3635' in idict['qpq']['NAME']: #DLA, should be excluded now
                    idx_mask.append(ii)
            if xai.ion_name(aline.data) == 'CIV':
                if 'J1002+0020' in idict['qpq']['NAME']: #overlaps with BAL of bg quasar
                    idx_mask.append(ii)
            if xai.ion_name(aline.data) == 'MgII':
                if (('J0822+1319' in idict['qpq']['NAME']) | ('J0908+4215' in idict['qpq']['NAME']) |
                        ('J1242+1817' in idict['qpq']['NAME']) | ('J1622+2031' in idict['qpq']['NAME'])):
                    #BAL, strange emission, sky emission, sky emission
                    idx_mask.append(ii)
        for idx in idx_mask:
            stck_mskN[idx,:] = 0.

        # Save the sub-sample
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
                sv_symsize.append(35)
            else:
                sv_symsize.append(70)

        # Axes
        ax = plt.subplot(gs[kk,0])
        ax.set_xlim(0,300)
        ax.set_ylim(np.min(sv_zfg)-0.1,np.max(sv_zfg)+0.1)
        ax.tick_params(labelsize=fontsize,length=5,width=1)
        # Labels
        ax.set_ylabel(r'$z_{\rm fg}$')
        if kk < len(wrest)-1:
            ax.set_xticklabels("",visible=False)
            pass
        else:
            ax.set_xlabel(r'$R_\perp$ (kpc)')
        ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])*0.05+ax.get_xlim()[0],
                (ax.get_ylim()[1]-ax.get_ylim()[0])*0.85+ax.get_ylim()[0],
                xai.ion_name(aline.data),size=fontsize)
        # Scatter
        sc = ax.scatter(sv_Rphys,sv_zfg,s=sv_symsize,c=np.log10(sv_gUV),alpha=0.6,edgecolors='none',
                        vmin=logguv_min,vmax=logguv_max)

        # Font
        xputils.set_fontsize(ax,fontsize)

    # Color bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.05,0.07,0.05,0.9])
    cb = fig.colorbar(sc,cax=cbar_ax)
    cb.ax.tick_params(labelsize=fontsize,length=5,width=1)
    cb.set_label(r'$\log_{10} \; g_{\rm UV}$',size=fontsize)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig(bbox_inches='tight')
    pp.close()
    print('experiment: Wrote {:s}'.format(outfil))


#  Stacks and Gaussian models
def stacks_and_fits(outfil=None):

    if outfil is None:
        outfil = 'fig_stack_and_fits.pdf'
    fontsize = 15

    # Lines
    lines = [1334.5323,1548.195,2796.354]*u.AA

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(10,8))
    fig.clf()
    gs = gridspec.GridSpec(len(lines),2)

    for kk,line in enumerate(lines):

        # Get line info
        aline = AbsLine(line)

        # Load the mean stack
        path = '../Analysis/Stacks/'
        mean_stack = fits.open(path+'Output/QPQ9_zIRMgII_'+aline.name[-4:]+'_mean.fits')
        relativistic_equiv = u.doppler_relativistic(line)
        velo = (mean_stack[1].data*u.AA).to(u.km/u.s,equivalencies=relativistic_equiv)

        # Load the model
        params = (ascii.read(path+aline.name[:-5]+'_MgII_mean_fit.dat'))[0]
        model_conti = models.Const1D(amplitude=params['amplitude_0'])
        model_gauss = models.GaussianAbsorption1D(
            amplitude=params['amplitude_1'],mean=params['mean_1'],stddev=params['stddev_1'])
        if len(params) == 4:
            model = model_conti*model_gauss
        else:
            model_gauss2 = models.GaussianAbsorption1D(
                amplitude=params['amplitude_2'],mean=params['mean_2'],stddev=params['stddev_2'])
            model = model_conti*model_gauss*model_gauss2

        # Axes
        ax = plt.subplot(gs[kk,0])
        ax.set_xlim(-2500,2500)
        ax.set_ylim(0.79,1.03)
        ax.tick_params(labelsize=fontsize,length=5,width=1)
        # Labels
        ax.set_ylabel('Normalized Flux')
        ax.set_xlabel('Relative Velocity (km/s)')
        ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])*0.05+ax.get_xlim()[0],
                (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05+ax.get_ylim()[0],
                xai.ion_name(aline.data)+', mean',size=fontsize)

        # Plot
        plt.plot(velo.value, mean_stack[0].data,drawstyle='steps-mid',linewidth=2.,color='k')
        plt.plot(velo.value, model(velo.value),color='b')
        plt.plot([model.mean_1.value,model.mean_1.value],[0,2],'b--')

        # Font
        xputils.set_fontsize(ax,fontsize)

        # Do median stack
        median_stack = fits.open(path+'Output/QPQ9_zIRMgII_'+aline.name[-4:]+'_med.fits')
        params = (ascii.read(path+aline.name[:-5]+'_MgII_median_fit.dat'))[0]
        model_conti = models.Const1D(amplitude=params['amplitude_0'])
        model_gauss = models.GaussianAbsorption1D(
            amplitude=params['amplitude_1'],mean=params['mean_1'],stddev=params['stddev_1'])
        if len(params) == 4:
            model = model_conti*model_gauss
        else:
            model_gauss2 = models.GaussianAbsorption1D(
                amplitude=params['amplitude_2'],mean=params['mean_2'],stddev=params['stddev_2'])
            model = model_conti*model_gauss*model_gauss2
        ax = plt.subplot(gs[kk,1])
        ax.set_xlim(-2500,2500)
        ax.set_ylim(0.79,1.03)
        ax.tick_params(labelsize=fontsize,length=5,width=1)
        ax.set_xlabel('Relative Velocity (km/s)')
        ax.set_yticklabels("",visible=False)
        ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])*0.05+ax.get_xlim()[0],
                (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05+ax.get_ylim()[0],
                xai.ion_name(aline.data)+', median',size=fontsize)
        plt.plot(velo.value, median_stack[0].data,drawstyle='steps-mid',linewidth=2.,color='k')
        plt.plot(velo.value, model(velo.value),color='b')
        plt.plot([model.mean_1.value,model.mean_1.value],[0,2],'b--')
        xputils.set_fontsize(ax,fontsize)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    pp.savefig(bbox_inches='tight')
    pp.close()
    print('stacks_and_fits: Wrote {:s}'.format(outfil))


# Mean stack for MgII at z = 1
def stack_z1(outfil=None):

    if outfil is None:
        outfil = 'fig_stack_z1.pdf'
    fontsize = 30

    line = 2796.354*u.AA

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(10,5))
    fig.clf()
    gs = gridspec.GridSpec(1,1)

    # Get line info
    aline = AbsLine(line)

    # Load the mean stack
    path = '../Analysis/Stacks/'
    mean_stack = fits.open(path+'Output/QPQ9_zIRMgII_'+aline.name[-4:]+'_z1_mean.fits')
    relativistic_equiv = u.doppler_relativistic(line)
    velo = (mean_stack[1].data*u.AA).to(u.km/u.s,equivalencies=relativistic_equiv)

    # Load the model
    params = (ascii.read(path+aline.name[:-5]+'_z1_mean_fit.dat'))[0]
    model_conti = models.Const1D(amplitude=params['amplitude_0'])
    model_gauss = models.GaussianAbsorption1D(
        amplitude=params['amplitude_1'],mean=params['mean_1'],stddev=params['stddev_1'])
    model_gauss2 = models.GaussianAbsorption1D(
        amplitude=params['amplitude_2'],mean=params['mean_2'],stddev=params['stddev_2'])
    model = model_conti*model_gauss*model_gauss2

    # Axes
    ax = plt.subplot(gs[0,0])
    ax.set_xlim(-2500,2500)
    ax.set_ylim(0.93,1.02)
    # Labels
    ax.tick_params(labelsize=fontsize,length=5,width=1)
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Relative Velocity (km/s)')
    ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])*0.05+ax.get_xlim()[0],
             (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05+ax.get_ylim()[0],
             xai.ion_name(aline.data)+', $z\sim0.9$, mean',size=fontsize)

    # Plot
    plt.plot(velo.value, mean_stack[0].data,drawstyle='steps-mid',linewidth=2.,color='k')
    plt.plot(velo.value, model(velo.value),color='b')
    plt.plot([model.mean_1.value,model.mean_1.value],[0,2],'b--')

    # Font
    xputils.set_fontsize(ax,fontsize)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.0)
    pp.savefig(bbox_inches='tight')
    pp.close()
    print('stack_z1: Wrote {:s}'.format(outfil))
