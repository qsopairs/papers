""" Generate the TPE sample
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
import numpy as np
import pdb
import warnings

from collections import OrderedDict
import h5py

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from collections import OrderedDict

from astropy.table import Table
from astropy import units as u

from linetools import utils as ltu

from specdb.specdb import IgmSpec

this_file = __file__
qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'

def plot_stack(stack, outfil):
    # Quick Plot
    plt.clf()
    ax = plt.gca()
    ax.plot(stack.wavelength, stack.flux, 'k', drawstyle='steps-mid')
    ax.set_xlabel('Rest wavelength')
    ax.set_ylabel('Normalized flux')
    ax.set_xlim(1170., 1270.)
    ax.set_ylim(0.35, 1.0)
    # Lya
    ax.plot([1215.67]*2, [0., 2.], 'g--')
    plt.savefig(outfil)
    print("Wrote {:s}".format(outfil))


def plot_spec_img(spec, outfil):
    """ Image of the spectra
    Parameters
    ----------
    spec
    outfil

    Returns
    -------

    """
    # Cut down
    all_flux = spec.data['flux']
    all_sig = spec.data['sig']
    gdp = (spec.wavelength.value > 1170.) & (spec.wavelength.value < 1270)
    sub_flux = all_flux[:,gdp]
    # Zero out bad pixels
    sub_sig = all_sig[:,gdp]
    bad = sub_sig <= 0.
    sub_flux[bad] = 1.
    #
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    ax = plt.gca()
    #extent=[C0_val[0], C0_val[-1], C1_val[0], C1_val[-1], ]
    #aspect=np.abs((C0_val[-1]-C0_val[0])/(C1_val[-1]-C1_val[0]))
    cax = ax.imshow(sub_flux, origin='lower', cmap=plt.cm.gist_heat)
    cax.set_clim(vmin=-0.5, vmax=1.5)
    cb = fig.colorbar(cax, fraction=0.046, pad=0.04)
    #cb.set_label(r'$12 - \epsilon_{\rm '+lion+r'} - \log[x({\rm '+lion+r'^{++}})/x({\rm H^0})]$')
    # Lya
    plt.savefig(outfil)
    print("Wrote {:s}".format(outfil))


def plot_sample(spec_file, outfil, dv=9000., dwv=30.,
                fg_wvlim=np.array([1200.,2830.]),
                bg_wvlim=np.array([1020.,1930.])):
    import sys
    from xastropy.plotting import utils as xputils

    sys.path.append(os.path.abspath("./py"))
    import tpe_stack as tstack
    # TPE
    cuts, xspec, tpe, spec_tbl, rebin_spec = tstack.stack_spec(spec_file)
    npair = len(tpe)
    assert npair == xspec.nspec

    # Load spectral datasets
    igmsp = IgmSpec()
    qpq_file = os.getenv('DROPBOX_DIR')+'/QSOPairs/spectra/qpq_oir_spec.hdf5'
    qpq = IgmSpec(db_file=qpq_file, skip_test=True)
    #

    pp = PdfPages(outfil)
    plt.figure(figsize=(9, 5))#,dpi=100)
    nobj = 2
    gs = gridspec.GridSpec(nobj*2,8)

    zem_lines = OrderedDict()  # Taking these from LINEWAVESHIFT
    zem_lines['Lya'] = 1218.2121
    zem_lines['CIV'] = 1544.662
    zem_lines['CIII'] = 1903.910
    zem_lines['MgII'] = 2799.402
    lsz = 11.
    asz = 9.

    npair = 9
    for qq,row in enumerate(tpe):
        # Indexing
        ii = qq % nobj

        # f/g QSO first
        fg_coord = ltu.radec_to_coord((row['FG_RA'], row['FG_DEC']))
        if row['FG_IGM_ID'] > 0:
            fg_spec, fg_meta = igmsp.spectra_from_coord(fg_coord)
        else:
            fg_spec, fg_meta = qpq.spectra_from_coord(fg_coord)

        # Full f/g
        ax_fg = plt.subplot(gs[2*ii, 0:4])
        ax_fg.plot(fg_spec.wavelength, fg_spec.flux, 'k', drawstyle='steps-mid')
        wvmnx = fg_wvlim*(1+row['FG_Z'])
        ax_fg.set_xlim(wvmnx)
        ax_fg.get_yaxis().set_ticks([])
        ax_fg.set_ylim(0., np.max(fg_spec.flux))

        lbl = 'FG{:s}{:s}'.format(
                fg_coord.ra.to_string(unit=u.hour,sep='',pad=True, precision=1),
                fg_coord.dec.to_string(sep='',pad=True,alwayssign=True, precision=1))
        ax_fg.text(0.9, 0.85, lbl, transform=ax_fg.transAxes, color='black', size=8., ha='right')#, bbox={'facecolor':'white'})
        xputils.set_fontsize(ax_fg,asz)

        # Cutouts
        for kk,label in enumerate(zem_lines.keys()):
            line = zem_lines[label]
            velo = fg_spec.relative_vel(line*(1+row['FG_Z'])*u.AA)
            gdp = (velo.value > (-1*dv)) & (velo.value < dv)
            if np.sum(gdp) > 0:
                # Big plot
                ax_fg.plot([line*(1+row['FG_Z'])]*2, [-1e9,1e9], 'g--')
                # Sub
                ax = plt.subplot(gs[2*ii+1, kk])
                ax.plot(velo, fg_spec.flux, 'k', drawstyle='steps-mid')
                ax.get_yaxis().set_ticks([])
                # Limits
                ax.set_xlim([-1*dv,dv])
                maxf = np.max(fg_spec.flux[gdp])
                ax.set_ylim(-0.1*maxf, 1.2*maxf)
                ax.plot([0., 0.], [-1e9, 1e9], 'g--')
                # Text
                ax.text(0.1, 0.10, label, transform=ax.transAxes, ha='left', size=lsz)
                xputils.set_fontsize(ax,7.)

        # b/g QSO next
        bg_coord = ltu.radec_to_coord((row['BG_RA'], row['BG_DEC']))
        xspec.select = qq

        # Full b/g
        ax = plt.subplot(gs[2*ii, 4:])
        ax.plot(xspec.wavelength, xspec.flux, 'k', drawstyle='steps-mid')
        ax.plot(xspec.wavelength, xspec.co, '--', color='cyan')
        wvmnx = bg_wvlim*(1+row['BG_Z'])
        ax.set_xlim(wvmnx)
        ax.set_ylim(0., np.max(xspec.co)*1.3)
        ax.get_yaxis().set_ticks([])
        # Label
        ax.plot([1215.67*(1+row['FG_Z'])]*2, [-1e9,1e9], 'g--')
        lbl = '{:s}_{:s}{:s}'.format(row['GROUP'],
                                     bg_coord.ra.to_string(unit=u.hour,sep='',pad=True, precision=1),
                                     bg_coord.dec.to_string(sep='',pad=True,alwayssign=True, precision=1) )
        ax.text(0.9, 0.88, lbl, transform=ax.transAxes, color='black', size=8., ha='right')#, bbox={'facecolor':'white'})
        xputils.set_fontsize(ax,asz)

        # Zoom in
        ax = plt.subplot(gs[2*ii+1, 4:])
        lya = 1215.67 #* (1+row['FG_Z'])
        gdp2 = (xspec.wavelength.value/(1+row['FG_Z']) > (lya-dwv)) & (
            xspec.wavelength.value/(1+row['FG_Z']) < (lya+dwv))
        maxf = np.max(xspec.co[gdp2])
        ax.set_ylim(-0.1*maxf, 1.7*maxf)
        ax.set_xlim(np.array([-1*dwv,dwv])+1215.67)
        ax.plot([1215.67*(1+row['FG_Z'])]*2, [-1e9,1e9], 'g--')
        # Plots
        ax.plot(xspec.wavelength/(1+row['FG_Z']), xspec.flux, 'k', drawstyle='steps-mid')
        ax.plot(xspec.wavelength/(1+row['FG_Z']), xspec.co, '--', color='cyan')
        ax.plot(xspec.wavelength/(1+row['FG_Z']), xspec.sig, 'r:')
        ax.get_yaxis().set_ticks([])
        xputils.set_fontsize(ax,asz)

        # Finish
        if (ii == (nobj-1)) or (qq == (npair-1)):
            plt.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.0)
            #plt.subplots_adjust(hspace=0)
            pp.savefig(bbox_inches='tight')
            plt.close()
        if qq == npair: # For debugging
            break
    # Finish
    pp.close()
    plt.close()
    print("Wrote: {:s}".format(outfil))


