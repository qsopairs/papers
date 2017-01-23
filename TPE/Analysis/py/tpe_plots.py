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

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo



from xastropy.xutils import xdebug as xdb

this_file = __file__

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

