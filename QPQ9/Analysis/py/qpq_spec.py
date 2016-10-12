"""
## DEPRECATED
###   see enigma.qpq.spec
#;+ 
#; NAME:
#; qpq_spec
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for dealing with spectra in the QPQ dataset 
#;   May-2015 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

# imports
import numpy as np
import glob, os, copy

from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy.io import fits
from astropy import units as u
from astropy import constants as const

from linetools.spectra import io as lsio

from xastropy.obs import radec as xrad
from xastropy.xutils import fits as xxf
from xastropy.xutils import xdebug as xdb
from xastropy.files import general as xfg



# ################
if __name__ == "__main__":

    flg_fig = 0 
    flg_fig += 2**0  # Test file grabbing
    #flg_fig += 2**1  # Test with spectrum
    
    # 
    if (flg_fig % 2**1) >= 2**0:
        bg_qso = 'J005718.90-000134.7'
        get_spec_files(bg_qso)
