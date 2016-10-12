"""
#;+ 
## DEPRECATED
## DEPRECATED
## DEPRECATED
#
#; NAME:
#; qpq_spec
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for dealing with QPQ samples
#;   May-2015 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

# imports
import numpy as np
import glob, os 

from scipy.interpolate import interp1d

from astropy.cosmology.core import LambdaCDM
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy.io import fits, ascii
from astropy import units as u
from astropy.utils import isiterable

from xastropy.spec import readwrite as xsr
from xastropy.obs import radec as xrad
from xastropy.xutils import fits as xxf
from xastropy.xutils import xdebug as xdb
from xastropy.files import general as xfg

##
def sdss_flux2mag(flux, filt):
    '''
    Calculate the magnitude given a flux

    Parameters:
    -----------
    flux: float
      Flux, usually de-reddened
    filter: string
      SDSS filter:  ugriz

    Returns:
    -----------
    mag: float
      apparent magnitude
    '''
    #  Smoothing parameter
    b_parms = {'u': 1.4, 'g': 0.9, 'r': 1.2, 'i': 1.8, 'z': 7.4}
    b_parm = b_parms[filt]

    # Calculate
    a = 1.08574
    ln10_min10 = -23.02585
    # 
    m = -a*(np.arcsinh(5.0*flux/b_parm)+ln10_min10 + np.log(b_parm))    

    # Return
    return m

##
def m_iz2_qso(mag, ztot, omega_m=0.3, omega_v=0.7, w=1.0, lit_h=0.7):
    '''
    Calculate Absolute magnitude
       Note these are M_i(z = 2) absolute magnitudes. 

    Parameters:
    -----------
    mag:  float
      SDSS magnitude
    ztot:  float
      Redshift of interest
    omega_m: float (0.3)
      Dark matter parameter
    omega_v: float (0.7)
      Vacuum energy parameter
    lit_h: float (0.7)
      Hubble parameter
    '''
    if isiterable(mag):
        try:
            query = len(mag) == len(ztot)
        except:
            xdb.set_trace()
        else:
            if not query:
                raise ValueError('The imag and ztot arrays must have the same size')
  
    # k-correction
    kfile = os.getenv('QSO_DIR') + '/misc/templates/kcorr_Miz2_richards.dat'
    k_tbl = ascii.read(kfile, data_start=23, names=('z_in','Kz_in'))
    #rdfloat, kfile, z_in, Kz_in, skip = 22
  
    fKz = interp1d(k_tbl['z_in'], k_tbl['Kz_in'])
    Kz = fKz(ztot) * u.mag

    # Define cosmology
    cosmo = LambdaCDM(H0=lit_h*100, Om0=omega_m, Ode0=omega_v)

    #
    # Distance modulus
    DM = cosmo.distmod(ztot)

    # And then
    M_abs = mag*u.mag - DM - Kz
  
    return M_abs

'''
# ################
if __name__ == "__main__":

    flg_fig = 0 
    flg_fig += 2**0  # Test file grabbing
    #flg_fig += 2**1  # Test with spectrum
    
    # 
    if (flg_fig % 2**1) >= 2**0:
        bg_qso = 'J005718.90-000134.7'
        get_spec_files(bg_qso)
'''
