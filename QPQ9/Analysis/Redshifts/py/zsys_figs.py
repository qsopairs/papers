# Module for zsys QA

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import subprocess, glob, copy, os

from scipy.signal import medfilt

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy import constants as const
from astropy import units as u
from astropy.units import Unit
from astropy.io import ascii, fits
from astropy.table import QTable, Table, Column
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.relativity import velocities as arv

from xastropy.xutils import fits as xxf
from xastropy.plotting import utils as xputils
from xastropy.spec import readwrite as xsr
from xastropy.xutils import xdebug as xdb


##
def zsys_flag_str(idx):  
    ''' 
    List of the lines used
    '''
    val = [ r'H$\alpha$', 
        r'H$\beta$',
        'MgII',
        'CIII]' ,           # 3
        'CIV'   ,           # 4
        'SiIV'  ,           # 5
        'SiIV-CIV' ,        # 6
        'SiIV-CIII]' ,      # 7
        'CIV-CIII]' ,       # 8
        'SiIV-CIV-CIII]' ,  # 9
        r'Ly$\alpha$', 
        '[OIII]'  ,         # 11
        '[OII]']            # 12
    # Return
    try:
        return val[idx]
    except IndexError:
        return 'NULL'

##
def zsys_qa(zem_fil, outfil=None, clobber=False):
    '''
    Generate a QA PDF file from a zem_fil
    '''
    # Outfil
    i0 = zem_fil.find('/')+1
    i1 = zem_fil.find('_zem')
    outfil = 'zQA/'+ zem_fil[i0:i1] + '_zQA.pdf'
    name = zem_fil[i0:i1]

    # Check
    dum = glob.glob(outfil)
    if (len(dum) > 0) & (not clobber):
        print('zsys_qa: Not overwriting {:s}'.format(outfil))
        return
    else:
        print('zsys_qa: Generating/over-writing {:s}'.format(outfil))

    #   
    cspeed = const.c.to('km/s')

    # Read the zem file
    out_struct = QTable.read(zem_fil)
    zstruct = QTable.read(zem_fil,2)
    nline = len(zstruct)

    # Read the spectrum
    spec = xsr.readspec(out_struct['SPEC_FIL'][0])
    npix = len(spec.sig)
    scale = 1. # Wavelength
    if np.median(spec.flux) < 1e-12:
        fscale = 1e17 # Flux
    else:
        fscale = 1.
    loglam = np.log10( spec.dispersion.to('micron').value)


    # Start the plot
    wbox = {'facecolor':'white', 'edgecolor':'white', 'pad':0}
    xtitle = 0.01
    ytitle = 0.90
    tsz = 12.
    lsz = 6.
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(9, 5))#,dpi=100)
    plt.clf()
    gs = gridspec.GridSpec(2, nline)

    ## ######
    # Spectrum first
    max_3 = 2.0*np.max(zstruct['LINEPEAK'] + zstruct['LINEALLBUTLEVEL'])
    igood = np.where(spec.sig > 0.0)[0]
    xrnge = [np.min(spec.dispersion[igood].to('micron').value), 
        np.max(spec.dispersion[igood].to('micron').value)]
    yrnge = [0.0, max_3] 
    #yrange = [0.0, 3] 

    ax = plt.subplot(gs[1,:])
    ax.set_xlabel(r'$\lambda (\mu \rm m)$')
    ax.set_ylabel(r'$f_\lambda \; (10^{-17} \; \rm erg/s/cm^2/\AA$)')
    ax.set_xlim(xrnge)
    ax.set_ylim(yrnge)

    # Smooth 
    nsmth = np.max( np.array( [npix//3000, 3] ))
    smth_spec = spec.box_smooth(nsmth, preserve=False)
    ax.plot( smth_spec.dispersion.to('micron'), smth_spec.flux*fscale, 
        'k-',drawstyle='steps-mid', linewidth=0.5)
    ax.plot( smth_spec.dispersion.to('micron'), smth_spec.sig*fscale, 'r', 
        linewidth=1)
    #xdb.set_trace()

    for ii,line in enumerate(zstruct):
        lam_zmode = (line['LINEMODE']*u.AA).to('micron')
        if lam_zmode < (0.*u.AA):
            continue
        ax.plot( [lam_zmode.value/scale]*2, yrnge, '--', color = 'black')
        line_z = line['LINEMODE']/line['LINEWAVESHIFT'] - 1.0
        tmp = ''
        name_str = tmp.join(line['LINENAME'].split('_'))
        ystag = ii % 2
        # Text
        ax.text((lam_zmode-(20.*u.AA).to('micron')).value/scale, 
            (0.7+ystag*0.1)*yrnge[1], 
            r'$z_{'+'{:s}'.format(name_str)+'}$'+'={:0.3f}'.format(line_z),
            size=lsz, color='black', ha='left', bbox=wbox, rotation=90)

    #xdb.set_trace()
    # Title
    z_sys = out_struct['ZSYS_ZSYS'][0]
    zsys_flag = out_struct['ZSYS_FLAG'][0]
    err = out_struct['ZSYS_ERR'][0]
    zerr = (1.0 + z_sys)*err/const.c.to('km/s').value
    zerr_str = '{:.4f}'.format(zerr)
    zsys_str = r'$z_{\rm sys}='+'{:.4f}'.format(z_sys) +'\pm' + zerr_str+'$'
    sig_str = r'$\sigma_{z}=$'+ '{:d}'.format(int(err)) + ' km/s'
    if zsys_flag == -1:
       title = 'ERROR: Not a single line was measured. ZSYS UNDEFINED'
       color = 'red'
    else:
       title = name+'  | '+zsys_str+' | '+zsys_flag_str(zsys_flag)+' | ' + sig_str
       color = 'black'
    ax.text(xtitle, ytitle, title, size = tsz, color = color, 
        transform=ax.transAxes, ha='left')

    # Loop on the lines
    for ii,line in enumerate(zstruct):
        #
        lam_zmode = (line['LINEMODE']*u.AA).to('micron')
        if lam_zmode < (0.*u.AA):
            continue

        # Fit Color
        if 'Ly_alpha' in line['LINENAME']:
            yfitcolor = 'lightgreen'
            flg_lya = 1
        else:
            yfitcolor = 'magenta'
            flg_lya = 0
    
        # Axis
        ax = plt.subplot(gs[0,ii])

        left3 = np.max( [line['LINEPIX']-3.2*line['LINEGPIXLEFT'], 0] )
        right3 = np.min( [line['LINEPIX']+3.2*line['LINEGPIXRIGHT'], npix-1] )
        xrnge = [spec.dispersion[left3].to('micron').value, 
            spec.dispersion[right3].to('micron').value]
        irange = np.where( (spec.dispersion.to('micron').value >= xrnge[0]) & 
            (spec.dispersion.to('micron').value <= xrnge[1])) [0]
        if len(irange) > 0:   
            yrnge = [0.0, 2.0*(line['LINEPEAK'] + line['LINEALLBUTLEVEL'])] 
        else:
            yrnge = [0.0, 1.5*np.max(medfilt(spec.flux[irange], kernel_size=15))]
        ax.set_xlim(xrnge)
        ax.set_ylim(yrnge)
        dx = xrnge[1]-xrnge[0]
        xtick = np.max( [int(dx*100/2.)/100., 0.01] )
        #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))
        #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))

        #var = (ivar GT 0.0)/(sqrt(ivar) + (ivar EQ 0.0))
        tmp = ''
        name_str = tmp.join(line['LINENAME'].split('_'))

        ax.plot( spec.dispersion.to('micron'), spec.flux*fscale, 
            'k-',drawstyle='steps-mid', linewidth=0.5)
        ax.plot( spec.dispersion.to('micron'), spec.sig*fscale, 'r', linewidth=0.2)

        # Fit
        #   this fitmask is used in the deepest my_linebackfit.pro as
        #   the ultimate fitmask, which is defined in zsys_gaussfit.pro
        yfit = line['YFIT']
        bfit = line['BFIT']
        allbutline = line['ALLBUTLINE']
        fitmask = line['INNER_FITMASK']
        zflag = line['ZFLAG']
        left = np.max( [line['LINEPIX']-line['LINEGPIXLEFT'], 0] )
        right = np.min( [line['LINEPIX']+line['LINEGPIXRIGHT'], npix-1] )
        flux_now = spec.flux[left:right]*fscale
        fit_now = yfit[left:right]
        fitmask_now = fitmask[left:right]
        loglam_now = loglam[left:right]
        back_now = allbutline[left:right]
        frac_peak = back_now + line['PEAK_FRAC']*line['LINEPEAK']
        #   Take points which are either above the cut or for points that 
        #   were masked, interpolate by using the value of the fit.  
        flux_trim  = (flux_now >= frac_peak) & (fitmask_now > 0)
        intrp_trim = (fit_now >= frac_peak) & (fitmask_now == 0)
        peak_inds = np.where(flux_trim | intrp_trim)[0]
        npeak = len(peak_inds)
        if (npeak == 0) | flg_lya: 
            lam_mode = -1.0
            n_real = 0
            n_intr = 0
        else:
            loglam_peak = loglam_now[peak_inds]
            real_inds = np.where(flux_trim)[0]
            n_real = len(real_inds)
            intr_inds = np.where(intrp_trim)[0]
            n_intr = len(intr_inds)
            med = np.median(loglam_peak)
            mean = np.sum(loglam_peak)/float(npeak)
            mode =  3*med - 2*mean
            lam_mode = 10.0**mode
            #xdb.set_trace()
            '''
            if lam_mode NE lam_zmode THEN BEGIN
                splog, 'ERROR: modes dont match'
                stop
            ENDIF
            '''
        #xdb.set_trace()
        # Model fit
        ax.plot(spec.dispersion.to('micron'), yfit, color=yfitcolor)
        if flg_lya==0: 
            ax.plot(spec.dispersion.to('micron'), bfit, color='blue')
        if (n_real > 0) & (flg_lya==0): 
            ax.scatter(10.0**loglam_now[real_inds]/scale, 
                flux_now[real_inds], color = 'lightgreen', s=6., zorder=0.9, alpha=0.7)
        if (n_intr > 0) & (flg_lya==0):
            ax.scatter(10.0**loglam_now[intr_inds]/scale, fit_now[intr_inds],
                color = 'cyan', s=6., zorder=1, alpha=0.7)

        # Model fit
        flg_OII = 0
        flg_OIII = 0
        if ('O_III_5007' in line['LINENAME']) & (zflag != 0):
            flg_OIII = 1
            #xdb.set_trace()
            ax.plot(np.array([(out_struct['ZSYS_OIII_FPEAK']*u.AA).to('micron')]*2)/scale, 
                yrnge, ':', color='black')
            zmode_color = 'lightgreen'
        elif ('O_II_3727' in line['LINENAME']) & (zflag != 0):
            flg_OII = 1
            ax.plot(np.array([(out_struct['ZSYS_OII_FPEAK']*u.AA).to('micron')]*2)/scale, 
                yrnge, ':', color='black')
            zmode_color = 'lightgreen'
        else:
            zmode_color = 'black'


        # Centroid
        ax.plot([lam_zmode.value/scale]*2, yrnge, '--', color=zmode_color)
        ax.text((lam_zmode + 10.0*u.AA).to('micron').value/scale, yrnge[0] + 0.2*(yrnge[1]-yrnge[0]), 
            r'$\lambda=$'+'{:0.5f}'.format(lam_zmode.value/scale), size=lsz, color='black', bbox=wbox)
        #xdb.set_trace()

        # Labels
        ax.set_xlabel(r'$\lambda (\mu \rm m)$')
        ax.text(0.5, 0.92, name_str, transform=ax.transAxes, 
            size='smaller', ha='center',bbox=wbox)

        # Font size
        xputils.set_fontsize(ax,7.)

        if flg_lya: 
            continue

        out_string = 'S/N={:0.1f}'.format(line['LINE_SN500'])
        if flg_OIII:
            gdl='OIII'
        elif flg_OII:
            gdl='OII'
        else:
            continue
        xlbl = xrnge[0]*(1.0 + (360.0/cspeed.value)) 

        # Good enough?
        if line['LINE_SN500'] <= get_zkey(out_struct, gdl, 'SN_MIN'):
            color = 'red'
            out_string = out_string + '< {:0.2f}'.format(get_zkey(out_struct,gdl,'SN_MIN'))
        else:
            color = 'black'
        ax.text(xlbl, ylbl(yrnge,0.33), out_string, size=lsz, color=color, ha='left', bbox=wbox)
        # VDIFF
        out_string = r'$v_{diff}=$'+'{:d}'.format(int(get_zkey(out_struct,gdl,'VDIFF')))
        if get_zkey(out_struct,gdl,'VDIFF') > get_zkey(out_struct,gdl,'VDIFF_MAX'):
            out_string = out_string + '> {:d}'.format(int(get_zkey(out_struct,gdl,'VDIFF_MAX')))
            color = 'red'
        else:
            color = 'black'
        out_string = out_string + 'km/s'
        ax.text(xlbl, ylbl(yrnge,0.27), out_string, size=lsz, color=color, ha='left', bbox=wbox)
        # FORMERR
        out_string = r'$v_{form}=$'+ '{:d}'.format(int(get_zkey(out_struct,gdl,'FORMERR')))
        if get_zkey(out_struct,gdl,'FORMERR') > get_zkey(out_struct,gdl,'FORMERR_MAX'):
            out_string = out_string + '> {:d}'.format(int(get_zkey(out_struct,gdl,'FORMERR_MAX')))
            color = 'red'
        else:
            color = 'black'
        ax.text(xlbl, ylbl(yrnge,0.21), out_string, size=lsz, color=color, ha='left', bbox=wbox)
        # PEAK2CONT
        out_string = 'P2C= {:0.1f}'.format(float(get_zkey(out_struct,gdl,'PEAK2CONT')))
        if get_zkey(out_struct,gdl,'PEAK2CONT') <= get_zkey(out_struct,gdl,'PEAK2CONT_MIN'):
            out_string = out_string + '< {:.1f}'.format(float(get_zkey(out_struct,gdl,'PEAK2CONT_MIN')))
            color = 'red'
        else:
            color = 'black'
        ax.text(xlbl, ylbl(yrnge,0.15), out_string, size=lsz, color=color, ha='left', bbox=wbox)

    
    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        pp.savefig(bbox_inches='tight')
        pp.close()
    else: 
        plt.show()
    plt.close()

def get_zkey(ostruct, line, subkey):
    '''
    Grab a ZSYS key.  Usually for OIII or OII
    '''
    key = 'ZSYS_'+line+'_'+subkey
    #
    return ostruct[key]

def ylbl(yrnge,yoff):
    return yrnge[0] + yoff*(yrnge[1]-yrnge[0])
    
# ##################################################
# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg_test = 0
    #flg_test += 2**0  # Generate the Structure
    flg_test += 2**1  # PDF QA plots

    if flg_test % 2**1 >= 2**0:
        mk_qpq9()

    if flg_test % 2**2 >= 2**1:
        all_zem = glob.glob('zem/S*fits')
        for zem_fil in all_zem:
            zsys_qa(zem_fil) 
