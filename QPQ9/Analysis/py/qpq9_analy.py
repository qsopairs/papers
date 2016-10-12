# Module for QPQ9 analysis

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, copy, os, sys

from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep # bspline


from astropy import units as u
from astropy.units import Unit
from astropy.io import ascii, fits
from astropy.table import QTable, Table, Column
from astropy.coordinates import SkyCoord
from astropy import constants as const

from linetools.spectralline import AbsLine
from linetools.lists import linelist as lll

from xastropy.xutils import fits as xxf
from xastropy.xutils import xdebug as xdb
from xastropy.atomic import ionization as xai
from xastropy.xguis import utils as xxgu

# Local 
sys.path.append(os.path.abspath("./py"))
import qpq_spec as qpqs

high_res_inst = ['ESI', 'MAGE', 'MIKE', 'HIRES', 'XSHOOTER', 'UVES']


##
def stack_bspline(in_tup, dv=100*u.km/u.s, debug=False, **kwargs):
    '''Provided a table of b/g QSOs or a stack image,
      stack their spectra with a b-spline approach

    Parameters:
    ------------
    in_tup: Tuple
      (QPQ9 Table, wrest)
      (fin_velo, stck_img, stck_msk, all_dict)
    dv: Quantity, optional
      Velocity spacing of breakpoints [100 km/s]

    Returns:
    ------------
    fin_velo
    fin_flx
    all_dict
    '''
    reload(qpqs)

    # Generate the image
    if len(in_tup) == 2: 
        fin_velo, stck_img, stck_msk, all_dict = load_stack_img(
            in_tup[0], in_tup[1], **kwargs)
    else:
        fin_velo, stck_img, stck_msk, all_dict = in_tup

    # Generate the full array for b-spline action
    list_vel = []
    list_fx = []
    list_wgt = []
    for tdict in all_dict:
        if tdict is None:
            continue
        vel = tdict['spec'].velo[tdict['spec'].sub_pix]
        idv = np.median(vel - np.roll(vel,1))
        # Save
        list_vel.append( tdict['spec'].velo[tdict['spec'].sub_pix] )
        list_fx.append( tdict['spec'].flux[tdict['spec'].sub_pix] )
        list_wgt.append( 1./ (np.ones(len(tdict['spec'].sub_pix))*np.sqrt(idv)) )
    all_velo = np.concatenate( list_vel )
    all_flux = np.concatenate( list_fx )
    all_weight = np.concatenate( list_wgt )

    # sort
    srt = np.argsort(all_velo)
    all_velo = all_velo[srt].value
    all_flux = all_flux[srt].value
    all_weight = all_weight[srt].value

    # ########
    # B-spline

    # Knots -- Must be well inside the data
    vmin = np.min(all_velo)
    vmax = np.max(all_velo)
    knots = np.arange(vmin+dv.value, vmax+10.-dv.value, dv.value) 

    # Check knots
    if debug:
        for knot in knots:
            gdi = np.where((all_velo>knot) & (all_velo<(knot+dv.value)))[0]
            if len(gdi) == 0:
                xdb.set_trace()
    tck = splrep( all_velo, all_flux, w=all_weight, k=3, t=knots)

    # Evaluate
    #xdb.set_trace()
    fin_flx = splev(fin_velo, tck, ext=1)

    # Return
    return fin_velo, fin_flx, all_dict

##
def stack_lnf(in_tup, min_fx=0.1, debug=False, **kwargs):
    '''Provided a table of b/g QSOs or a stack image,
      stack their spectra with ln f processing

    Parameters:
    ------------
    in_tup: Tuple
      (QPQ9 Table, wrest)
      (fin_velo, stck_img, stck_msk, all_dit)
    min_fx: float, optional
      Minimum normalized flux [0.1]

    Returns:
    ------------
    fin_velo
    fin_flx
    all_dict
    '''
    reload(qpqs)

    # Generate the image
    if len(in_tup) == 2: 
        fin_velo, stck_img, stck_msk, all_dict = load_stack_img(
            in_tup[0], in_tup[1], **kwargs)
    else:
        fin_velo, stck_img, stck_msk, all_dict = in_tup

    # Kuldge for nan
    stck_img = np.nan_to_num(stck_img)
    stck_img = np.minimum(stck_img,10.)
    stck_img = np.maximum(stck_img,min_fx) # Avoid singularities

    # log f
    stck_img = np.log(stck_img)

    # Simple average
    tot_flx = np.sum(stck_img*stck_msk,0)
    navg = np.sum(stck_msk,0)
    fin_flx = np.ones(len(fin_velo)) + tot_flx / np.maximum(navg, 1.)
    #xdb.set_trace()

    # Return
    return fin_velo, fin_flx, all_dict

##
def mk_kin_template(wrest,qpq=None,vmnx=[-2000.,2000.],
    all_dict=None):
    '''
    Generate an ASCII table of entries for generating
    the Kin template
    '''
    # QPQ structure
    if qpq is None:
        qpq = load_qpq(wrest)
    # Load stack image to establish those with good spectra
    if all_dict is None:
        fin_velo, stck_img, stck_msk, all_dict = load_stack_img(qpq,wrest)
    # Good idx
    msk = qpq == qpq
    for kk,idict in enumerate(all_dict):
        if idict is None:
            msk[kk] = False
    # Names 
    coords = SkyCoord(ra=qpq['RA']*u.deg, dec=qpq['DEC']*u.deg) #BG
    fgcoords = SkyCoord(ra=qpq['FG_RA']*u.deg, dec=qpq['FG_DEC']*u.deg) #BG
    fgnames = []
    bgnames = []
    for coord,fgcoord in zip(coords,fgcoords):
        bgnames.append('J{:s}{:s}'.format(coord.ra.to_string(unit=u.hour,pad=True,sep='',precision=2), 
            coord.dec.to_string(pad=True,alwayssign=True,sep='',precision=1)))
        fgnames.append('J{:s}{:s}'.format(fgcoord.ra.to_string(unit=u.hour,pad=True,sep='',precision=2), 
            fgcoord.dec.to_string(pad=True,alwayssign=True,sep='',precision=1)))

    # Create columns
    cwrest = Column([wrest.value]*len(qpq),name='wrest')
    cflg = Column([0]*len(qpq),name='flag')
    cvmin = Column([vmnx[0]]*len(qpq),name='VMIN')
    cvmax = Column([vmnx[1]]*len(qpq),name='VMAX')
    cfgname = Column(fgnames,name='FGQSON')
    cbgname = Column(bgnames,name='BGQSON')
    ccomment = Column(['Null                  -']*len(qpq),name='Comment')
    qpq.add_columns([cwrest,cflg,cvmin,cvmax,cfgname,cbgname,ccomment])
    qpq.rename_column('FG_ZIR','Z_FG')

    # Write table
    aline = AbsLine(wrest)
    outfil = 'kin_templ_{:s}{:d}.dat'.format(
        xai.ion_name(aline.data),int(wrest.value))
    #xdb.set_trace()
    qpq[msk]['FGQSON','Z_FG','BGQSON','wrest', 
        'flag','VMIN','VMAX','Comment'].write(
        outfil, delimiter='|',format='ascii.fixed_width', 
        formats={'VMIN': '%0.1f','VMAX': '%0.1f', 
        'Z_FG': '%0.6f'})
    print('Wrote {:s}'.format(outfil))


##

def setup_kin_gui(kinfil, outfil, qpq=None):
    '''Loads Kinematic driver file and launches a GUI 
    for fiddling about with the data and velocity ranges
    '''
    reload(qpqs)
    from xastropy.igm.abs_sys import abssys_utils as xiaa
    from xastropy.xguis import spec_guis as xxsg
    from xastropy.xguis import spec_widgets as xxsw
    from PyQt4 import QtGui
    from PyQt4 import QtCore

    # Line list for plotting
    lines = np.array([1215.6700, 1334.5323, 1526.7070, 
        1548.195, 1550.770, 1670.7874])*u.AA
    # Read kinfil
    kintab = Table.read(kinfil,format='ascii.fixed_width',
        delimiter='|')

    # Loop
    app = QtGui.QApplication(sys.argv)
    for ss,row in enumerate(kintab): 
        #if ss <= 1:
        #    continue
        # Update lines if necessary
        if row['wrest']*u.AA not in lines:
            lines.append(row['wrest']*u.AA)
            lines.sort()
        llist = xxgu.set_llist(lines)
        # Find spectrum for key line
        aspec_dict = qpqs.spec_qpq_wvobs(row['BGQSON'],
            lines*(1+row['Z_FG']), high_res=1,
            normalize=True)
        # Splice
        sv_files = []
        spec = None
        for spec_dict in aspec_dict:
            # Nothing
            if spec_dict is None:
                continue
            # 
            if spec is None:
                sv_files.append(spec_dict['file'])
                spec = spec_dict['spec']
            else:
                if spec_dict['file'] not in sv_files:
                    sv_files.append(spec_dict['file'])
                    # Splice
                    spec = spec.splice(spec_dict['spec'])
                    spec.filename = 'spliced'
        # Generate an abs_sys
        if spec is None:
            xdb.set_trace()
        abs_sys = xiaa.GenericAbsSystem(zabs=row['Z_FG'])
        abs_sys.lines = [AbsLine(row['wrest']*u.AA)]
        abs_sys.lines[0].analy['vlim'] = [row['VMIN'],row['VMAX']]*u.km/u.s
        abs_sys.lines[0].attrib['z'] = row['Z_FG']
        abs_sys.lines[0].analy['do_analysis'] = row['flag'] % 2
        # GUI
        print('Fiddle about with line {:g}'.format(row['wrest']))
        gui = xxsg.XVelPltGui(spec, abs_sys=abs_sys,
            llist=llist,vmnx=[-2000.,2000]*u.km/u.s)
        gui.exec_()
        # Fill up
        row['VMIN'] = gui.vplt_widg.abs_sys.lines[0].analy['vlim'][0].value
        row['VMAX'] = gui.vplt_widg.abs_sys.lines[0].analy['vlim'][1].value
        row['flag'] = gui.vplt_widg.abs_sys.lines[0].analy['do_analysis']
        row['flag'] += gui.vplt_widg.abs_sys.lines[0].analy['flg_eye']*2
        # Comment?
        tgui = xxsw.EnterTextGUI('Enter Comment (if any; 20char max):')
        tgui.exec_()
        if len(tgui.text) > 0:
            row['Comment'] = tgui.text

        # Write
        kintab.write(outfil, delimiter='|', format='ascii.fixed_width',
            formats={'VMIN': '%0.1f','VMAX': '%0.1f', 'Z_FG': '%0.6f'})

##
def qpq9_init():
    '''Dict of standard analysis values
    '''
    qpq9a_dict = dict(vmnx=(-3000.,3000.)*u.km/u.s)

    # Return
    return qpq9a_dict
    
# ##################################################
# ##################################################
# ##################################################
# Command line execution for testing
# ##################################################
if __name__ == '__main__':

    flg_test = 0
    #flg_test += 2**0  # Simple stack on Lya
    #flg_test += 2**1  # Simple stack on CII
    flg_test += 2**2   # Test GUI
    #flg_test += 2**3   # Make kin_templates

    if flg_test % 2**1 >= 2**0:
        qpq9 = QTable.read('qpq9_final.fits')
        #
        wrest = 1215.6701 * u.AA
        msk = cut_sample(qpq9,wrest,3)
        #
        velo, fx = stack_spec(qpq9[msk],wrest)#,debug=True)
        xdb.xplot(velo,fx, drawstyle='steps')
        
    if flg_test % 2**2 >= 2**1:
        qpq9 = QTable.read('qpq9_final.fits')
        #
        wrest = 1334.5323 * u.AA
        msk = cut_sample(qpq9,wrest,3)
        #
        velo, fx = stack_spec(qpq9[msk],wrest)#,debug=True)
        xdb.xplot(velo,fx, drawstyle='steps')
        #xdb.set_trace()

    if flg_test % 2**3 >= 2**2:
        setup_kin_gui('QPQ9_kin_driver.dat', 
            'QPQ9_kin_driver_v2.dat')
        
    if flg_test % 2**4 >= 2**3:
        mk_kin_template(1334.5323*u.AA)
        
