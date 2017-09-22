# imports
import copy, os, glob, imp, shutil
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Column, Table, QTable
from astropy.io import fits, ascii

from specdb.specdb import IgmSpec, SpecDB

from linetools import utils as ltu

from astropy.cosmology import FlatLambdaCDM

import enigma

cosmo = FlatLambdaCDM(H0=70, Om0=0.26)

# Load newly measured redshifts

zsys_line = ['H$\\alpha$','H$\\beta$','MgII','[OIII]']
zem_list = []
zem_coord = []
zem_fil = glob.glob(os.getenv('QPQ9')+'Analysis/Redshifts/zem/From_X_Joe/*')
for ff in zem_fil:
    zem = QTable(fits.open(ff)[1].data)
    if zem['ZSYS_FLAG'] not in [0,1,2,11]:
        continue
    zem_list.append(zem)
    ipos = zem['SPEC_FIL'][0].rfind('/SDSSJ')+6
    radec = zem['SPEC_FIL'][0][ipos:]
    ipos2 = radec.rfind('+')
    if ipos2 == -1:
        ipos2 = radec.rfind('-')
    ipos3 = radec.rfind('_F.fits')
    if ipos3 == -1:
        ipos3 = radec.rfind('.fits')
    RA = radec[0:2]+'h'+radec[2:4]+'m'+radec[4:ipos2]+'s'
    dec = radec[ipos2:ipos2+3]+'d'+radec[ipos2+3:ipos2+5]+'m'+radec[ipos2+5:ipos3]+'s'
    zem_coord.append(SkyCoord(RA,dec))
zem_fil = glob.glob(os.getenv('QPQ9')+'Analysis/Redshifts/zem/*_zem.fits')
for ff in zem_fil:
    zem = QTable(fits.open(ff)[1].data)
    if zem['ZSYS_FLAG'] not in [0,1,2,11]:
        continue
    zem_list.append(zem)
    ipos = zem['SPEC_FIL'][0].rfind('/SDSSJ')+6
    radec = zem['SPEC_FIL'][0][ipos:]
    RA = radec[0:2]+'h'+radec[2:4]+'m'+radec[4:9]+'s'
    dec = radec[9:12]+'d'+radec[12:14]+'m'+radec[14:18]+'s'
    zem_coord.append(SkyCoord(RA,dec))
zem_coord = SkyCoord(zem_coord)

# Make a new sample merging QPQ9_zIR and QPQ7

## Fix problems in the QPQ9_zIR structure
zIR_fil = 'qpq9_zIR.fits'
QPQ_zIR = QTable(fits.open(zIR_fil)[1].data)
QPQ9 = copy.deepcopy(QPQ_zIR)
QPQ9.rename_column('FG_ZIR', 'Z_FG')
QPQ9.rename_column('FG_SIG_ZIR','Z_FSIG')
for ii,qq in enumerate(QPQ9):
    try:
        assert 'J0225+0048' in qq['NAME'] # redshift is marked to be OII by mistake
        QPQ9.remove_row(ii)
    except AssertionError:
        pass
    try:
        assert 'J0913-0107' in qq['NAME'] # Hb redshift is not good, use MgII
        for zz,zem in enumerate(zem_list):
            if SkyCoord(qq['FG_RA']*u.deg,qq['FG_DEC']*u.deg).separation(zem_coord[zz]) < 1*u.arcsec:
                qq['Z_FG'] = zem['ZSYS_ZSYS'][0]
                qq['Z_FSIG'] = np.rint(zem['ZSYS_ERR'][0])
                qq['ZFG_LINE'] = zsys_line[np.min([zem['ZSYS_FLAG'][0],3])]
    except AssertionError:
        pass
    try:
        assert 'J1433+0641' in qq['NAME'] # redshift is Null in the zIR structure
        for zz,zem in enumerate(zem_list):
            if SkyCoord(qq['FG_RA']*u.deg,qq['FG_DEC']*u.deg).separation(zem_coord[zz]) < 1*u.arcsec:
                qq['Z_FG'] = zem['ZSYS_ZSYS'][0]
                qq['Z_FSIG'] = np.rint(zem['ZSYS_ERR'][0])
                qq['ZFG_LINE'] = zsys_line[np.min([zem['ZSYS_FLAG'][0],3])]
    except AssertionError:
        pass

## Add QPQ7 pairs with newly measured redshifts
QPQ7_fil = imp.find_module('enigma')[1] + '/data/qpq/qpq7_pairs.fits.gz'
QPQ7 = QTable(fits.open(QPQ7_fil)[1].data)
c_QPQzIR = SkyCoord(ra=QPQ_zIR['FG_RA']*u.deg, dec=QPQ_zIR['FG_DEC']*u.deg)
c_QPQ7 = SkyCoord(ra=QPQ7['RAD']*u.deg, dec=QPQ7['DECD']*u.deg)
c_QPQ7_bg = SkyCoord(ra=QPQ7['RAD_BG']*u.deg,dec=QPQ7['DECD_BG']*u.deg)
kpc_amin = cosmo.kpc_proper_per_arcmin(QPQ7['Z_FG'])
ang_sep = c_QPQ7.separation(c_QPQ7_bg).to('arcmin')
for qq,cc in enumerate(c_QPQ7):
    ## do not copy the entry with wrong spectral file
    wrong_c = SkyCoord(ra=8.59753707028004*u.deg,dec=-10.832307685613417*u.deg)
    if c_QPQ7_bg[qq].separation(wrong_c).to('arcsec') < 0.5*u.arcsec:
        continue
    if len(np.where(cc.separation(c_QPQzIR).to('arcsec') < 0.5*u.arcsec)[0]) == 0:
        if kpc_amin[qq]*ang_sep[qq] < 300*u.kpc:
            for zz,zem in enumerate(zem_list):
                if cc.separation(zem_coord[zz]) < 1*u.arcsec:
                    print(cc.to_string('hmsdms'),'added')
                    QPQ9.add_row({'FG_RA':QPQ7[qq]['RAD'],'FG_DEC':QPQ7[qq]['DECD'],
                                  'NAME':QPQ7[qq]['QSO_BG'],'RA':QPQ7[qq]['RAD_BG'],
                                  'DEC':QPQ7[qq]['DECD_BG'],'Z_FG':zem['ZSYS_ZSYS'][0],
                                  'Z_FSIG':np.rint(zem['ZSYS_ERR'][0]),'BG_Z':QPQ7[qq]['Z_BG'],
                                  'R_PHYS':QPQ7[qq]['R_PHYS'],
                                  'ZFG_LINE':zsys_line[np.min([zem['ZSYS_FLAG'][0],3])],
                                  'G_UV':QPQ7[qq]['G_UV']})
QPQ9.sort(['FG_RA','FG_DEC'])

## Add QPQ7 pairs with MgII redshifts
c_QPQ9 = SkyCoord(ra=QPQ9['FG_RA']*u.deg, dec=QPQ9['FG_DEC']*u.deg)
for qq,cc in enumerate(c_QPQ7):
    # do not copy the entry with wrong spectral file
    wrong_c = SkyCoord(ra=8.59753707028004*u.deg,dec=-10.832307685613417*u.deg)
    if c_QPQ7_bg[qq].separation(wrong_c).to('arcsec') < 0.5*u.arcsec:
        continue
    if len(np.where(cc.separation(c_QPQ9).to('arcsec') < 1*u.arcsec)[0]) == 0:
        if kpc_amin[qq]*ang_sep[qq] < 300*u.kpc:
            if QPQ7['Z_FSIG'][qq] < 300:
                QPQ9.add_row({'FG_RA':QPQ7['RAD'][qq],'FG_DEC':QPQ7['DECD'][qq],
                              'NAME':QPQ7['QSO_BG'][qq],'RA':QPQ7['RAD_BG'][qq],
                              'DEC':QPQ7['DECD_BG'][qq],'Z_FG':QPQ7['Z_FG'][qq],
                              'Z_FSIG':272,'BG_Z':QPQ7['Z_BG'][qq],
                              'R_PHYS':QPQ7['R_PHYS'][qq],'ZFG_LINE':'MgII',
                              'G_UV':QPQ7['G_UV'][qq]})

QPQ9.sort(['FG_RA','FG_DEC'])

## write
Table(QPQ9).write('qpq9_final.fits',format='fits',overwrite=True)

# Add pairs in QPQ database with newly measured redshifts

QPQ9 = QTable(fits.open('qpq9_final.fits')[1].data)
c_QPQ9 = SkyCoord(QPQ9['FG_RA']*u.deg,QPQ9['FG_DEC']*u.deg)
c_QPQ9_bg = SkyCoord(ra=QPQ9['RA']*u.deg,dec=QPQ9['DEC']*u.deg)
qpq_fil = '/Users/lwymarie/Documents/Databases/qpq_oir_spec.hdf5'
qpqsp = SpecDB(db_file=qpq_fil,verbose=False,idkey='QPQ_ID')
ID_fg,ID_bg = qpqsp.qcat.pairs(0.92*u.arcmin, 3000.*u.km/u.s)
c_qpqsp = SkyCoord(qpqsp.cat['RA'][ID_fg]*u.deg,qpqsp.cat['DEC'][ID_fg]*u.deg)
c_qpqsp_bg = SkyCoord(qpqsp.cat['RA'][ID_bg]*u.deg,qpqsp.cat['DEC'][ID_bg]*u.deg)
kpc_amin = cosmo.kpc_proper_per_arcmin(qpqsp.cat['zem'][ID_fg])
ang_sep = c_qpqsp.separation(c_qpqsp_bg).to('arcmin')
for qq,cc in enumerate(c_qpqsp):
    if kpc_amin[qq]*ang_sep[qq] < 300 *u.kpc:
        if len(np.where(cc.separation(c_QPQ9) < 0.5*u.arcsec)[0]) == 0: # pair not in QPQ9 already
            if len(np.where(cc.separation(zem_coord) < 1*u.arcsec)[0]) > 0:
                #J0225 has no good line measured. Skip.
                if cc.separation(SkyCoord('02h25m17.68s +00d48m22s')) < 1*u.arcsec:
                    continue
                #J0239 add [OIII]. Did not pass Lyb cut. Add back.
                #J1112 has good Hb, but currently it's MgII from QPQ7.
                #The redshift from QPQ7 or Hb here does not pass the > 3000 km/s requirement. Skip.
                if cc.separation(SkyCoord('11h12m42.69s +66d11m52.8s')) < 1*u.arcsec:
                    continue
                #J1215 add [OIII].
                #J2338 add [OIII].
                name = 'BOSSJ'
                name = name + c_qpqsp_bg[qq].to_string('hmsdms')[0:2] + c_qpqsp_bg[qq].to_string('hmsdms')[3:5]
                ipos = c_qpqsp_bg[qq].to_string('hmsdms').rfind(' ')+1
                name = name + c_qpqsp_bg[qq].to_string('hmsdms')[ipos:ipos+3] + \
                c_qpqsp_bg[qq].to_string('hmsdms')[ipos+4:ipos+6]
                index = np.where(zem_coord.separation(cc) < 1*u.arcsec)[0]
                if len(index) == 0:
                    print('no match found for',cc.to_string('hmsdms'),c_qpqsp_bg[qq].to_string('hmsdms'))
                    continue
                QPQ9.add_row({'FG_RA':qpqsp.cat[ID_fg[qq]]['RA'],'FG_DEC':qpqsp.cat[ID_fg[qq]]['DEC'],
                              'NAME':name,'RA':qpqsp.cat[ID_bg[qq]]['RA'],'DEC':qpqsp.cat[ID_bg[qq]]['DEC'],
                              'Z_FG':zem_list[index[0]]['ZSYS_ZSYS'][0],
                              'Z_FSIG':np.rint(zem_list[index[0]]['ZSYS_ERR'][0]),
                              'BG_Z':qpqsp.cat[ID_bg[qq]]['zem'],'R_PHYS':(kpc_amin[qq]*ang_sep[qq]).value,
                              'ZFG_LINE':zsys_line[np.min([zem_list[index[0]]['ZSYS_FLAG'][0],3])]})
                print(cc.to_string('hmsdms'),'added')
QPQ9.sort(['FG_RA','FG_DEC'])

## write
Table(QPQ9).write('qpq9_final.fits',format='fits',overwrite=True)

# Add pairs in igmspec with newly measured redshifts

QPQ9 = QTable(fits.open('qpq9_final.fits')[1].data)
c_QPQ9 = SkyCoord(QPQ9['FG_RA']*u.deg,QPQ9['FG_DEC']*u.deg)
c_QPQ9_bg = SkyCoord(ra=QPQ9['RA']*u.deg,dec=QPQ9['DEC']*u.deg)
igmsp = IgmSpec(version="02.1",verbose=False,
                groups=['BOSS_DR12','COS-Dwarfs','COS-Halos',
                        'ESI_DLA','GGG','HD-LLS_DR1',
                        'HDLA100','HSTQSO','HST_z2',
                        'KODIAQ_DR1','SDSS_DR7','XQ-100',
                        'UVES_Dall','UVpSM4','MUSoDLA'])
ID_fg,ID_bg = igmsp.qcat.pairs(0.92*u.arcmin,3000.*u.km/u.s) # at z = 0.4, 0.92' = 300 kpc
c_igmsp = SkyCoord(igmsp.cat['RA'][ID_fg]*u.deg,igmsp.cat['DEC'][ID_fg]*u.deg)
c_igmsp_bg = SkyCoord(igmsp.cat['RA'][ID_bg]*u.deg,igmsp.cat['DEC'][ID_bg]*u.deg)
kpc_amin = cosmo.kpc_proper_per_arcmin(igmsp.cat['zem'][ID_fg])
ang_sep = c_igmsp.separation(c_igmsp_bg).to('arcmin')
for qq,cc in enumerate(c_igmsp):
    if kpc_amin[qq]*ang_sep[qq] < 300 *u.kpc:
        if len(np.where(cc.separation(c_QPQ9) < 0.5*u.arcsec)[0]) == 0: # pair not in QPQ9 already
            if len(np.where(cc.separation(zem_coord) < 1*u.arcsec)[0]) > 0:
                # Skip J0225. No good line measured.
                if cc.separation(SkyCoord('02h25m17.68s +00d48m22s')) < 5*u.arcsec:
                    continue
                # Skip J0201. Binary quasar, with wrong redshift in catalog.
                if cc.separation(SkyCoord('02h01m43.4873s +00d32m22.713s')) < 5*u.arcsec:
                    continue
                name = 'SDSSJ'
                name = name + c_igmsp_bg[qq].to_string('hmsdms')[0:2] + c_igmsp_bg[qq].to_string('hmsdms')[3:5]
                ipos = c_igmsp_bg[qq].to_string('hmsdms').rfind(' ')+1
                name = name + c_igmsp_bg[qq].to_string('hmsdms')[ipos:ipos+3] + \
                c_igmsp_bg[qq].to_string('hmsdms')[ipos+4:ipos+6]
                index = np.where(zem_coord.separation(cc) < 1*u.arcsec)[0]
                QPQ9.add_row({'FG_RA':igmsp.cat[ID_fg[qq]]['RA'],'FG_DEC':igmsp.cat[ID_fg[qq]]['DEC'],
                              'NAME':name,'RA':igmsp.cat[ID_bg[qq]]['RA'],'DEC':igmsp.cat[ID_bg[qq]]['DEC'],
                              'Z_FG':zem_list[index[0]]['ZSYS_ZSYS'][0],
                              'Z_FSIG':np.rint(zem_list[index[0]]['ZSYS_ERR'][0]),'BG_Z':igmsp.cat[ID_bg[qq]]['zem'],
                              'R_PHYS':(kpc_amin[qq]*ang_sep[qq]).value,
                              'ZFG_LINE':zsys_line[np.min([zem_list[index[0]]['ZSYS_FLAG'][0],3])]})
                print(cc.to_string('hmsdms'),'added')
QPQ9.sort(['FG_RA','FG_DEC'])

# write
Table(QPQ9).write('qpq9_final.fits',format='fits',overwrite=True)

# Add and update g_UV values

## Search SDSS DR12 catalog for magnitudes
QPQ9 = QTable(fits.open('qpq9_final.fits')[1].data)
c_QPQ9 = SkyCoord(QPQ9['FG_RA']*u.deg,QPQ9['FG_DEC']*u.deg)
c_QPQ9_bg = SkyCoord(ra=QPQ9['RA']*u.deg,dec=QPQ9['DEC']*u.deg)
hdulist = fits.open('/Volumes/Data/Data of surveys/SDSS/DR12Q.fits')
SDSSdr12 = hdulist[1].data
c_SDSSdr12 = SkyCoord(list(SDSSdr12.field('RA'))*u.deg,list(SDSSdr12.field('DEC'))*u.deg)
hdulist = fits.open('/Volumes/Data/Data of surveys/SDSS/dr7qso.fit')
SDSSdr7 = hdulist[1].data
c_SDSSdr7 = SkyCoord(list(SDSSdr7.field('RA'))*u.deg,list(SDSSdr7.field('DEC'))*u.deg)

## Put in g_UV by hand for a quasar that does not have SDSS spectra but photometry and no existing g_UV values
for qq,cc in enumerate(c_QPQ9):
    if QPQ9[qq]['G_UV'] == 0.:
        if len(np.where(cc.separation(c_SDSSdr12) < 0.5*u.arcsec)[0]) == 0:
            if len(np.where(cc.separation(c_SDSSdr7) < 0.5*u.arcsec)[0]) == 0:
                ang_sep = cc.separation(c_QPQ9_bg[qq]).to('arcsec')
                print(cc.to_string('hmsdms'),QPQ9[qq]['Z_FG'],ang_sep,qq)
                QPQ9[qq]['G_UV'] = 4624

## Write a file for IDL input, for rescaling QPQ g_UV values to F.-G. Lya forest flux
gUV_exist = np.where(QPQ9['G_UV'] != 0.)[0]
data = Table([QPQ9[gUV_exist]['FG_RA'],QPQ9[gUV_exist]['FG_DEC'],QPQ9[gUV_exist]['Z_FG'],QPQ9[gUV_exist]['G_UV']],
             names=['FG_RA','FG_DEC','Z_FG','OLD_GUV'])
ascii.write(data,'QPQ_gUV_idl_input.txt',overwrite=True)
# Write a file for IDL input, for calculating new g_UV values
gUV_none = np.where(QPQ9['G_UV'] == 0.)[0]
ang_sep = []
SDSSfilter = []
mag = []
filter_beg = [3125.0, 3880.0, 5480.0, 6790.0, 8090.0]
filter_names = ['u','g','r','i','z']
for qq,cc in enumerate(c_QPQ9[gUV_none]):
    ang_sep.append(cc.separation(c_QPQ9_bg[gUV_none[qq]]).to('arcsec').value)
    min_beg = 1215.6701*(1+QPQ9[gUV_none[qq]]['Z_FG'])
    good_inds = np.where(filter_beg >= min_beg)
    SDSSfilter.append(filter_names[np.min(good_inds)])
    in_dr12 = np.where(cc.separation(c_SDSSdr12) < 0.5*u.arcsec)[0]
    if len(in_dr12) == 1:
        mag.append((SDSSdr12[in_dr12].field('PSFMAG')[0][good_inds]-
                    SDSSdr12[in_dr12].field('EXTINCTION')[0][good_inds])[0])
    else:
        in_dr7 = np.where(cc.separation(c_SDSSdr7) < 0.5*u.arcsec)[0]
        psfmag = [SDSSdr7[in_dr7].field('UMAG'),SDSSdr7[in_dr7].field('GMAG'),SDSSdr7[in_dr7].field('RMAG'),
                  SDSSdr7[in_dr7].field('IMAG'),SDSSdr7[in_dr7].field('ZMAG')]
        extinct = SDSSdr7[in_dr7].field('AU')*np.array([1,0.736,0.534,0.405,0.287])
        mag.append((psfmag[np.min(good_inds)]-extinct[np.min(good_inds)])[0])
data = Table([QPQ9[gUV_none]['FG_RA'],QPQ9[gUV_none]['FG_DEC'],QPQ9[gUV_none]['Z_FG'],ang_sep,SDSSfilter,mag],
             names=['FG_RA','FG_DEC','Z_FG','theta','filter','mag'])
ascii.write(data,'no_gUV_idl_input.txt',overwrite=True)

## Go to run IDL

## Read IDL output for rescaled QPQ g_UV values
data = ascii.read('QPQ_gUV_idl_output.txt')
c_idl = SkyCoord(data['FG_RA']*u.deg,data['FG_DEC']*u.deg)
for qq,cc in enumerate(c_QPQ9):
    index = np.where(cc.separation(c_idl) < 0.5*u.arcsec)[0]
    if len(index) == 1:
        QPQ9[qq]['G_UV'] = data[index]['NEW_GUV']
## Read IDL output for new g_UV values
data = ascii.read('no_gUV_idl_output.txt')
c_idl = SkyCoord(data['FG_RA']*u.deg,data['FG_DEC']*u.deg)
for qq,cc in enumerate(c_QPQ9):
    index = np.where(cc.separation(c_idl) < 0.5*u.arcsec)[0]
    if len(index) == 1:
        QPQ9[qq]['G_UV'] = data[index]['GUV']

## write
Table(QPQ9).write('qpq9_final.fits',format='fits',overwrite=True)

# Update systematic offsets and scatters of emission lines to Shen+16

## Add the systematic biases, intrinsic scatter + measurement scatter
QPQ9 = QTable(fits.open('qpq9_final.fits')[1].data)
for ii,qq in enumerate(QPQ9):
    if qq['ZFG_LINE'] == 'MgII':
        # bring to z_em
        qq['Z_FG'] = ltu.z_from_dv(+70.*u.km/u.s,qq['Z_FG'])
        # bring to z_sys
        qq['Z_FG'] = ltu.z_from_dv(+57.*u.km/u.s,qq['Z_FG'])
        # scatter
        qq['Z_FSIG'] = 226
    if qq['ZFG_LINE'] == 'H$\\beta$':
        # bring to z_sys
        qq['Z_FG'] = ltu.z_from_dv(+109.*u.km/u.s,qq['Z_FG'])
        # scatter
        qq['Z_FSIG'] = 417
    if qq['ZFG_LINE'] == '[OIII]':
        # bring to z_em
        qq['Z_FG'] = ltu.z_from_dv(-27.*u.km/u.s,qq['Z_FG'])
        # bring to z_sys
        qq['Z_FG'] = ltu.z_from_dv(+48.*u.km/u.s,qq['Z_FG'])
        # scatter
        qq['Z_FSIG'] = 68
    if qq['ZFG_LINE'] == 'H$\\alpha$':
        # scatter
        qq['Z_FSIG'] = 300

## write
Table(QPQ9).write('qpq9_final.fits',format='fits',overwrite=True)

# copy to enigma
shutil.copy('qpq9_final.fits',enigma.__path__[0]+'/data/qpq/qpq9_final.fits')