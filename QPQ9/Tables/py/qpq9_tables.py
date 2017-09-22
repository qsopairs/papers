# imports
from __future__ import print_function,absolute_import,division,unicode_literals
import numpy as np
import glob,os,sys,copy,imp
import matplotlib as mpl
mpl.rcParams['font.family']='stixgeneral'
from astropy.table import QTable,Table
from astropy.io import ascii,fits
from astropy import units as u
from astropy.coordinates import SkyCoord
sys.path.append(os.path.abspath("../../../py"))
sys.path.append(os.path.abspath("../Analysis/Stacks/py"))
import qpq9_stacks as qpq9k
import enigma

# Load QPQ9 structure
path = enigma.__path__[0]+'/data/qpq/'
QPQ9 = QTable(fits.open(path+'qpq9_final.fits')[1].data)

## Make the table that lists the full QPQ9 sample

# Load the dictionaries of quasars used in stack for CII
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                               vsig_cut=400*u.km/u.s,plot_indiv=False)
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

# Save the properties of pairs used in stack for CII
sv_fg_qso = []
sv_zfg = []
sv_line = []
sv_bg_qso = []
sv_zbg = []
sv_bg_instr = []
sv_Rphys = []
sv_gUV = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    # fg qso
    coord = SkyCoord(idict['qpq']['FG_RA']*u.deg,idict['qpq']['FG_DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    if '-' in dec_str:
        dec_fmt = '$-$' + dec_str[1:8]+str(np.around(float(dec_str),decimals=1))[-1]
    else:
        dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    sv_fg_qso.append(coord_fmt)
    # z_fg
    zfg = str(np.around(idict['qpq']['Z_FG'],decimals=4))
    sv_zfg.append(zfg)
    # line analyzed for z_fg
    line = idict['qpq']['ZFG_LINE']
    sv_line.append(line)
    # bg qso
    coord = SkyCoord(idict['qpq']['RA']*u.deg,idict['qpq']['DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    sv_bg_qso.append(coord_fmt)
    # z_bg
    zbg = str(np.around(idict['qpq']['BG_Z'],decimals=3))
    sv_zbg.append(zbg)
    # bg qso instr
    bg_instr = idict['instr']
    if bg_instr == 'LRISb':
        bg_instr = 'LRIS'
    if bg_instr == 'MIKE-Blue':
        bg_instr = 'MIKE'
    if bg_instr == 'MODS1B':
        bg_instr = 'MODS1'
    sv_bg_instr.append(bg_instr)
    # R_perp
    R_phys = str(np.int(np.rint(idict['qpq']['R_PHYS'])))
    sv_Rphys.append(R_phys)
    # g_UV
    g_UV = str(np.int(np.rint(idict['qpq']['G_UV'])))
    sv_gUV.append(g_UV)

# Load the dictionaries of quasars used in stack for CIV
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1548.195*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                              vsig_cut=400*u.km/u.s,plot_indiv=False)
# Mask
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J1002+0020' in idict['qpq']['NAME']: #CIV overlaps with BAL of background quasar
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.

# Save the properties
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    # fg qso
    coord = SkyCoord(idict['qpq']['FG_RA']*u.deg,idict['qpq']['FG_DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    if '-' in dec_str:
        dec_fmt = '$-$' + dec_str[1:8]+str(np.around(float(dec_str),decimals=1))[-1]
    else:
        dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    if coord_fmt in sv_fg_qso: # already saved for CII stack
        continue
    sv_fg_qso.append(coord_fmt)
    # z_fg
    zfg = str(np.around(idict['qpq']['Z_FG'],decimals=4))
    sv_zfg.append(zfg)
    # line analyzed for z_fg
    line = idict['qpq']['ZFG_LINE']
    sv_line.append(line)
    # bg qso
    coord = SkyCoord(idict['qpq']['RA']*u.deg,idict['qpq']['DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    sv_bg_qso.append(coord_fmt)
    # z_bg
    zbg = str(np.around(idict['qpq']['BG_Z'],decimals=3))
    sv_zbg.append(zbg)
    # bg qso instr
    bg_instr = idict['instr']
    if bg_instr == 'LRISb':
        bg_instr = 'LRIS'
    if bg_instr == 'MIKE-Blue':
        bg_instr = 'MIKE'
    if bg_instr == 'MODS1B':
        bg_instr = 'MODS1'
    sv_bg_instr.append(bg_instr)
    # R_perp
    R_phys = str(np.int(np.rint(idict['qpq']['R_PHYS'])))
    sv_Rphys.append(R_phys)
    # g_UV
    g_UV = str(np.int(np.rint(idict['qpq']['G_UV'])))
    sv_gUV.append(g_UV)

# Load the dictionaries of quasars used in stack for MgII
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=2796.354*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                              vsig_cut=400*u.km/u.s,plot_indiv=False)
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J0822+1319' in idict['qpq']['NAME']: #BAL
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.

# Save the properties
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    # fg qso
    coord = SkyCoord(idict['qpq']['FG_RA']*u.deg,idict['qpq']['FG_DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    if '-' in dec_str:
        dec_fmt = '$-$' + dec_str[1:8]+str(np.around(float(dec_str),decimals=1))[-1]
    else:
        dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    if coord_fmt in sv_fg_qso: # already saved for CII, CIV stack
        continue
    sv_fg_qso.append(coord_fmt)
    # z_fg
    zfg = str(np.around(idict['qpq']['Z_FG'],decimals=4))
    sv_zfg.append(zfg)
    # line analyzed for z_fg
    line = idict['qpq']['ZFG_LINE']
    sv_line.append(line)
    # bg qso
    coord = SkyCoord(idict['qpq']['RA']*u.deg,idict['qpq']['DEC']*u.deg)
    RA_str = coord.ra.to_string(unit=u.hour,sep='',pad=True)
    RA_fmt = RA_str[0:8]+str(np.around(float(RA_str),decimals=2))[-1]
    dec_str = coord.dec.to_string(unit=u.deg,sep='',pad=True,alwayssign=True)
    dec_fmt = dec_str[0:8]+str(np.around(float(dec_str),decimals=1))[-1]
    coord_fmt = 'J'+RA_fmt+dec_fmt
    sv_bg_qso.append(coord_fmt)
    # z_bg
    zbg = str(np.around(idict['qpq']['BG_Z'],decimals=3))
    sv_zbg.append(zbg)
    # bg qso instr
    bg_instr = idict['instr']
    if bg_instr == 'LRISb':
        bg_instr = 'LRIS'
    if bg_instr == 'MIKE-Blue':
        bg_instr = 'MIKE'
    if bg_instr == 'MODS1B':
        bg_instr = 'MODS1'
    sv_bg_instr.append(bg_instr)
    # R_perp
    R_phys = str(np.int(np.rint(idict['qpq']['R_PHYS'])))
    sv_Rphys.append(R_phys)
    # g_UV
    g_UV = str(np.int(np.rint(idict['qpq']['G_UV'])))
    sv_gUV.append(g_UV)

tab = Table([sv_fg_qso,sv_zfg,sv_line,sv_bg_qso,sv_zbg,sv_bg_instr,sv_Rphys,sv_gUV],
            names=['fg_qso','zfg','line','bg_qso','zbg','bg_instr','Rphys','gUV'])
tab.sort('fg_qso')

# Print to a tex file
afile = open('tab_sample.tex','w')
afile.write('\\LongTables\n')
afile.write('\\begin{deluxetable*}{ccccccccc}\n')
afile.write('\\tablewidth{0pc}\n')
afile.write('\\tablecaption{Properties of the Projected Quasar Pairs in the QPQ9 sample')
afile.write('\\label{tab:sample}}\n')
afile.write('\\tabletypesize{\\scriptsize}\n')
afile.write('\\setlength{\\tabcolsep}{0in}\n')
afile.write('\\tablehead{\\colhead{Foreground Quasar} & \\colhead{$z_{\\rm fg}$} & \n')
afile.write('\\colhead{Line for $z_{\\rm fg}^a$} & \\colhead{Background Quasar} & \n')
afile.write('\\colhead{$z_{\\rm bg}$} & \\colhead{BG Quasar Instrument} & \n')
afile.write('\\colhead{$R_\\perp$ (kpc)} & \\colhead{$g_{\\rm UV}$}} \n')
afile.write('\\startdata \n')
for ii in np.arange(len(tab)):
    afile.write(tab['fg_qso'][ii] + ' & ' + tab['zfg'][ii] + ' & ' + tab['line'][ii] + ' & ')
    afile.write(tab['bg_qso'][ii] + ' & ' + tab['zbg'][ii] + ' & ' + tab['bg_instr'][ii] + ' & ')
    afile.write(tab['Rphys'][ii] + ' & ' + tab['gUV'][ii] + ' \\\ \n')
afile.write('\\enddata \n')
afile.write('\\tablenotetext{a}{The emission-line analyzed for measuring $z_{\\rm fg}$.} \n')
afile.write('\\end{deluxetable*}')
afile.close()

## Make the table of QPQ9 summary

# Properties of the CII sample
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                               vsig_cut=400*u.km/u.s,plot_indiv=False)
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if 'J1508+3635' in dd['qpq']['NAME']: #DLA not excluded by forest cut, should be excluded now
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_CII = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_CII = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_CII = str(int(np.around(np.median(sv_Rperp))))

# Properties of the CIV sample
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1548.195*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                              vsig_cut=400*u.km/u.s,plot_indiv=False)
# Mask
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J1002+0020' in idict['qpq']['NAME']: #CIV overlaps with BAL of background quasar
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_CIV = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_CIV = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_CIV = str(int(np.around(np.median(sv_Rperp))))

# Properties of the MgII sample
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=2796.354*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                              vsig_cut=400*u.km/u.s,plot_indiv=False)
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J0822+1319' in idict['qpq']['NAME']: #BAL
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_MgII = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_MgII = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_MgII = str(int(np.around(np.median(sv_Rperp))))

# CII centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CII_MgII_mean_fit.dat'))[0]
cenofmean_CII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_CII = '+'+cenofmean_CII
dispofmean_CII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_1334_mean.fits')
scatterincenofmean_CII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_CII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CII_MgII_median_fit.dat'))[0]
cenofmedian_CII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_CII = '+'+cenofmedian_CII
dispofmedian_CII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_1334_med.fits')
scatterincenofmedian_CII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_CII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# CIV centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CIV_MgII_mean_fit.dat'))[0]
cenofmean_CIV = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_CIV = '+'+cenofmean_CIV
dispofmean_CIV = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_1548_mean.fits')
scatterincenofmean_CIV = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_CIV = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CIV_MgII_median_fit.dat'))[0]
cenofmedian_CIV = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_CIV = '+'+cenofmedian_CIV
dispofmedian_CIV = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_1548_med.fits')
scatterincenofmedian_CIV = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_CIV = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# MgII centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'MgII_MgII_mean_fit.dat'))[0]
cenofmean_MgII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_MgII = '+'+cenofmean_MgII
dispofmean_MgII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_2796_mean.fits')
scatterincenofmean_MgII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_MgII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'MgII_MgII_median_fit.dat'))[0]
cenofmedian_MgII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_MgII = '+'+cenofmedian_MgII
dispofmedian_MgII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'IRMgII_2796_med.fits')
scatterincenofmedian_MgII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_MgII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# Print to a tex file
afile = open('tab_summary.tex','w')
afile.write('\\begin{deluxetable*}{lccc} \n')
afile.write('\\tablewidth{0pc} \n')
afile.write('\\tablecaption{Summary of the Data and Analysis\label{tab:summary}} \n')
afile.write('\\tabletypesize{\\small} \n')
afile.write('\\tablehead{\\colhead{Measure} & \\colhead{\\ion{C}{2}\\,1334} & \\colhead{\\ion{C}{4}\\,1548} \n')
afile.write('& \\colhead{\\ion{Mg}{2}\\,2796}} \n')
afile.write('\\startdata \n')
afile.write('\\cutinhead{For the Full QPQ9 Sample} \n')
afile.write('Number of pairs & '+npairs_CII+' & '+npairs_CIV+' & '+npairs_MgII+' \\\ \n')
afile.write('Median $z_{\\rm fg}$ & '+medzfg_CII+' & '+medzfg_CIV+' & '+medzfg_MgII+' \\\ \n')
afile.write('Median $R_\\perp$ & '+medRperp_CII+' & '+medRperp_CIV+' & '+medRperp_MgII+' \\\ \n')
afile.write('Centroid of mean stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+cenofmean_CII+'\\pm'+scatterincenofmean_CII+'$'+' & '
            +'$'+cenofmean_CIV+'\\pm'+scatterincenofmean_CIV+'$'+' & '
            +'$'+cenofmean_MgII+'\\pm'+scatterincenofmean_MgII+'$'+' \\\ \n')
afile.write('1$\sigma$ dispersion of mean stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+dispofmean_CII+'\\pm'+scatterindispofmean_CII+'$'+' & '
            +'$'+dispofmean_CIV+'\\pm'+scatterindispofmean_CIV+'$'+' & '
            +'$'+dispofmean_MgII+'\\pm'+scatterindispofmean_MgII+'$'+' \\\ \n')
afile.write('Centroid of median stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+cenofmedian_CII+'\\pm'+scatterincenofmedian_CII+'$'+' & '
            +'$'+cenofmedian_CIV+'\\pm'+scatterincenofmedian_CIV+'$'+' & '
            +'$'+cenofmedian_MgII+'\\pm'+scatterincenofmedian_MgII+'$'+' \\\ \n')
afile.write('1$\sigma$ dispersion of median stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+dispofmedian_CII+'\\pm'+scatterindispofmedian_CII+'$'+' & '
            +'$'+dispofmedian_CIV+'\\pm'+scatterindispofmedian_CIV+'$'+' & '
            +'$'+dispofmedian_MgII+'\\pm'+scatterindispofmedian_MgII+'$'+' \\\ \n')
afile.close()

# Properties of the CII sample with [OIII] redshifts
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                               vsig_cut=70*u.km/u.s,plot_indiv=False)
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if 'J1508+3635' in dd['qpq']['NAME']: #DLA not excluded by forest cut, should be excluded now
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_CII = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_CII = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_CII = str(int(np.around(np.median(sv_Rperp))))

# Properties of the CIV sub-sample with [OIII] redshifts
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1548.195*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                               vsig_cut=70*u.km/u.s,plot_indiv=False)
# Mask
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J1002+0020' in idict['qpq']['NAME']: #CIV overlaps with BAL of background quasar
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_CIV = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_CIV = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_CIV = str(int(np.around(np.median(sv_Rperp))))

# Properties of the MgII sub-sample with [OIII] redshifts
stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=2796.354*u.AA,S2N_cut=5.5/u.AA,zfg_mnx=(1.6,9999),
                               vsig_cut=70*u.km/u.s,plot_indiv=False)
fin_velo, stck_img, stck_msk, all_dict = stack_tup0
stck_mskN = copy.deepcopy(stck_msk)
idx_mask = []
for ii,idict in enumerate(all_dict):
    if idict is None:
        continue
    if 'J0822+1319' in idict['qpq']['NAME']: #BAL
        idx_mask.append(ii)
for idx in idx_mask:
    stck_mskN[idx,:] = 0.
npairs_MgII = str(int(np.max((np.sum(stck_mskN,0)))))
sv_zfg = []
sv_Rperp = []
for ii,dd in enumerate(all_dict):
    if dd is None:
        continue
    if np.sum(stck_mskN[ii,:]) == 0:
        continue
    sv_zfg.append(dd['qpq']['Z_FG'])
    sv_Rperp.append(dd['qpq']['R_PHYS'])
medzfg_MgII = str(np.around(np.median(sv_zfg),decimals=2))
medRperp_MgII = str(int(np.around(np.median(sv_Rperp))))

# CII centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CII_OIII_mean_fit.dat'))[0]
cenofmean_CII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_CII = '+'+cenofmean_CII
dispofmean_CII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_1334_mean.fits')
scatterincenofmean_CII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_CII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CII_OIII_median_fit.dat'))[0]
cenofmedian_CII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_CII = '+'+cenofmedian_CII
dispofmedian_CII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_1334_med.fits')
scatterincenofmedian_CII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_CII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# CIV centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CIV_OIII_mean_fit.dat'))[0]
cenofmean_CIV = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_CIV = '+'+cenofmean_CIV
dispofmean_CIV = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_1548_mean.fits')
scatterincenofmean_CIV = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_CIV = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'CIV_OIII_median_fit.dat'))[0]
cenofmedian_CIV = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_CIV = '+'+cenofmedian_CIV
dispofmedian_CIV = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_1548_med.fits')
scatterincenofmedian_CIV = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_CIV = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# MgII centroid and dispersion of stacks
path = '../Analysis/Stacks/'
params = (ascii.read(path+'MgII_OIII_mean_fit.dat'))[0]
cenofmean_MgII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmean_MgII = '+'+cenofmean_MgII
dispofmean_MgII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_2796_mean.fits')
scatterincenofmean_MgII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmean_MgII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()
path = '../Analysis/Stacks/'
params = (ascii.read(path+'MgII_OIII_median_fit.dat'))[0]
cenofmedian_MgII = str(int(np.around(params['mean_1'])))
if params['mean_1'] > 0:
    cenofmedian_MgII = '+'+cenofmedian_MgII
dispofmedian_MgII = str(int(np.around(params['stddev_1'])))
path = '../Analysis/Bootstrap/Output/'
boot_hdulist = fits.open(path+'OIII_2796_med.fits')
scatterincenofmedian_MgII = str(int(np.around(np.std(boot_hdulist[3].data))))
scatterindispofmedian_MgII = str(int(np.around(np.std(boot_hdulist[4].data))))
boot_hdulist.close()

# Continue to print to the tex file
afile = open('tab_summary.tex','a')
afile.write('\\cutinhead{For the Sub-sample with [OIII] Redshifts} \n')
afile.write('Number of pairs & '+npairs_CII+' & '+npairs_CIV+' & '+npairs_MgII+' \\\ \n')
afile.write('Median $z_{\\rm fg}$ & '+medzfg_CII+' & '+medzfg_CIV+' & '+medzfg_MgII+' \\\ \n')
afile.write('Median $R_\\perp$ & '+medRperp_CII+' & '+medRperp_CIV+' & '+medRperp_MgII+' \\\ \n')
afile.write('Centroid of mean stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+cenofmean_CII+'\\pm'+scatterincenofmean_CII+'$'+' & '
            +'$'+cenofmean_CIV+'\\pm'+scatterincenofmean_CIV+'$'+' & '
            +'$'+cenofmean_MgII+'\\pm'+scatterincenofmean_MgII+'$'+' \\\ \n')
afile.write('1$\sigma$ dispersion of mean stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+dispofmean_CII+'\\pm'+scatterindispofmean_CII+'$'+' & '
            +'$'+dispofmean_CIV+'\\pm'+scatterindispofmean_CIV+'$'+' & '
            +'$'+dispofmean_MgII+'\\pm'+scatterindispofmean_MgII+'$'+' \\\ \n')
afile.write('Centroid of median stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+cenofmedian_CII+'\\pm'+scatterincenofmedian_CII+'$'+' & '
            +'$'+cenofmedian_CIV+'\\pm'+scatterincenofmedian_CIV+'$'+' & '
            +'$'+cenofmedian_MgII+'\\pm'+scatterincenofmedian_MgII+'$'+' \\\ \n')
afile.write('1$\sigma$ dispersion of median stack (${\\rm km\\,s^{-1}}$) & '
            +'$'+dispofmedian_CII+'\\pm'+scatterindispofmedian_CII+'$'+' & '
            +'$'+dispofmedian_CIV+'\\pm'+scatterindispofmedian_CIV+'$'+' & '
            +'$'+dispofmedian_MgII+'\\pm'+scatterindispofmedian_MgII+'$'+' \\\ \n')
afile.write('\\enddata \n')
afile.write('\\end{deluxetable*}')
afile.close()