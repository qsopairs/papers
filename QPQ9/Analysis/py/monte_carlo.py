from __future__ import print_function,absolute_import,division,unicode_literals
import numpy as np
import glob,os,sys,copy,imp
from scipy import stats as scistats
import matplotlib as mpl
mpl.rcParams['font.family']='stixgeneral'
from matplotlib import pyplot as plt
from astropy.table import QTable,Table
from astropy.io import ascii,fits
from astropy import units as u
import linetools.utils as ltu
sys.path.append(os.path.abspath("./Stacks/py"))
import qpq9_stacks as qpq9k
import scipy.integrate as integrate
from astropy.modeling import models,fitting
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70,Om0=0.26)


# the constants
n_trials = 1000
Deltav = 10.
v_grid = -2000. + np.arange((2000.-(-2000.))/Deltav+1)*Deltav
relativistic_equiv = u.doppler_relativistic(1334.5323*u.AA)
wave_grid = (v_grid*u.km/u.s).to(u.AA,equivalencies=relativistic_equiv)

# Generate a stack image of n_trials monte carlo trials
def gen_stck_img(R_phys,z_fg,stddev_oneabs,sigma1D): # R_phys in kpc, Deltav in km/s
    probs = np.zeros(len(v_grid))
    cumulative_probs = np.zeros(len(v_grid))
    Hubble_h = cosmo.H(0).value/100.
    ell_IGM_DLA = 0.2*((1+z_fg)/(1+2.5))**2.1
    ell_IGM_SLLS = 0.44*((1+z_fg)/(1+2.5))**2.1
    ell_IGM_LLS = 1.05*((1+z_fg)/(1+2.5))**2.1
    R_comov = R_phys/1000.*(1+z_fg)*Hubble_h # in h^-1 Mpc
    for ii,vv in enumerate(v_grid):
        gamma_DLA = 1.6
        r0_DLA = 3.9
#        gamma_SLLS = 1.6
#        r0_SLLS = 15.5
#        gamma_LLS = 1.6
#        r0_LLS = 13.9
        gamma_SLLS = 1.68
        r0_SLLS = 14.0
        gamma_LLS = 1.68
        r0_LLS = 12.5
        chi_DLA = 1./Deltav*integrate.quad(
            lambda v:(np.sqrt(R_comov**2+(v/(cosmo.H(z_fg).value/(1+z_fg)))**2)/r0_DLA)**(-gamma_DLA),
            v_grid[ii]-Deltav/2,v_grid[ii]+Deltav/2)[0]
        chi_SLLS = 1./Deltav*integrate.quad(
            lambda v:(np.sqrt(R_comov**2+(v/(cosmo.H(z_fg).value/(1+z_fg)))**2)/r0_SLLS)**(-gamma_SLLS),
            v_grid[ii]-Deltav/2,v_grid[ii]+Deltav/2)[0]
        chi_LLS = 1./Deltav*integrate.quad(
            lambda v:(np.sqrt(R_comov**2+(v/(cosmo.H(z_fg).value/(1+z_fg)))**2)/r0_LLS)**(-gamma_LLS),
            v_grid[ii]-Deltav/2,v_grid[ii]+Deltav/2)[0]
        Deltaz = ltu.z_from_dv((v_grid[ii]+Deltav/2)*u.km/u.s,z_fg)-ltu.z_from_dv((v_grid[ii]-Deltav/2)*u.km/u.s,z_fg)
        prob = (ell_IGM_DLA*(1+chi_DLA)+ell_IGM_SLLS*(1+chi_SLLS)+ell_IGM_LLS*(1+chi_LLS))*Deltaz
        # test effect of doubling number of absorbers
#        prob = prob*2
        probs[ii] = prob
        cumulative_probs[ii] = prob + np.sum(probs[0:ii])
    norm = np.sum(probs)
    probs = probs/norm
#    print('Expected number of absorbers at R_phys',norm,R_phys)
    stck_img = np.zeros((n_trials,len(v_grid)))
    for nt in np.arange(n_trials):
        N_abs = np.random.poisson(norm)
        flux = np.ones(len(v_grid))
        for ii,na in enumerate(np.arange(N_abs)):
            v_Hubble = np.random.choice(v_grid,p=probs)
            v_peculiar = np.random.normal(loc=0.,scale=sigma1D)
            v_add = v_Hubble + v_peculiar
            one_abs = models.GaussianAbsorption1D(amplitude=3.0,mean=v_add,stddev=stddev_oneabs)
            flux = flux*one_abs(v_grid)
            flux[np.where(flux < 0.)] = 0.
        stck_img[nt,:] = flux

    return stck_img


# Convert the standard deviation of one absorber to equivalent width
def stddev_oneabs_to_WCII(stddev_oneabs):
    one_abs = models.GaussianAbsorption1D(amplitude=3.0,mean=0,stddev=stddev_oneabs)
    flux = one_abs(v_grid)
    flux[np.where(flux < 0.)] = 0.
    tau = -np.log(flux)
    total_flux = np.exp(-tau)
    W_CII = np.sum((1.-total_flux[:-1])*np.diff(wave_grid.value))
    return W_CII

# Model the velocity width of a stack
def model_width(stddev_oneabs,sigma1D,all_dict):
    sv_mean_flux = []
    for ii,idict in enumerate(all_dict):
        if idict is None:
            continue
        if 'J1508+3635' in idict['qpq']['NAME']:
            continue
        else:
            stck_img = gen_stck_img(idict['qpq']['R_PHYS'],idict['qpq']['Z_FG'],stddev_oneabs,sigma1D)
            mean_flux = np.sum(stck_img,0)/n_trials
            sv_mean_flux.append(mean_flux)
    mean_flux = np.mean(sv_mean_flux,0)
    model_init = models.GaussianAbsorption1D(amplitude=0.11,mean=0.,stddev=250.)
    c_init = models.Const1D(amplitude=0.997)
    model_init = c_init*model_init
    fit = fitting.LevMarLSQFitter()
    model_final = fit(model_init,v_grid,mean_flux)
    return model_final.stddev_1.value,model_final.amplitude_1.value


# Generate the data points for the contour plot
def contour_data(all_dict):
    stddev_oneabs_grid = np.arange(20.,71.,2.5)
    sigma1D_grid = np.arange(110.,311.,5.)
    sigma1D_GRID,stddev_oneabs_GRID = np.meshgrid(sigma1D_grid,stddev_oneabs_grid)
    widths = np.zeros_like(stddev_oneabs_GRID)
    amps = np.zeros_like(stddev_oneabs_GRID)
    for ii in np.arange(widths.shape[0]):
        print('ii=',ii,'of',widths.shape[0],'loops total')
        for jj in np.arange(widths.shape[1]):
            widths[ii,jj],amps[ii,jj] = model_width(stddev_oneabs_GRID[ii,jj],sigma1D_GRID[ii,jj],all_dict)
    WCII_grid = np.zeros_like(stddev_oneabs_grid)
    width_levels = [142,186,224,295,329,362]
    amp_bounds = [0.109,0.173] # 3-sigma bounds
    for ii,wcii in enumerate(WCII_grid):
        WCII_grid[ii] = stddev_oneabs_to_WCII(stddev_oneabs_grid[ii])
    mass_grid = np.zeros_like(sigma1D_grid)
    for ii,mass in enumerate(mass_grid):
        mass_grid[ii] = np.log10(2.37*10**26*(sigma1D_grid[ii]*1.4/1.07*1000)**3/(1.99*10**30))
    mass_GRID,WCII_GRID = np.meshgrid(mass_grid,WCII_grid)
    levels = np.zeros_like(stddev_oneabs_GRID)
    for ii in np.arange(widths.shape[0]):
        for jj in np.arange(widths.shape[1]):
            if ((widths[ii,jj] >= width_levels[0]) & (widths[ii,jj] < width_levels[1])) | \
            ((widths[ii,jj] >= width_levels[4]) & (widths[ii,jj] < width_levels[5])):
                levels[ii,jj] = 1.
            if ((widths[ii,jj] >= width_levels[1]) & (widths[ii,jj] < width_levels[2])) | \
            ((widths[ii,jj] >= width_levels[3]) & (widths[ii,jj] < width_levels[4])):
                levels[ii,jj] = 2.
            if (widths[ii,jj] >= width_levels[2]) & (widths[ii,jj] < width_levels[3]):
                levels[ii,jj] = 3.
            if (amps[ii,jj] < amp_bounds[0]) | (amps[ii,jj] > amp_bounds[1]):
                levels[ii,jj] = 0.
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(mass_GRID)
    hdu2 = fits.ImageHDU(WCII_GRID)
    hdu3 = fits.ImageHDU(levels)
    hdulist = fits.HDUList([hdu0,hdu1,hdu2,hdu3])
    hdulist.writeto('contour.fits',clobber=True)


######################
def main(contour_only=False):

    stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,
                               vsig_cut=400*u.km/u.s,zfg_mnx=(1.6,9999),plot_indiv=False)
    fin_velo, stck_img, stck_msk, all_dict = stack_tup0

    if contour_only is False:
        # 3 times the modeling error smaller than observed width, minus redshift error broadening,
        # gives intrinsic width 142 km/s
        width, amp = model_width(25.2,114.,all_dict)
        plt.figure(figsize=(8,5))
        plt.axis([-2000,2000,0.65,1.05])
        model_final = models.GaussianAbsorption1D(amplitude=amp,mean=0.,stddev=width)
        plt.plot(v_grid,model_final(v_grid))
        print('width and ammplitude, for 3 times error samller',width,amp)
        # QPQ halo mass sigma_1D
        width, amp = model_width(47,246.,all_dict)
        plt.figure(figsize=(8,5))
        plt.axis([-2000,2000,0.65,1.05])
        model_final = models.GaussianAbsorption1D(amplitude=amp,mean=0.,stddev=width)
        plt.plot(v_grid,model_final(v_grid))
        print('width and amplitude, for QPQ halo mass',width,amp)
        # save model
        dict = {}
        for ii,pp in enumerate(model_final.parameters):
            dict[model_final.param_names[ii]] = [pp]
        ascii.write(Table(dict),'monte.dat')
        # Find mean equivalent width of QPQ8 LLS
        mean_QPQ8 = np.mean([0.000,0.042,0.345,0.195,0.025,0.030,0.053,0.168,0.504,0.248,0.779,0.281,
                             0.341,0.328,0.756,0.650,0.154,0.179,0.130,0.950])
        print('mean EW of QPQ8 LLS',mean_QPQ8)

    # Generate data for contour plot
    contour_data(all_dict)


# Command line execution
if __name__ == '__main__':
    main(countour_only=sys.argv[1])