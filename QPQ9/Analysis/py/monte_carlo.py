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
sys.path.append(os.path.abspath("../Stacks/py"))
import qpq9_stacks as qpq9k
import scipy.integrate as integrate
from astropy.modeling import models,fitting
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70,Om0=0.26)

def gen_stck_img(R_phys,z_fg,Deltav,v_grid,n_trials): # R_phys in kpc, Deltav in km/s
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
        Deltaz = ltu.z_from_v(z_fg,v_grid[ii]+Deltav/2)-ltu.z_from_v(z_fg,v_grid[ii]-Deltav/2)
        prob = (ell_IGM_DLA*(1+chi_DLA)+ell_IGM_SLLS*(1+chi_SLLS)+ell_IGM_LLS*(1+chi_LLS))*Deltaz
        # test effect of doubling number of absorbers
#        prob = prob*2
        probs[ii] = prob
        cumulative_probs[ii] = prob + np.sum(probs[0:ii])
    norm = np.sum(probs)
    probs = probs/norm
    print('Expected number of absorbers at R_phys',norm,R_phys)
    stck_img = np.zeros((n_trials,len(v_grid)))
    for nt in np.arange(n_trials):
        N_abs = np.random.poisson(norm)
        flux = np.ones(len(v_grid))
        for ii,na in enumerate(np.arange(N_abs)):
            v_Hubble = np.random.choice(v_grid,p=probs)
#            v_peculiar = np.random.normal(loc=0.,scale=246.) # sigma_1D for QPQ halo mass
            v_peculiar = np.random.normal(loc=0.,scale=74.) # Try to produce sigma_1D^2 +182^2 = (312-35*3)^2
            v_add = v_Hubble + v_peculiar
            # Higher rest EW to match data amplitude
#            one_abs = models.GaussianAbsorption1D(amplitude=3.0,mean=v_add,stddev=45.)
            one_abs = models.GaussianAbsorption1D(amplitude=3.0,mean=v_add,stddev=17.) # To match (312-35*3)^2
            flux = flux*one_abs(v_grid)
            flux[np.where(flux < 0.)] = 0.
        stck_img[nt,:] = flux

    return stck_img

######################
def main():

    stack_tup0 = qpq9k.qpq9_IRMgII(passback=True,wrest=1334.5323*u.AA,S2N_cut=5.5/u.AA,
                               vsig_cut=400*u.km/u.s,zfg_mnx=(1.6,9999),plot_indiv=False)
    fin_velo, stck_img, stck_msk, all_dict = stack_tup0

    n_trials = 1000
    Deltav = 10.
    v_grid = -2000. + np.arange((2000.-(-2000.))/Deltav+1)*Deltav
    relativistic_equiv = u.doppler_relativistic(1334.5323*u.AA)
    wave_grid = (v_grid*u.km/u.s).to(u.AA,equivalencies=relativistic_equiv)
    sv_mean_flux = []
    for ii,idict in enumerate(all_dict):
        if idict is None:
            continue
        if 'J1508+3635' in idict['qpq']['NAME']:
            continue
        else:
            stck_img = gen_stck_img(idict['qpq']['R_PHYS'],idict['qpq']['Z_FG'],Deltav,v_grid,n_trials)
            mean_flux = np.sum(stck_img,0)/n_trials
            sv_mean_flux.append(mean_flux)

    mean_flux = np.mean(sv_mean_flux,0)
    plt.figure(figsize=(8,5))
    plt.plot(v_grid,mean_flux,drawstyle='steps-mid',linewidth=2.,)
    plt.axis([-2000,2000,0.65,1.05])
    print('Rest EW in angstrom:',np.sum((1.-mean_flux[:-1])*np.diff(wave_grid.value)))
    model_init = models.GaussianAbsorption1D(amplitude=0.11,mean=0.,stddev=250.)
    c_init = models.Const1D(amplitude=0.997)
    model_init = c_init*model_init
    fit = fitting.LevMarLSQFitter()
    model_final = fit(model_init,v_grid,mean_flux)
    plt.plot(v_grid,model_final(v_grid))
    print(model_final)

    # save model
    dict = {}
    for ii,pp in enumerate(model_final.parameters):
        dict[model_final.param_names[ii]] = [pp]
    ascii.write(Table(dict),'monte.dat')

# Command line execution
if __name__ == '__main__':
    main()