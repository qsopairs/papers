#Module for QPQ9 stacking
# Imports
from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import glob, os, sys, copy
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from astropy import units as u
from linetools.spectra.xspectrum1d import XSpectrum1D
sys.path.append(os.path.abspath("../../../../py"))
from enigma.qpq import stacks as qpqk
from enigma.qpq import qpq as eqpq

def qpq9_IRMgII(wrest=None, outfil=None, nboot=10000,
                vmnx = (-3000., 3000.)*u.km/u.s,
                vsig_cut = None,
                zfg_mnx = (-9999,9999),
                S2N_cut = None,
                atmosphere_cut = True,
                stack_fg = False,
                plot_indiv = True,
                stack_tup=None,passback=False,debug=False):
    ''' Stack the QPQ9 sample
    '''
    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA

    # Load QPQ9
    qpq9 = eqpq.QPQ('QPQ9')
    # Avoid Lya forest
    qpq9.cull(wrest,4,vsig_cut=vsig_cut)
    # Cut on redshift
    zfg = qpq9._fulldata['Z_FG']
    bad = np.where((zfg < zfg_mnx[0]) | (zfg > zfg_mnx[1]))[0]
    qpq9.mask[bad] = False

    if outfil is None:
        outfil = 'Output/QPQ9_IRMgII_{:d}.fits'.format(int(wrest.value))

    # Load the stack image
    if stack_tup is None:
        stack_tup = qpqk.load_stack_img(qpq9.data,wrest,vmnx=vmnx,spec_dv=100.*u.km/u.s,high_res=0,
                                        S2N_cut=S2N_cut,atmosphere_cut=atmosphere_cut,
                                        stack_fg=stack_fg,plot_indiv=plot_indiv,
                                        QPQ_conti_path='/Users/lwymarie/Dropbox/Marie_X/QPQ9/Continua/QPQ/data/',
                                        igmspec_conti_path='/Users/lwymarie/Dropbox/Marie_X/QPQ9/Continua/igmspec/')
        if passback:
            return stack_tup

    # Stack
    fin_velo, fin_flx, all_dict = qpqk.stack_avg(stack_tup)

    # Write spectrum (in Rest Wavelength)
#    fin_wave = ((fin_velo/const.c)*wrest).to(u.AA) + wrest
    relativistic_equiv = u.doppler_relativistic(wrest)
    fin_wave = fin_velo.to(u.AA,equivalencies=relativistic_equiv)

    xspec1d = XSpectrum1D.from_tuple((fin_wave, u.Quantity(fin_flx)))
    xspec1d.write_to_fits(outfil)

    return stack_tup

#######
def plt_qpq9(stack_tup=None,wrest=None,
             S2N_cut=None,atmosphere_cut=True,
             vmnx=(-3000.,3000.)*u.km/u.s,
             stack_fg=False,zfg_mnx=(-9999,9999),plot_indiv=False):

    # Rest wavelength
    if wrest is None:
        wrest = 1334.5323*u.AA

    ymnx = (-0.1, 1.2)
    if stack_tup is None:
        stack_tup = qpq9_IRMgII(wrest=wrest,S2N_cut=S2N_cut,
                                atmosphere_cut=atmosphere_cut,
                                vmnx=vmnx,
                                stack_fg=stack_fg,zfg_mnx=zfg_mnx,
                                plot_indiv=plot_indiv,
                                passback=True)
    fin_velo, stck_img, stck_msk, all_dict = stack_tup


    nplt = len(all_dict) 
    pages = nplt//15 + ((nplt % 15) > 0)
    ncol = 3 
    nrow = 5
    for ipage in range(pages):
        # Start the plot
        if stack_fg is True:
            outfil = 'plt_qpq9_IRMgII_{:d}_fg_page{:d}.pdf'.format(int(wrest.value),ipage+1)
        else:
            outfil = 'plt_qpq9_IRMgII_{:d}_page{:d}.pdf'.format(int(wrest.value),ipage+1)
        pp = PdfPages(outfil)
        plt.figure(figsize=(8,5))
        plt.clf()
        gs = gridspec.GridSpec(nrow,ncol)

        # Loop on systems 
        end_plt = np.min([ipage*15+15,nplt])
        for ss,idict in enumerate(all_dict[ipage*15:end_plt]):
            if idict is None:
                continue
            ax = plt.subplot(gs[ss%nrow,ss//nrow])
            ax.set_xlim(vmnx.value)
            ax.set_ylim(ymnx)
            if ((ss+1)%nrow) == 0:
                ax.set_xlabel('Relative Velocity (km/s)',size=9)
            else:
                ax.get_xaxis().set_ticks([])
            ax.set_ylabel('Normalized flux',size=9)

            # Generate stuff
            wvobs = wrest*(1+idict['zfg'])
            velo = idict['spec'].relative_vel(wvobs)

            # Plot original data
            subp = idict['spec'].sub_pix
            ax.plot(velo[subp],idict['spec'].flux[subp],drawstyle='steps-mid',color='black')
            # Plot processed 
            ax.plot(fin_velo,stck_img[ipage*15+ss,:],color='green',drawstyle='steps-mid')

            # Label 
            ax.text(-900,0,'{:s}'.format(idict['qpq']['NAME']),size=7,ha='left')
            # Lines
            ax.plot(vmnx.value, [1.]*2, 'g:')
            ax.plot([0]*2, ymnx, 'b--')
            # Font
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(7)

        # Layout and save
        try:
            plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
        except ValueError:
            pass
        pp.savefig(bbox_inches='tight')
        pp.close()
        print('Wrote {:s}'.format(outfil))

#### ########################## #########################
#### ########################## #########################

def main(flg_stck):

    if flg_stck == 'all':
        flg_stck = np.sum( np.array( [2**ii for ii in range(1)] ))
    else:
        flg_stck = int(flg_stck)

    # Simple bootstrap of CII 1334, restricted to OIII
    # not yet bootstrapped 
    if (flg_stck % 2**1) >= 2**0:
        qpq9()

    # Plot all the transitions
    if (flg_stck % 2**2) >= 2**1:
        plt_qpq9()

    # Simple bootstrap of CIV 1548
    #if (flg_fig % 2**1) >= 2**0:
    #    boot_trans() # CII 1334

    # All done
    print('All done!')


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1: #
        flg_stck = 0
      #  flg_stck += 1     # 1334 
        flg_stck += 2**1  # Plot all the systems
    else:
        flg_stck = sys.argv[1]

    main(flg_stck)
