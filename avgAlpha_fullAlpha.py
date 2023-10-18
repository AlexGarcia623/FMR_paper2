import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import illustris_python as il

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

mpl.rcParams['text.usetex']        = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

mpl.rcParams['font.size'] = 25
mpl.rcParams['axes.linewidth'] = 2.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 7.5
mpl.rcParams['ytick.major.size'] = 7.5
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

BLUE = './blue_FMR/'

WHICH_SIM    = "eagle".upper() 
STARS_OR_GAS = "gas".upper() # stars or gas

BLUE_DIR = BLUE + WHICH_SIM + "/"

whichSim2Tex = {
    'TNG'     :r'${\rm TNG}$',
    'ORIGINAL':r'${\rm Illustris}$',
    'EAGLE'   :r'${\rm EAGLE}$'
}

run, base, out_dir, snapshots = None, None, None, []
snap2z = {}
color  = {}

def switch_sim(WHICH_SIM):
    BLUE_DIR = BLUE + WHICH_SIM + "/"
    if (WHICH_SIM.upper() == "TNG"):
        # TNG
        run       = 'L75n1820TNG'
        base      = '/orange/paul.torrey/IllustrisTNG/Runs/' + run + '/' 
        out_dir   = base 
        snapshots = [99,50,33,25,21,17,13,11,8] # 6,4
        snap2z = {
            99:'z=0',
            50:'z=1',
            33:'z=2',
            25:'z=3',
            21:'z=4',
            17:'z=5',
            13:'z=6',
            11:'z=7',
            8 :'z=8',
            6 :'z=9',
            4 :'z=10',
        }
    elif (WHICH_SIM.upper() == "ORIGINAL"):
        # Illustris
        run       = 'L75n1820FP'
        base      = '/orange/paul.torrey/Illustris/Runs/' + run + '/'
        out_dir   = base
        snapshots = [135,86,68,60,54,49,45,41,38] # 35,32
        snap2z = {
            135:'z=0',
            86 :'z=1',
            68 :'z=2',
            60 :'z=3',
            54 :'z=4',
            49 :'z=5',
            45 :'z=6',
            41 :'z=7',
            38 :'z=8',
            35 :'z=9',
            32 :'z=10',
        }
    elif (WHICH_SIM.upper() == "EAGLE"):
        EAGLE_DATA = BLUE_DIR + 'data/'
        snapshots = [28,19,15,12,10,8,6,5,4] # 3,2
        snap2z = {
            28:'z=0',
            19:'z=1',
            15:'z=2',
            12:'z=3',
            10:'z=4',
             8:'z=5',
             6:'z=6',
             5:'z=7',
             4:'z=8',
             3:'z=9',
             2:'z=10'
        }
    return snapshots, snap2z, BLUE_DIR
        

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02
Zsun   = 1.27E-02

m_star_min = 8.0
m_star_max = 10.5
m_gas_min  = 8.5


def get_full_FMR(sim,plot=False):
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
    
        
    for snap in snapshots:
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
    OH     = Zgas * (zo/xh) * (1.00/16.00)

    Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
    gas_mass  = gas_mass [nonans]
    star_mass = star_mass[nonans]
    SFR       = SFR      [nonans]
    sSFR      = sSFR     [nonans]
    Zstar     = Zstar    [nonans]
    Zgas      = Zgas     [nonans]
    redshifts = redshifts[nonans]
    R_gas     = R_gas    [nonans]
    R_star    = R_star   [nonans]

    star_mass     = np.log10(star_mass)
    Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)

    disps = np.ones(len(alphas)) * np.nan
    
    a_s, b_s = np.ones( len(alphas) ), np.ones( len(alphas) )
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        muCurrent  = star_mass - alpha*np.log10( SFR )

        mu_fit = muCurrent
        Z_fit  = Z_use

        popt = np.polyfit(mu_fit, Z_fit, 1)
        a_s[index], b_s[index] = popt
        interp = np.polyval( popt, mu_fit )

        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 

    argmin = np.argmin(disps)

    min_alpha = round( alphas[argmin], 2 )
    min_a, min_b = a_s[argmin], b_s[argmin]
    
    unique, n_gal = np.unique(redshifts, return_counts=True)

    newcolors   = plt.cm.rainbow(np.linspace(0, 1, len(unique)-1))
    CMAP_TO_USE = ListedColormap(newcolors)
    
    if plot:
        plt.clf()
        
        plt.figure(figsize=(8,6))
        
        mu = star_mass - min_alpha * np.log10(SFR)
        
        Hist1, xedges, yedges = np.histogram2d(mu,Z_use,weights=redshifts,bins=(100,100))
        Hist2, _     , _      = np.histogram2d(mu,Z_use,bins=[xedges,yedges])

        Hist1 = np.transpose(Hist1)
        Hist2 = np.transpose(Hist2)

        hist = Hist1/Hist2

        mappable = plt.pcolormesh( xedges, yedges, hist, vmin = 0, vmax = 8, 
                                   cmap=CMAP_TO_USE, rasterized=True )
        
        plt.colorbar(label=r'${\rm Redshift}$')
        
        plt.xlabel( r"$\mu_{%s}$" %(min_alpha) )
        plt.ylabel( r"$\log({\rm O/H}) + 12 ~{\rm (dex)}$" )
        
        plt.text( 0.05, 0.9, whichSim2Tex[sim], transform=plt.gca().transAxes )
        
        xs = np.linspace( np.min(mu), np.max(mu), 100 )
        ys = min_a * xs + min_b
        
        plt.plot( xs, ys, color='k', lw=2 )
        
        plt.tight_layout()
        plt.savefig( BLUE + 'FMR_paper2/' + '%s_fullFMR' %sim + '.pdf' )
    
    return min_alpha, min_a, min_b

def get_avg_FMR(sim,full_alpha,plot=False):
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_Zgas      = []
    all_Zstar     = []
    all_star_mass = []
    all_gas_mass  = []
    all_SFR       = []
    all_R_gas     = []
    all_R_star    = []
    
    redshifts  = []
    redshift   = 0
        
    for snap in snapshots:
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
        R_gas     = np.load( currentDir + 'R_gas.npy' )
        R_star    = np.load( currentDir + 'R_star.npy' )
        
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))
        
        gas_mass  =  gas_mass[desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       =       SFR[desired_mask]
        Zstar     =     Zstar[desired_mask]
        Zgas      =      Zgas[desired_mask]
        R_gas     =     R_gas[desired_mask]
        R_star    =    R_star[desired_mask]
        
        nbins = 25
        star_mass = np.log10(star_mass)
        OH     = Zgas * (zo/xh) * (1.00/16.00)
        Zgas   = np.log10(OH) + 12
        
        star_mass, Zgas, SFR = medianZR( star_mass, Zgas, SFR, nbins=nbins )
                
        all_Zgas     += list(Zgas     )
        all_Zstar    += list(Zstar    )
        all_star_mass+= list(star_mass)
        all_gas_mass += list(gas_mass )
        all_SFR      += list(SFR      )
        all_R_gas    += list(R_gas    )
        all_R_star   += list(R_star   )

        redshifts += list( np.ones(len(Zgas)) * redshift )
        
        redshift  += 1
        
    Zgas      = np.array(all_Zgas      )
    Zstar     = np.array(all_Zstar     )
    star_mass = np.array(all_star_mass )
    gas_mass  = np.array(all_gas_mass  )
    SFR       = np.array(all_SFR       )
    R_gas     = np.array(all_R_gas     )
    R_star    = np.array(all_R_star    )
    redshifts = np.array(redshifts     )

    Zstar /= Zsun
#     OH     = Zgas * (zo/xh) * (1.00/16.00)

#     Zgas      = np.log10(OH) + 12

    # Get rid of nans and random values -np.inf
    # nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

    sSFR      = SFR/star_mass
    
#     gas_mass  = gas_mass [nonans]
#     star_mass = star_mass[nonans]
#     SFR       = SFR      [nonans]
#     sSFR      = sSFR     [nonans]
#     Zstar     = Zstar    [nonans]
#     Zgas      = Zgas     [nonans]
#     redshifts = redshifts[nonans]
#     R_gas     = R_gas    [nonans]
#     R_star    = R_star   [nonans]

    # star_mass     = np.log10(star_mass)
    # Zstar         = np.log10(Zstar)

    alphas = np.linspace(0,1,100)

    disps = np.ones(len(alphas)) * np.nan
    
    a_s, b_s = np.ones( len(alphas) ), np.ones( len(alphas) )
    
    if (STARS_OR_GAS == "GAS"):
        Z_use = Zgas
    elif (STARS_OR_GAS == "STARS"):
        Z_use = Zstar

    for index, alpha in enumerate(alphas):

        muCurrent  = star_mass - alpha*np.log10( SFR )

        mu_fit = muCurrent
        Z_fit  = Z_use

        popt = np.polyfit(mu_fit, Z_fit, 1)
        a_s[index], b_s[index] = popt
        interp = np.polyval( popt, mu_fit )

        disps[index] = np.std( np.abs(Z_fit) - np.abs(interp) ) 

    argmin = np.argmin(disps)

    min_alpha = round( alphas[argmin], 2 )
    min_a, min_b = a_s[argmin], b_s[argmin]
    
    unique, n_gal = np.unique(redshifts, return_counts=True)

    newcolors   = plt.cm.rainbow(np.linspace(0, 1, len(unique)-1))
    CMAP_TO_USE = ListedColormap(newcolors)
    
    if plot:
        plt.clf()
        
        fig,axs = plt.subplots(2,1,figsize=(8,12),gridspec_kw={'height_ratios': [0.75, 1]})
        
        mu = star_mass - min_alpha * np.log10(SFR)
        
        for _z_ in unique:
            norm = mpl.colors.Normalize(vmin=0, vmax=8)
            
            mask = redshifts==_z_
            
            axs[1].plot( mu[mask] , Z_use[mask], color=CMAP_TO_USE(norm(_z_)) )
        
        mappable = axs[1].scatter( mu , Z_use, c=redshifts, cmap=CMAP_TO_USE, s=0.1 )
        
        axs[1].set_xlabel( r"$\langle\mu_{%s}\rangle$" %(min_alpha) )
        axs[1].set_ylabel( r"$\langle\log({\rm O/H}) + 12\rangle ~{\rm (dex)}$" )
        
        axs[1].text( 0.05, 0.8, r'$\alpha_{\rm full} = %s$' %full_alpha, transform=axs[1].transAxes )
        axs[1].text( 0.05, 0.7, r'$\alpha_{\rm avg}  = %s$' %min_alpha , transform=axs[1].transAxes )
        
        xs = np.linspace( np.min(mu), np.max(mu), 100 )
        ys = min_a * xs + min_b
        
        axs[1].plot( xs, ys, color='k', lw=2 )
        
        plt.tight_layout()
        # plt.savefig( BLUE + 'FMR_paper2/' + '%s_avgFMR' %sim + '.pdf' )
        
        # plt.figure(figsize=(8,6))
        
        for _z_ in unique:
            norm = mpl.colors.Normalize(vmin=0, vmax=8)
            
            mask = redshifts==_z_
            
            axs[0].plot( star_mass[mask] , Z_use[mask], color=CMAP_TO_USE(norm(_z_)) )
        
        mappable = axs[0].scatter(star_mass, Z_use, c=redshifts, cmap=CMAP_TO_USE, s=0.1)
        plt.colorbar(mappable,label=r'${\rm Redshift}$',orientation='horizontal')
        # plt.colorbar(label=r'${\rm Redshift}$')
        
        axs[0].set_xlabel( r"$\log \langle M_*\rangle~(\log M_\odot)$" )
        axs[0].set_ylabel( r"$\langle\log({\rm O/H}) + 12\rangle ~{\rm (dex)}$" )
        
        axs[0].text( 0.05, 0.9, whichSim2Tex[sim], transform=axs[0].transAxes )
        
        plt.subplots_adjust( wspace=0.0 )
        
        plt.tight_layout()
        plt.savefig( BLUE + 'FMR_paper2/' + '%s_avg' %sim + '.pdf' )
    
    return min_alpha, min_a, min_b

def get_avg(data, n_bins):
    
    low  = np.min(data)
    high = np.max(data)
    
    return np.linspace(low, high, n_bins)

def medianZR( x, y, z, nbins = 5 ):
    
    start = np.min(x)
    end   = np.max(x)
    
    bin_centers = np.linspace(start,end,nbins)
    binWidth = (end - start) / nbins
    
    bin_vals = np.ones(len(bin_centers))
    bin_cols = np.ones(len(bin_centers))
    
    for index, current in enumerate(bin_centers):
        mask = ((x > current - binWidth/2) & (x < current + binWidth/2))
        
        if (sum(mask) < 10):
            bin_vals[index] = np.nan
            bin_cols[index] = np.nan
        else:
            bin_vals[index] = np.nanmedian( y[mask] )
            bin_cols[index] = np.nanmedian( z[mask] )
    
    no_nans = ~(np.isnan(bin_vals))
    
    return bin_centers[no_nans], bin_vals[no_nans], bin_cols[no_nans]

def line(data, a, b):
    return a*data + b

def sfmscut(m0, sfr0, threshold=-5.00E-01):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) & 
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.0200E+01
    mstp    = 2.0000E-01
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []


    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > threshold
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00
    
    # Alex added this as a quick bug fix, no idea if it's ``correct''
    nonans = (~(np.isnan(mcs)) &
              ~(np.isnan(rdgs)) &
              ~(np.isnan(rdgs)))
        
    parms, cov = curve_fit(line, mcs[nonans], rdgs[nonans], sigma = rdgstds[nonans])
    mmin    = mbrk
    mmax    = m_star_max
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
        
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > threshold
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool


sims   = ['ORIGINAL','TNG','EAGLE']


for sim in sims:
    
    alpha_full, a, b = get_full_FMR(sim,plot=False)
    
    alpha_avg , a, b = get_avg_FMR(sim, alpha_full, plot=True)
    
    print( sim ) 
    print( '\t' + 'Full FMR: %s' %alpha_full + ' ' + 'Avg FMR: %s' %alpha_avg )