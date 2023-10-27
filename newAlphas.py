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

def alpha_scatter(sim, ax, color, marker):
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_offsets_Z   = []
    all_offsets_SFR = []
    all_offsets_R   = []
    
    redshifts  = []
    redshift   = 0
    
    for snap_index, snap in enumerate(snapshots): 
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
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
        R_star    =    R_star[desired_mask]
        
        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)
        Zgas   = np.log10(OH) + 12
        
        star_mass     = np.log10(star_mass)
        Zstar         = np.log10(Zstar    )
            
        MZR_mass , MZR_metals, _ = medianZR( star_mass, Zgas )
        SFMS_mass, SFMS_SFRs , _ = medianZR( star_mass, SFR )
        MS_mass  , MS_radius , _ = medianZR( star_mass, R_star )
        
        MZR         = interp1d( MZR_mass, MZR_metals, fill_value='extrapolate' )
        MZR_interp  = MZR(star_mass)
        
        SFMS        = interp1d( SFMS_mass, SFMS_SFRs, fill_value='extrapolate' )
        SFMS_interp = SFMS(star_mass)
        
        MS          = interp1d( MS_mass, MS_radius, fill_value='extrapolate' )
        MS_interp   = MS(star_mass)
        
        offsets_SFR = np.log10(SFR/SFMS_interp)
        offsets_Z   = Zgas - MZR_interp
        offsets_R   = R_star - MS_interp
        filter_nans = ~np.isnan( offsets_SFR )
        
        all_offsets_Z   += list( offsets_Z  [filter_nans] )
        all_offsets_SFR += list( offsets_SFR[filter_nans] )
        all_offsets_R   += list( offsets_R  [filter_nans] )
        redshifts       += list( np.ones(sum(filter_nans)) * redshift )
        
        redshift  += 1
        

    all_offsets_Z   = np.array( all_offsets_Z   )
    all_offsets_SFR = np.array( all_offsets_SFR )
    all_offsets_R   = np.array( all_offsets_R   )
    redshifts       = np.array( redshifts       )
        
    zs = np.arange(0,9)
    
    alpha_scatter_global, _ = np.polyfit( all_offsets_SFR, all_offsets_Z, 1 )
    
    alpha_scatter_global *= -1
    
    alpha_scatter_idv_z = np.ones( len(zs) ) * -1
    
    for index, z in enumerate(zs):
        mask = ( redshifts == z )
        
        alpha_scatter_idv_z[index], _ = np.polyfit( all_offsets_SFR[mask], all_offsets_Z[mask], 1 )
    
    alpha_scatter_idv_z *= -1
    
    ax.scatter( zs, alpha_scatter_idv_z, color=color, label=whichSim2Tex[sim], marker=marker, s=100, alpha=0.75 )
    ax.axhline( alpha_scatter_global, color=color, linestyle='--' )
    
    beta_scatter_global, _ = np.polyfit( all_offsets_R, all_offsets_Z, 1 )
    
    beta_scatter_global *= -1
    
    return alpha_scatter_global, alpha_scatter_idv_z, beta_scatter_global
    
def alpha_evo(sim, ax, color, marker):
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_offsets_Z   = []
    all_offsets_SFR = []
    all_offsets_R   = []
    
    redshifts  = []
    redshift   = 0
    
    MZRz0  = None
    SFMSz0 = None
    MSz0   = None
    
    for snap_index, snap in enumerate(snapshots): 
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
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
        R_star    =    R_star[desired_mask]
        
        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)
        Zgas   = np.log10(OH) + 12
        
        star_mass     = np.log10(star_mass)
        Zstar         = np.log10(Zstar    )
            
        MZR_mass , MZR_metals, _ = medianZR( star_mass, Zgas )
        SFMS_mass, SFMS_SFRs , _ = medianZR( star_mass, SFR )
        MS_mass  , MS_radius , _ = medianZR( star_mass, R_star )
        
        MZR         = interp1d( MZR_mass, MZR_metals, fill_value='extrapolate' )
        MZR_interp  = MZR(star_mass)
        
        SFMS        = interp1d( SFMS_mass, SFMS_SFRs, fill_value='extrapolate' )
        SFMS_interp = SFMS(star_mass)
        
        MS          = interp1d( MS_mass, MS_radius, fill_value='extrapolate' )
        MS_interp   = MS(star_mass)
        
        if snap_index == 0:
            MZRz0  = MZR
            SFMSz0 = SFMS
            MSz0   = MS
        
        offsets_SFR = np.log10(SFMS_interp/SFMSz0(star_mass))
        offsets_Z   = MZR_interp - MZRz0(star_mass)
        offsets_R   = MS_interp - MSz0(star_mass)
        filter_nans = ~np.isnan( offsets_SFR )
        
        all_offsets_Z   += list( offsets_Z  [filter_nans] )
        all_offsets_SFR += list( offsets_SFR[filter_nans] )
        all_offsets_R   += list( offsets_R  [filter_nans] )
        redshifts       += list( np.ones(sum(filter_nans)) * redshift )
        
        redshift  += 1
        

    all_offsets_Z   = np.array( all_offsets_Z   )
    all_offsets_SFR = np.array( all_offsets_SFR )
    all_offsets_R   = np.array( all_offsets_R   )
    redshifts       = np.array( redshifts       )
    
    zs = np.arange(0,9)
    
    alpha_evo_global, _ = np.polyfit( all_offsets_SFR, all_offsets_Z, 1 )
    alpha_evo_global *= -1
    
    alpha_evo_idv_z = np.zeros( len(zs) )
    
    for index, z in enumerate(zs):
        if index == 0:
            continue
        mask = (redshifts == z)
        x, y = all_offsets_SFR[mask], all_offsets_Z[mask]
        
        alpha_evo_idv_z[index], b = np.polyfit( x, y, 1 )
        
    alpha_evo_idv_z *= -1
    
    ax.scatter( zs, alpha_evo_idv_z, color=color, label=whichSim2Tex[sim], marker=marker, s=100, alpha=0.75 )
    ax.axhline( alpha_evo_global, color=color, linestyle='--' )
    
    beta_evo_global, _ = np.polyfit( all_offsets_R, all_offsets_Z, 1 )
    beta_evo_global *= -1
    
    
    return alpha_evo_global, alpha_evo_idv_z, beta_evo_global

def strong_FMR(sim):
    snapshots, snap2z, BLUE_DIR = switch_sim(sim)
    
    all_offsets_Z   = []
    all_offsets_SFR = []
    all_offsets_R   = []
    
    redshifts  = []
    redshift   = 0
    
    MZRz0  = None
    SFMSz0 = None
    MSz0   = None
    
    for snap_index, snap in enumerate(snapshots): 
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )
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
        R_star    =    R_star[desired_mask]
        
        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)
        Zgas   = np.log10(OH) + 12
        
        star_mass     = np.log10(star_mass)
        Zstar         = np.log10(Zstar    )
            
        MZR_mass , MZR_metals, _ = medianZR( star_mass, Zgas )
        SFMS_mass, SFMS_SFRs , _ = medianZR( star_mass, SFR )
        MS_mass  , MS_radius , _ = medianZR( star_mass, R_star )
        
        MZR         = interp1d( MZR_mass, MZR_metals, fill_value='extrapolate' )
        MZR_interp  = MZR(star_mass)
        
        SFMS        = interp1d( SFMS_mass, SFMS_SFRs, fill_value='extrapolate' )
        SFMS_interp = SFMS(star_mass)
        
        MS          = interp1d( MS_mass, MS_radius, fill_value='extrapolate' )
        MS_interp   = MS(star_mass)
        
        if snap_index == 0:
            MZRz0  = MZR
            SFMSz0 = SFMS
            MSz0   = MS
        
        offsets_SFR = np.log10(SFR/SFMSz0(star_mass))
        offsets_Z   = Zgas - MZRz0(star_mass)
        offsets_R   = R_star - MSz0(star_mass)
        filter_nans = ~np.isnan( offsets_SFR )
        
        all_offsets_Z   += list( offsets_Z  [filter_nans] )
        all_offsets_SFR += list( offsets_SFR[filter_nans] )
        all_offsets_R   += list( offsets_R  [filter_nans] )
        redshifts       += list( np.ones(sum(filter_nans)) * redshift )
        
        redshift  += 1
        

    all_offsets_Z   = np.array( all_offsets_Z   )
    all_offsets_SFR = np.array( all_offsets_SFR )
    all_offsets_R   = np.array( all_offsets_R   )
    redshifts       = np.array( redshifts       )
    
    zs = np.arange(0,9)
    
    alpha_global, _ = np.polyfit( all_offsets_SFR, all_offsets_Z, 1 )
    alpha_global *= -1
    
    alpha_idv_z = np.zeros( len(zs) )
    
    for index, z in enumerate(zs):
        if index == 0:
            continue
        mask = (redshifts == z)
        x, y = all_offsets_SFR[mask], all_offsets_Z[mask]
        
        alpha_idv_z[index], b = np.polyfit( x, y, 1 )
        
    alpha_idv_z *= -1
    
    beta_global, _ = np.polyfit( all_offsets_R, all_offsets_Z, 1 )
    beta_global *= -1
    
    return alpha_global, alpha_idv_z, beta_global


def medianZR( x, y, nbins = 30 ):
    
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
            bin_cols[index] = np.std( y[mask] )
    
    no_nans = ~(np.isnan(bin_vals))
    
    return bin_centers[no_nans], bin_vals[no_nans], bin_cols[no_nans]

def line(data, a, b):
    return a*data + b

def sfmscut(m0, sfr0):
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
        idxb  = (ssfrb - rdg) > -5.000E-01
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
        idxb  = (ssfrb - ssfrlin[i]) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool  

sims    = ['ORIGINAL','TNG','EAGLE']
colors  = ['C1','C2','C0']
markers = ['^','*','s']

scatter_globals = []
scatter_idv     = []
beta_scatter    = []
beta_evo        = []
strong_globals  = []
strong_idv      = []
strong_beta     = []

scatter = False
evo     = False
strong  = True

if scatter:
    fig = plt.figure( figsize=(12,6) ) 
    ax  = plt.gca()

    print('scatter')
    for idx, sim in enumerate(sims):
        print(sim)
        alpha_scatter_global, alpha_scatter_idv_z, beta_scatter_global = alpha_scatter(sim, ax, colors[idx], markers[idx])

        scatter_globals.append(alpha_scatter_global)
        scatter_idv    .append(alpha_scatter_idv_z)
        
        beta_scatter.append( beta_scatter_global )

        np.save( BLUE + 'FMR_paper2/data/' + '%s_alpha_s_g.npy' %sim, alpha_scatter_global  )
        np.save( BLUE + 'FMR_paper2/data/' + '%s_alpha_s_z.npy' %sim, alpha_scatter_idv_z   )
        
    leg = plt.legend( frameon=False, handlelength=0, labelspacing=0.05 )
    for index, text in enumerate(leg.get_texts()):
        text.set_color(colors[index])

    plt.xlabel( r'${\rm Redshift}$' )
    plt.ylabel( r'$\alpha_{\rm scatter}$' )

    plt.tight_layout()
    plt.savefig( BLUE + 'FMR_paper2/' + 'alpha_scatter' + '.pdf', bbox_inches='tight' )

if evo:
    plt.clf()

    fig = plt.figure( figsize=(12,6) ) 
    ax  = plt.gca()

    evo_globals = []
    evo_idv     = []

    if scatter:
        print('')
    print('evolution')
    for idx, sim in enumerate(sims):
        print(sim)
        alpha_evo_global, alpha_evo_idv_z, beta_evo_global = alpha_evo( sim , ax, colors[idx], markers[idx])

        evo_globals.append(alpha_evo_global)
        evo_idv    .append(alpha_evo_idv_z)
        
        beta_evo.append( beta_evo_global )
        
        np.save( BLUE + 'FMR_paper2/data/' + '%s_alpha_e_g.npy' %sim, alpha_evo_global  )
        np.save( BLUE + 'FMR_paper2/data/' + '%s_alpha_e_z.npy' %sim, alpha_evo_idv_z   )

    leg = plt.legend( frameon=False, handlelength=0, labelspacing=0.05 )
    for index, text in enumerate(leg.get_texts()):
        text.set_color(colors[index])

    plt.xlabel( r'${\rm Redshift}$' )
    plt.ylabel( r'$\alpha_{\rm evo}$' )

    plt.tight_layout()
    plt.savefig( BLUE + 'FMR_paper2/' + 'alpha_evo' + '.pdf', bbox_inches='tight' )

if strong:
    print('')
    print('strong FMR')
    
    for idx, sim in enumerate(sims):
        print(sim)
        alpha_global, alpha_idv_z, beta_global = strong_FMR( sim )
        
        strong_globals.append( alpha_global )
        strong_idv.append( alpha_idv_z )
        
        strong_beta.append( beta_global )
        
        np.save( BLUE + 'FMR_paper2/data/' + '%s_strong_g.npy' %sim, alpha_global  )
        np.save( BLUE + 'FMR_paper2/data/' + '%s_strong_z.npy' %sim, alpha_idv_z   )
    
if scatter and evo:
    print('')
    print('comparison')
    plt.clf()
    fig = plt.figure( figsize=(12,6) ) 
    
    for idx, sim in enumerate(sims):
        print(sim)
        plt.axhline( scatter_globals[idx] / evo_globals[idx] , color=colors[idx], linestyle='--' )
        # plt.scatter( np.arange(0,9), scatter_idv[idx] / evo_idv[idx], color=colors[idx],
        #              marker=markers[idx], label=whichSim2Tex[sim], s=100 )
        
    print(scatter_globals, evo_globals)
    
    print( beta_scatter, beta_evo )
        
    plt.axhline( 1, color='k' )
    leg = plt.legend( frameon=False, handlelength=0, labelspacing=0.05 )
    for index, text in enumerate(leg.get_texts()):
        text.set_color(colors[index])

    plt.xlabel( r'${\rm Redshift}$' )
    plt.ylabel( r'$\alpha_{\rm scatter} / \alpha_{\rm evo}$' )

    plt.tight_layout()
    plt.savefig( BLUE + 'FMR_paper2/' + 'alpha_comparison' + '.pdf', bbox_inches='tight' )