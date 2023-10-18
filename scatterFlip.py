import sys
import os
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

fs_og = 32
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*2
mpl.rcParams['ytick.major.width'] = 1.5*2
mpl.rcParams['xtick.minor.width'] = 1.0*2
mpl.rcParams['ytick.minor.width'] = 1.0*2
mpl.rcParams['xtick.major.size']  = 7.5*2
mpl.rcParams['ytick.major.size']  = 7.5*2
mpl.rcParams['xtick.minor.size']  = 3.5*2
mpl.rcParams['ytick.minor.size']  = 3.5*2
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

BLUE = './blue_FMR/'

FLIP = False

WHICH_SIM    = "eagle".upper() 
STARS_OR_GAS = "gas".upper() # stars or gas

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

def do(flip, plot=False):
    
    sim_MZR     = 'EAGLE'
    sim_scatter = 'ORIGINAL'
    
    if flip:
        sim_MZR     = 'ORIGINAL'
        sim_scatter = 'EAGLE'
    
    # First get the scatters about the MZR from the scatter simulation
    snapshots, snap2z, BLUE_DIR = switch_sim(sim_scatter)
    print('Getting offsets')
    
    # What we want is:
    #    (i)   the scatter about the MZR
    #    (ii)  the masses of all the galaxies
    #    (iii) the SFRs of all the galaxies
    all_scatters = []
    all_masses   = []
    all_SFRs     = []
    
    fig, axs = plt.subplots( 1, 3, figsize=(16,7), sharey=True, sharex=True )
    
    ax_left   = axs[0]
    ax_center = axs[1]
    ax_right  = axs[2]
    
    for snap_index, snap in enumerate(snapshots):
        # if (snap_index) != 7:
        #     all_scatters.append([])
        #     all_masses.append([])
        #     all_SFRs.append([])
        #     continue
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )

        THRESHOLD = -5.00E-01
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))

        gas_mass  = gas_mass [desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       = SFR      [desired_mask]
        Zstar     = Zstar    [desired_mask]
        Zgas      = Zgas     [desired_mask]

        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)

        Zgas      = np.log10(OH) + 12

        # Get rid of nans and random values -np.inf
        nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

        sSFR      = SFR/star_mass

        sSFR[~(sSFR > 0.0)] = 1e-16

        star_mass = star_mass[nonans]
        sSFR      = sSFR     [nonans]
        Zstar     = Zstar    [nonans]
        Zgas      = Zgas     [nonans]

        gas_mass      = np.log10(gas_mass)
        star_mass     = np.log10(star_mass)
        Zstar         = np.log10(Zstar)

        if (STARS_OR_GAS == "GAS"):
            Z_use = Zgas
        elif (STARS_OR_GAS == "STARS"):
            Z_use = Zstar
        
        masses, metals = getMedians(star_mass,Z_use)
        
        filter_fit_nans = ~(np.isnan(metals))
        
        masses = masses[filter_fit_nans]
        metals = metals[filter_fit_nans]
        
        MZsR = interp1d(masses,metals,fill_value='extrapolate')
        
        if plot:
            color = 'C' + str(snap_index)
            bin_centers, bin_vals, bin_errs = medianZR( star_mass, Z_use, nbins=10, return_err_bars=True )
            ax_left.fill_between( bin_centers, bin_vals - bin_errs, bin_vals + bin_errs, alpha=0.15 )
            
            ax_left.scatter( star_mass, MZsR(star_mass), color=color )
            ax_left.plot( bin_centers, bin_vals, lw=3, color=color )
            
            # ax_left.scatter( masses,metals,color=color )
            
            ax_left.text( 0.05, 0.9, whichSim2Tex[sim_scatter] + ' ' + r'${\rm MZR}$', transform=ax_left.transAxes )
        
        offsets = Z_use - MZsR(star_mass)
        
        all_scatters.append( offsets )
        all_masses.append( star_mass )
        all_SFRs.append( SFR )
    
    # Next get the MZR of the other simulation (ensuring that they're the same length)
    snapshots, snap2z, BLUE_DIR = switch_sim(sim_MZR)
    print('Getting MZR')
    
    # What I want out of this is:
    #    (i) Previous scatter about the new MZR's evolution
    flipped_scatters = []
    
    for snap_index, snap in enumerate(snapshots):
        # if (snap_index) != 7:
        #     flipped_scatters.append([])
        #     continue
        currentDir = BLUE_DIR + 'data/' + 'snap%s/' %snap

        Zgas      = np.load( currentDir + 'Zgas.npy' )
        Zstar     = np.load( currentDir + 'Zstar.npy' ) 
        star_mass = np.load( currentDir + 'Stellar_Mass.npy'  )
        gas_mass  = np.load( currentDir + 'Gas_Mass.npy' )
        SFR       = np.load( currentDir + 'SFR.npy' )

        THRESHOLD = -5.00E-01
        sfms_idx = sfmscut(star_mass, SFR)

        desired_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                        (star_mass < 1.00E+01**(m_star_max)) &
                        (gas_mass  > 1.00E+01**(m_gas_min))  &
                        (sfms_idx))

        gas_mass  = gas_mass [desired_mask]
        star_mass = star_mass[desired_mask]
        SFR       = SFR      [desired_mask]
        Zstar     = Zstar    [desired_mask]
        Zgas      = Zgas     [desired_mask]

        Zstar /= Zsun
        OH     = Zgas * (zo/xh) * (1.00/16.00)

        Zgas      = np.log10(OH) + 12

        # Get rid of nans and random values -np.inf
        nonans    = ~(np.isnan(Zgas)) & ~(np.isnan(Zstar)) & (Zstar > 0.0) & (Zgas > 0.0) 

        sSFR      = SFR/star_mass

        sSFR[~(sSFR > 0.0)] = 1e-16

        star_mass = star_mass[nonans]
        sSFR      = sSFR     [nonans]
        Zstar     = Zstar    [nonans]
        Zgas      = Zgas     [nonans]

        gas_mass      = np.log10(gas_mass)
        star_mass     = np.log10(star_mass)
        Zstar         = np.log10(Zstar)

        if (STARS_OR_GAS == "GAS"):
            Z_use = Zgas
        elif (STARS_OR_GAS == "STARS"):
            Z_use = Zstar
            
        if plot:
            color = 'C' + str(snap_index)
            bin_centers, bin_vals, bin_errs = medianZR( star_mass, Z_use, nbins=10, return_err_bars=True )
            ax_right.fill_between( bin_centers, bin_vals - bin_errs, bin_vals + bin_errs, alpha=0.15 )
            ax_right.plot( bin_centers, bin_vals, lw=3, color=color )
                    
            ax_right.text( 0.05, 0.9, whichSim2Tex[sim_MZR] + ' ' + r'${\rm MZR}$', transform=ax_right.transAxes )
        
        masses, metals = getMedians(star_mass,Z_use)
        
        filter_fit_nans = ~(np.isnan(metals))
        
        masses = masses[filter_fit_nans]
        metals = metals[filter_fit_nans]
        
        # Create function to interpolate the MZR
        MZsR = interp1d(masses,metals,fill_value='extrapolate')
        # Interpolate the MZR at the scatter simulation masses
        pred_metals = MZsR( all_masses[snap_index] )
        # pred_metals = MZsR( star_mass )
        # Add back in the scatter
        # scatters = Z_use - pred_metals
        flipped_scatters.append( pred_metals + all_scatters[snap_index] )
        # flipped_scatters.append( pred_metals + scatters )
        # all_masses.append( star_mass )
        # all_SFRs.append( SFR )
    
    # At this point, we have (from each redshift):
    #   (i)   the masses from scatter sim
    #   (ii)  the SFRs from scatter sim
    #   (iii) the metallicity offsets from scatter applied to redshift evolution of MZR sim
    
    star_mass = []
    SFR       = []
    Z_use     = []
    
    for index in range(len(snapshots)):
        # if (index) != 7:
        #     continue
        star_mass += list( all_masses[index] )
        SFR       += list( all_SFRs[index] )
        Z_use     += list( flipped_scatters[index] ) 
        
        if plot:
            color = 'C' + str(index)
            bin_centers, bin_vals, bin_errs = medianZR( all_masses[index], flipped_scatters[index],
                                                        nbins=10, return_err_bars=True )
            ax_center.fill_between( bin_centers, bin_vals - bin_errs, bin_vals + bin_errs, alpha=0.15 )
            ax_center.plot( bin_centers, bin_vals, lw=3, color=color )
                        
            ax_center.text( 0.05, 0.9, whichSim2Tex[sim_MZR] + ' ' + r'${\rm MZR}$',
                            transform=ax_center.transAxes )
            ax_center.text( 0.05, 0.8, whichSim2Tex[sim_scatter] + ' ' + r'${\rm scatter}$',
                            transform=ax_center.transAxes )
    
    star_mass = np.array(star_mass, dtype=np.float64)
    SFR       = np.array(SFR      , dtype=np.float64)
    Z_use     = np.array(Z_use    , dtype=np.float64)
    
    alphas = np.linspace(0,1,100)
    disp   = np.zeros( len(alphas) )
    a_s    = np.zeros( len(alphas) )
    b_s    = np.zeros( len(alphas) )
    
    for index, alpha in enumerate(alphas):
        
        muCurrent = star_mass - alpha * np.log10(SFR)

        popt   = np.polyfit( muCurrent, Z_use, 1 )

        a_s[index], b_s[index] = popt

        interp = np.polyval( popt, muCurrent )

        disp[index] = np.std( np.abs(Z_use) - np.abs(interp) )

    alpha_flipped = round( alphas[ np.argmin(disp) ], 2 )
        
    alphas_true = {
        "EAGLE":0.71,
        "ORIGINAL":0.26,
        "TNG": 0.54
    }
    
    print('\n' + '#'*100)
    print( 'Flipped = Scatter about %s applied to %s MZR evolution' %(sim_scatter, sim_MZR) )
    print( 'alpha flipped: %s' %alpha_flipped )
    print('')
    print( 'Normal = Scatter about %s applied to %s MZR evolution' %(sim_scatter, sim_scatter) )
    print( 'alpha normal: %s' %alphas_true[sim_scatter] )
    print('')
    print( 'Other = Scatter about %s applied to %s MZR evolution' %(sim_MZR, sim_MZR) )
    print( 'alpha other: %s' %alphas_true[sim_MZR] )
    print('#'*100)
    
    if plot:
        ax_left  .set_xlabel( r'$\log M_*~[M_\odot]$' )
        ax_center.set_xlabel( r'$\log M_*~[M_\odot]$' )
        ax_right .set_xlabel( r'$\log M_*~[M_\odot]$' )
        
        ax_left.set_ylabel( r'$\log({\rm O/H}) + 12 ~{\rm (dex)}$' )
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0)
        plt.savefig( BLUE + '/FMR_paper2/' + '%s_about_%s' %(sim_scatter,sim_MZR) + '.pdf',
                     bbox_inches='tight' )
    
def getMedians(mass,metals,nbins=100):
    start = np.min(mass)
    end   = np.max(mass)
    
    current = start
    
    medians = []
    masses  = []
    
    step = ( end - start ) / nbins
    
    while (current < end + step):
        
        mask = ((mass > (current)) & (mass < (current + step)))
        
        if (len(metals[mask]) > 5):
            medians.append( np.median( metals[mask] ) )
        else:
            medians.append( np.nan )
            
        masses.append( current )
    
        current += step
        
    return np.array(masses),np.array(medians)

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

def medianZR( x, y, nbins = 5, return_err_bars = False ):
    
    start = np.min(x)
    end   = np.max(x)
    
    bin_centers = np.linspace(start,end,nbins)
    binWidth = (end - start) / nbins
    
    bin_vals = np.ones(len(bin_centers))
    bin_errs = np.ones(len(bin_centers))
    
    for index, current in enumerate(bin_centers):
        mask = ((x > current - binWidth/2) & (x < current + binWidth/2))
        
        if (sum(mask) < 20):
            bin_vals[index] = np.nan
            bin_errs[index] = np.nan
        else:
            bin_vals[index] = np.nanmedian( y[mask] )
            bin_errs[index] = np.nanstd( y[mask] )
    
    if return_err_bars:
        return bin_centers, bin_vals, bin_errs
    else:
        return bin_centers, bin_vals

do(FLIP,plot=True)