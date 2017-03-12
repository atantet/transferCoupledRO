import numpy as np
import ergoPlot
from ergoInt import *
from ergoCont import *
import sys
sys.path.append('../cfg/')
from coupledRO import *

def getPhaseDiffusion(Mt, T, Q, FloquetExp, eigVec, eigVecLeft):
    # Check if fundamental matrix is singular
    detMT = np.abs(np.linalg.det(Mt[-1]))
    print 'det MT = ', detMT
    if detMT > 1.e-8:
        # Get correlation matrix and phase diffusion coefficient
        dim = eigVec.shape[0]
        CT = np.zeros((dim, dim))
        for t in np.arange(T):
            iMt = np.linalg.inv(Mt[t])
            CT += np.dot(np.dot(iMt, Q), iMt.T) * tdim
            #CT = np.dot(np.dot(MT, CT), MT.T)
        norm = np.dot(eigVecLeft[:, 0], eigVec[:, 0])
        print 'Norm = ', norm
        Phi = - np.dot(np.dot(eigVecLeft[:, 0], CT),
                       eigVecLeft[:, 0]) / norm / (T * tdim)
        print 'Phi = ', Phi
    else:
        print 'Singular matrix'
        Phi = np.nan
        
    return Phi

            
# Reference variables
pdim = {}
pdim['T0'] = 30. # (K) Radiative equilibrium temperature
pdim['Ts0'] = 24. # (K) Thermocline reference temperature
pdim['DeltaT'] = 1. # (K) Reference temperature difference
pdim['Hm'] = 50. # (m) Mixed-layer depth
pdim['Hs'] = 50. # (m) Steepness of the tanh
pdim['h0'] = 25. # (m) Offset of the tanh
pdim['L'] = 1.5 * 10**7 # (m) Width of the basin
pdim['Ly'] = 1. * 10**6 # (m) Meridional length
pdim['epsT'] = 1. / (150 * 24*60*60) # (s^-1) SST damping rate
pdim['c0'] = 2. # (m s^-1) Velocity of the first baroclinic Kelvin mode
pdim['aM'] = 1.3*10**(-8) # (s^-1) Rayleigh friction coefficient
pdim['tau0'] = 2.667 * 10**(-7) # (m s^-2) Reference wind stress
pdim['r'] = 1. / (500 * 24*60*60) # (s^-1) Dynamical adjustment timescale
# of the western equatorial thermocline depth by the zonally integrated
# Sverdrup meridional mass transport resulting from wind-forced Rossby waves
pdim['b'] = 4.874 * 0.9# (s) Efficiency of wind stress in driving the thermocline

pdim['tauExt'] = -0.2 * pdim['tau0'] # (m s^-2) External wind stress
pdim['w0'] = 0. # (m s^-1) Upwelling due to mixing and/or stronger clim resp
minTE = pdim['T0'] - 2 * (pdim['T0'] - pdim['Ts0'])
maxTE = pdim['T0']
minhW = -10.
maxhW = 60.

# Adimensional parameters
p = {}
p['alpha'] = pdim['epsT'] * pdim['L'] / pdim['c0']
p['eta1'] = pdim['Hm'] / pdim['Hs']
p['eta2'] = pdim['h0'] / pdim['Hs']
p['rp'] = pdim['r'] * pdim['L'] / pdim['c0']
p['gam'] = pdim['b'] * pdim['L'] * pdim['tau0'] / pdim['Hm']
p['xs0'] = (pdim['Ts0'] - pdim['T0']) / pdim['DeltaT']
p['tauExtp'] = pdim['tauExt'] / pdim['tau0']
p['w0p'] = pdim['w0'] * pdim['L'] / (pdim['Hm'] * pdim['c0'])

p['von'] = 1.
p['deltas'] = 1.
#p['mu'] = .4

# Stochastic forcing configuration
sigmaInf2 = 0.1
p['epsh'] = 100.
p['sigmah'] = np.sqrt(sigmaInf2 * 2*p['epsh'])

# Config model
dim = 3
day2sec = 24 * 60 * 60
year2day = 365
tdim = pdim['L'] / pdim['c0']

# Config simulation
spinup = 50. * year2day * day2sec / tdim # 50 years
dt = 1. * day2sec / tdim  # 1 day
tdimYear = dt * tdim / day2sec / year2day
ntSpinup = int(spinup / dt + 0.1)

#muRng = np.arange(0., 0.361, 0.03)
#muRng = np.arange(0.5, 0.8, 0.01)
muRng = np.array([0.])

# Define diffusion matrix (constant)
V1 = fieldNoise(np.empty((dim,)), p)
V1Mat = np.tile(V1, (dim, 1))
Q = V1Mat * V1Mat.T

TEFP = np.empty((muRng.shape[0],))
phiRng = np.empty((muRng.shape[0],))
for imu in np.arange(muRng.shape[0]):
    # Set coupling
    p['mu'] = muRng[imu]
    print 'mu = ', p['mu']

    # Get initial state from converged simulation
    x0s = np.array([-1., 0., 0.])
    xt = propagate(x0s, fieldRO2D, p, stepRK4, dt, ntSpinup)
    #xt = xt[:40]
    time = np.arange(xt.shape[0]) * tdimYear
    xtCut = xt[xt.shape[0]/2:]
    timeCut = np.arange(xtCut.shape[0]) * tdimYear
    x0 = xt[-1]

    # Get period
    (T, dist) = getPeriod(xtCut, step=1)

    # Test for fixed point, periodic or aperiodic orbit
    print 'dist = ', dist
    if np.abs(T) < 1.e-8:
        print 'Fixed point'
        
    elif dist > 1.e-5:
        print 'Not closed'
        FloquetExp = np.nan
        Phi = np.nan
        
        # Plot diagnostic
        diagnostic = diagnose(xt, p)
        plotDiagnostic(diagnostic, [minTE, maxTE], [minhW, maxhW])
        
    else:
        print 'Period = ', timeCut[T]

        # Floquet analysis
        M0 = np.eye(dim)
        (Mt, FloquetExp, eigVec, eigVecLeft) \
            = Floquet(x0, fieldRO2D, M0, JacFieldRO2D, p, stepRK4,
                      dt, T)
        
        print 'Exp 0 = ', FloquetExp[0]
        print 'Exp 1 = ', FloquetExp[1]

        # Get phase diffusion
        phiRng[imu] = getPhaseDiffusion(Mt, T, Q, FloquetExp,
                                        eigVec, eigVecLeft)

        # Plot diagnostic
        plotDiagnostic(diagnostic, [minTE, maxTE], [minhW, maxhW])

# Plot phase diffusion coefficient
lw=2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(muRng, phiRng, linewidth=lw)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Phi$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(muRng[0], muRng[-1])
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/phaseDiffusion.%s' % (plotDir, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
