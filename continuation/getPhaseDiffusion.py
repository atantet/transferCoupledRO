import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import sys
sys.path.append('../cfg/')
from coupledRO import *
from ergoInt import *
from ergoCont import *
import ergoPlot


configFile = '../cfg/coupledRO.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
getModelParam(cfg)
p["eta2"] = 0.6
p["r"] = 0.18
p["gamma"] = 0.4

dim = cfg.model.dim
dt = cfg.simulation.dt
fileFormat = cfg.general.fileFormat
if (fileFormat == 'bin'):
    readFile = np.fromfile
else:
    readFile = np.loadtxt
srcPostfix = "_%s" % (cfg.model.caseName,)

# Define diffusion matrix (constant)
# Stochastic forcing configuration
V1 = fieldNoise(np.empty((dim,)), p)
V1Mat = np.tile(V1, (dim, 1))
Q = V1Mat * V1Mat.T

# List of continuations to plot
initCont = 0.
contStep = 0.001

contAbs = np.sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
contPostfix = "_cont%04d_contStep%de%d" \
              % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                 (int(exp*1.01)))
dstPostfix = "%s_eta2%04d_r%04d_gamma%04d%s" \
             % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                int(p["r"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
                contPostfix)
poFileName = '%s/poCont/poState%s.%s' % (contDir, dstPostfix, fileFormat)
FloquetExpFileName = '%s/poExp/poExp%s.%s' \
                     % (contDir, dstPostfix, fileFormat)
FloquetVecLeftFileName = '%s/poVecLeft/poVecLeft%s.%s' \
                         % (contDir, dstPostfix, fileFormat)
FloquetVecRightFileName = '%s/poVecRight/poVecRight%s.%s' \
                         % (contDir, dstPostfix, fileFormat)

# Read periodic orbit
print 'Reading states...'
state = readFile(poFileName)
state = state.reshape(-1, dim+2)
po = state[:, :dim]
TRng = state[:, dim+1]
contRng = state[:, dim]
# Convert to relaxation rate in years
TDimRng = TRng * pdim['tadim2year']
print 'Reading Floquet exponents...'
FloquetExp = readFile(FloquetExpFileName)
FloquetExp = FloquetExp.reshape(-1, dim, 2)
FloquetExp = FloquetExp[:, :, 0] + 1j * FloquetExp[:, :, 1]
print 'Reading left Floquet vectors...'
FloquetVecLeft = readFile(FloquetVecLeftFileName)
FloquetVecLeft = FloquetVecLeft.reshape(-1, dim, dim, 2)
FloquetVecLeft = FloquetVecLeft[:, :, :, 0] \
                 + 1j * FloquetVecLeft[:, :, :, 1]
print 'Reading right Floquet vectors...'
FloquetVecRight = readFile(FloquetVecRightFileName)
FloquetVecRight = FloquetVecRight.reshape(-1, dim, dim, 2)
FloquetVecRight = FloquetVecRight[:, :, :, 0] \
                 + 1j * FloquetVecRight[:, :, :, 1]
            
dt *= 50
nOrbit = 20
contSel = np.linspace(np.min(contRng), np.max(contRng), nOrbit)
phiRng = np.empty((contSel.shape[0],))
contSelRng = np.empty((contSel.shape[0],))
for icont in np.arange(contSel.shape[0]):
    # Set coupling
    icontSel = np.argmin((contSel[icont] - contRng)**2)
    cont = contRng[icontSel]
    FE = FloquetExp[icontSel]
    FVL = np.real(FloquetVecLeft[icontSel, :, 0])
    FVR = np.real(FloquetVecRight[icontSel, :, 0])
    contSelRng[icont] = cont
    p['mu'] = cont
    T = TRng[icontSel]
    x0 = po[icontSel]
    dstPostfixMu = "%s_eta2%04d_r%04d_gamma%04d%s_mu%04d" \
                   % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                      int(p["r"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
                      contPostfix, int(p["mu"] * 1000 + 0.1))
    
    # propagate over range
    print 'Performing Floquet analysis for orbit of period ', \
        T, ' at mu = ', cont, ' from x(0) = ', x0
    nt = int(np.ceil(T / dt))
    (xt, Mts) = propagateFundamentalRange(x0, fieldRO2D, JacFieldRO2D, p,
                                          stepRK4, dt, nt)

    # Get time-dependent diffusion matrix
    Qs = np.empty((nt, dim, dim))
    for s in np.arange(nt):
        Qs[s] = Q.copy()
    
    # Get phase diffusion
    print 'Getting phase diffusion...'
    phiRng[icont] = np.abs(getPhaseDiffusion(Qs, Mts, FVL, FVR, dt))
    if np.abs(phiRng[icont]) > 1.e6:
        phiRng[icont] = np.nan
    
    # Time conversion
    Mts *= pdim['tadim2year']
    phiRng[icont] /= pdim['tadim2year']
    FloquetExp /= pdim['tadim2year']
    for eig in np.arange(dim):
        print 'Floquet exp ', eig, ' = ', FloquetExp[eig]
    print 'Phi = ', phiRng[icont], '\n'
    
    # # Plot diagnostic
    # diagnostic = diagnose(xt, p)
    # plotOrbit(diagnostic, p)
    # plt.savefig('%s/continuation/po/orbit/orbit%s.%s' \
    #             % (plotDir, dstPostfixMu, ergoPlot.figFormat),
    #             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot phase diffusion coefficient
print phiRng
lw=2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(contSelRng, phiRng, linewidth=lw)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Phi$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRng), np.max(contRng))
ax.set_ylim(0., 20.)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/po/phaseDiffusion%s.%s' \
            % (plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
