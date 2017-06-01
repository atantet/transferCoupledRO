import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot
sys.path.append('../cfg/')
from coupledRO import *

configFile = '../cfg/coupledRO.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
os.system('mkdir %s/spectrum/ 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir %s/spectrum/eigval/ 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir %s/spectrum/eigvec/ 2> /dev/null' % cfg.general.plotDir)

L = cfg.simulation.LCut + cfg.simulation.spinup
spinup = cfg.simulation.spinup
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
dim = cfg.model.dim
obsComp = np.array(cfg.observable.components)
dimObs = obsComp.shape[0]
nProc = ''
if (hasattr(cfg.sprinkle, 'nProc')):
    nProc = '_nProc' + str(cfg.sprinkle.nProc)

N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    if (hasattr(cfg.grid, 'gridLimitsLow')
        & hasattr(cfg.grid, 'gridLimitsHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.gridLimitsLow[d],
                                        cfg.grid.gridLimitsHigh[d])
    else:
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.sprinkle.minInitState[d],
                                        cfg.sprinkle.maxInitState[d])
        
readSpec = ergoPlot.readSpectrum
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'


muRng = np.arange(0.8, 1.041, 0.02)
phiRng = np.empty((muRng.shape[0],))
for imu in np.arange(muRng.shape[0]):
    mu = muRng[imu]

    srcPostfixSim = "_%s_mu%03d_eta2%03d_gamma%03d_r%03d_sigmahInf2%03d\
_L%d_spinup%d_dt%d_samp%d" \
    % (caseName, int(mu * 100 + 0.1),
       int(p["eta2"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
       int(p["r"] * 1000 + 0.1),
       int(p["sigmahInf2"] * 1000 + 0.1), int(L + 0.1),
       int(spinup + 0.1),
       int(-np.round(np.log10(cfg.simulation.dt)) + 0.1),
       printStepNum)
    postfix = "%s_nTraj%d%s" % (srcPostfixSim, cfg.sprinkle.nTraj,
                                gridPostfix)
    postfixTau = "%s_tau%03d" % (postfix, int(tau * 100 + 0.1))
    
    # Define file names
    eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfixTau,
                           cfg.general.fileFormat)

    print 'Readig spectrum for mu = %.2f...' % mu
    (eigVal,) = readSpec(eigValForwardFile, makeBiorthonormal=True)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen = np.log(eigVal) / (tau * pdim['tadim2year'])
    phiRng[imu] = -eigValGen[1].real

# Parameter ranges
eta2Min = 0.5; eta2Max = 0.7; neta2 = 21
rMin = 0.1; rMax = 0.3; nr = 21
gammaMin = 0.3; gammaMax = 0.5; ngamma = 21;

# List of continuations to plot
initCont = 0.
contStep = 0.01
contAbs = np.sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
contPostfix = "_cont%04d_contStep%de%d" \
              % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                 (int(exp*1.01)))

print "Getting phase diffusion for eta2 = ", p["eta2"], \
    ", r = ", p["r"], ", gamma = ", p["gamma"]
dstPostfix = "%s_eta2%04d_r%04d_gamma%04d%s_dt%d" \
             % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                int(p["r"] * 1000 + 0.1),
                int(p["gamma"] * 1000 + 0.1), contPostfix,
                int(np.round(-np.log10(cfg.simulation.dt)) + 0.1))
poFileName = '%s/poState/poState%s.%s' \
             % (contDir, dstPostfix, cfg.general.fileFormat)
phiFileName = '%s/phase/phaseDiffusion%s.%s' \
              % (contDir, dstPostfix, cfg.general.fileFormat)

try:
    X = readFile(phiFileName).reshape(-1, 2)
    contRngSel = X[:, 0]
    phiRngExp = X[:, 1]
    # Read orbit information
    state = readFile(poFileName)
    state = state.reshape(-1, dim+2)
    contRng = state[:, dim]
    # Scale with time
    phiRngExp /= pdim['tadim2year']
except IOError:
    traceback.print_exc()
    pass

# Adapt to noise level (0.1 for expansion)
phiRngExp /= 0.1 / p["sigmahInf2"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(muRng, phiRng, '-b', linewidth=2)
ax.plot(contRngSel, phiRngExp, '-r', linewidth=2)
ax.set_xlim(muRng[0], muRng[-1])
#ax.set_xlim(np.min((np.min(contRng), muRng[0])),
#            np.max((np.max(contRng), muRng[-1])))
ax.set_ylim(0., 0.015)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\phi$', fontsize=ergoPlot.fs_latex)
plt.savefig('%s/spectrum/eigVal/phiVSmuCmp%s.%s'\
            % (cfg.general.plotDir, postfixTau, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
