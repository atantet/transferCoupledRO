import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
from ergoPack import ergoPlot
from coupledRO import *

#ergoPlot.dpi = 2000

configFile = '../cfg/coupledRO.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

L = cfg.simulation.LCut + cfg.simulation.spinup
spinup = cfg.simulation.spinup
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
dim = cfg.model.dim
dimObs = len(cfg.observable.components)
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


muRng = np.arange(0.6, 1.21, 0.06)
for imu, mu in enumerate(muRng) :
    srcPostfixSim = "_%s_mu%03d_eta2%03d_gamma%03d_r%03d_sigmahInf2%03d\
_L%d_spinup%d_dt%d_samp%d" \
    % (caseName, int(mu * 100 + 0.1),
       int(p['eta2'] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
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

    print('Readig spectrum for mu = {:.2f}...'.format(mu))
    (eigValForward,) = readSpec(eigValForwardFile)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen = np.log(eigValForward) / tau

    ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel,
                     xlim=xlimEig, ylim=ylimEig)
    plt.text(xlimEig[0]*0.2, ylimEig[1]*1.05, r'$\mu = {:.2f}$'.format(mu),
             fontsize=ergoPlot.fs_latex)
    plt.savefig('%s/spectrum/eigVal/eigVal%s.%s'\
                % (cfg.general.plotDir, postfixTau, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

