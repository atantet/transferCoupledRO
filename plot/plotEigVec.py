import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pylibconfig2
from ergoPack import ergoPlot
sys.path.append('../cfg/')
from coupledRO import *

#figFormat = ergoPlot.figFormat
figFormat = 'png'

direct = 'Forward'
#direct = 'Backward'

configFile = '../cfg/coupledRO.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
os.system('mkdir %s/spectrum/ 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir %s/spectrum/eigval/ 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir %s/spectrum/eigvec/ 2> /dev/null' % cfg.general.plotDir)

compNames = (r'$T_E~({}^\circ C)$', r'$h_W~(m)$')
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
(ev_xlabel, ev_ylabel) = compNames
#xlimEV = [-20., 20.]
#ylimEV = [-20., 20.]


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
srcPostfixSim = "_%s_mu%03d_eta2%03d_gamma%03d_r%03d_sigmahInf2%03d\
_L%d_spinup%d_dt%d_samp%d" \
% (caseName, int(p["mu"] * 100 + 0.1),
   int(p["eta2"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
   int(p["r"] * 1000 + 0.1),
   int(p["sigmahInf2"] * 1000 + 0.1), int(L + 0.1),
   int(spinup + 0.1),
   int(-np.round(np.log10(cfg.simulation.dt)) + 0.1),
   printStepNum)
postfix = "%s_nTraj%d%s" % (srcPostfixSim, cfg.sprinkle.nTraj,
                            gridPostfix)
postfixTau = "%s_tau%03d" % (postfix, int(tau * 100 + 0.1))

# Read grid
gridFile = '%s/grid/grid_%s%s.txt' \
           % (cfg.general.resDir, caseName, gridPostfix)
coord = ergoPlot.readGrid(gridFile, dimObs)
xyz = np.zeros((coord[0].shape[0], 3))
xyz[:, 0] = coord[0]
xyz[:, 1] = coord[1]
diag = diagnose(xyz, p)
X, Y = np.meshgrid(diag['TE'], diag['hW'])
X, Y = X.T, Y.T


# Define file names
eigValFile = '%s/eigval/eigval%s_nev%d%s.%s' \
             % (cfg.general.specDir, direct, cfg.spectrum.nev, postfixTau,
                cfg.general.fileFormat)
eigVecFile = '%s/eigvec/eigvec%s_nev%d%s.%s' \
             % (cfg.general.specDir, direct, cfg.spectrum.nev, postfixTau,
             cfg.general.fileFormat)
initDistFile = '%s/transfer/initDist/initDist%s.%s' \
           % (cfg.general.resDir, postfix, cfg.general.fileFormat)
maskFile = '%s/transfer/mask/mask%s.%s' \
           % (cfg.general.resDir, postfix, cfg.general.fileFormat)

# Read stationary distribution
if initDistFile is not None:
    if cfg.general.fileFormat == 'bin':
        initDist = np.fromfile(initDistFile, float)
    else:
        initDist = np.loadtxt(initDistFile, float)
else:
    initDist = None
    
# Read mask
if maskFile is not None:
    if cfg.general.fileFormat == 'bin':
        mask = np.fromfile(maskFile, np.int32)
    else:
        mask = np.loadtxt(maskFile, np.int32)
else:
    mask = np.arange(N)
NFilled = np.max(mask[mask < N]) + 1

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print('Readig spectrum for tau = {:.3f}...'.format(tau))
if direct == 'Forward':
    (eigVal, eigVec) \
        = ergoPlot.readSpectrum(eigValForwardFile=eigValFile,
                                eigVecForwardFile=eigVecFile)
if direct == 'Backward':
    (eigVal, eigVec) \
        = ergoPlot.readSpectrum(eigValBackwardFile=eigValFile,
                                eigVecBackwardFile=eigVecFile)
eigVec = eigVec.T
nev = eigVal.shape[0]
eigValGen = np.log(eigVal) / (tau * pdim['tadim2year'])

# Sort
isort = np.argsort(-np.abs(eigVal))
eigVal = eigVal[isort]
eigVec = eigVec[:, isort]

# Plot eigenvectors of transfer operator
alpha = 0.0
ss = 4
os.system('mkdir %s/spectrum/eigvec 2> /dev/null' % cfg.general.plotDir)
#eigVec *= -1.
# eigVec[:, 0]  /= eigVec[:, 0].sum()
ampMin = 0.
ampMax = 0.020
# nlevAmp = int((ampMax - ampMin) * 100 + 0.1) + 1
nlevAmp = 11
# csfilter = 0.
csfilter = 1
csfmt = '%1.1f'
# for ev in np.arange(cfg.spectrum.nEigVecPlot):
for ev in np.arange(1, cfg.spectrum.nEigVecPlot):
    if ev == 0:
        cmap = cm.hot_r
        positive=True
    else:
        cmap = cm.RdBu_r
        positive=False
    print('Plotting eigenvector {:d}...'.format(ev + 1))
    # vec = eigVec[:, ev].real
    vec = eigVec[:, ev]
    # if (initDist is not None) and (ev == 0):
    #     vec *= initDist
    # ergoPlot.plotEigVec(X, Y, vec, mask, xlabel=ev_xlabel, ylabel=ev_ylabel,
    #                     alpha=alpha, cmap=cmap, positive=(ev == 0))
    # #, xlim=xlimEV, ylim=ylimEV)
    # dstFile = '%s/spectrum/eigvec/eigvec%sReal_ev%03d%s.%s' \
    #           % (cfg.general.plotDir, direct, ev + 1, postfixTau, figFormat)
    ergoPlot.plotEigVecPolarCombine(X, Y, vec, mask,
                                    xlabel=ev_xlabel, ylabel=ev_ylabel,
                                    alpha=alpha, cmap=cmap, csfmt=csfmt,
                                    ampMin=ampMin, ampMax=ampMax, nlevAmp=nlevAmp)
    #, xlim=xlimEV, ylim=ylimEV)
    dstFile = '%s/spectrum/eigvec/eigvec%sPolar_ev%03d%s.%s' \
              % (cfg.general.plotDir, direct, ev + 1, postfixTau, figFormat)
    plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
