import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../cfg/')
from coupledRO import *
from ergoInt import *
from ergoCont import *
import ergoPlot

p["eta2"] = 0.6
p["r"] = 0.17
p["gamma"] = 0.39

dim = cfg.model.dim
fileFormat = cfg.general.fileFormat
dt = cfg.simulation.dt

# Define diffusion matrix (constant)
# Stochastic forcing configuration
V1 = fieldNoise(np.empty((dim,)), p)
V1Mat = np.tile(V1, (dim, 1))
Q = V1Mat * V1Mat.T

# List of continuations to plot
initCont = 0.
contStep = 0.01
initContFP= 0.
contStepFP = 0.002

# Define file names for limit cycle
contAbs = np.sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
contPostfix = "_cont%04d_contStep%de%d" \
              % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                 (int(exp*1.01)))
dstPostfix = "%s_eta2%04d_r%04d_gamma%04d%s_dt%d" \
             % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                int(p["r"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
                contPostfix, int(np.round(-np.log10(dt)) + 0.1))

poFileName = '%s/poState/poState%s.%s' % (contDir, dstPostfix, fileFormat)
FloquetExpFileName = '%s/poExp/poExp%s.%s' \
                     % (contDir, dstPostfix, fileFormat)
vecLeftFileName = '%s/poVecLeft/poVecLeft%s.%s' \
                  % (contDir, dstPostfix, fileFormat)
vecRightFileName = '%s/poVecRight/poVecRight%s.%s' \
                   % (contDir, dstPostfix, fileFormat)
phiFileName = '%s/phase/phaseDiffusion%s.%s' \
              % (contDir, dstPostfix, fileFormat)

# Define file names for stationary point
contAbsFP = np.sqrt(contStepFP * contStepFP)
signFP = contStepFP / contAbsFP
expFP = np.log10(contAbsFP)
mantisFP = signFP * np.exp(np.log(contAbsFP) / expFP)
contPostfixFP = "_cont%04d_contStep%de%d" \
                % (int(initContFP * 1000 + 0.1), int(mantisFP*1.01),
                   (int(expFP*1.01)))
dstPostfixFP = "%s_eta2%04d_r%04d_gamma%04d%s" \
             % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                int(p["r"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
                contPostfixFP)
fpFileName = '%s/fpState/fpState%s.%s' % (contDir, dstPostfixFP, fileFormat)
eigValFileName = '%s/fpEigVal/fpEigValCont%s.%s' \
                 % (contDir, dstPostfixFP, fileFormat)
# Read fixed point and cont
state = readFile(fpFileName).reshape(-1, dim+1)
fp = state[:, :dim]
contRngFP = state[:, dim]
# Read eigValenvalues
eigValJacRng = readFile(eigValFileName).reshape(-1, dim, 2)
eigValJacRng = eigValJacRng[:, :, 0] + 1j * eigValJacRng[:, :, 1]
# Convert to relaxation rate in years^-1
eigValJacRng /= pdim['tadim2year']


# Read periodic orbit
state = readFile(poFileName)
state = state.reshape(-1, dim+2)
poRng = state[:, :dim]
contRng = state[:, dim]
TRng = state[:, dim+1]
TDimRng = TRng * pdim['tadim2year']
data = readFile(phiFileName).reshape(-1, 2)
contSelRng = data[:, 0]
phiRng = data[:, 1] / pdim['tadim2year']

# Read Floquet exponents
FloquetExp = readFile(FloquetExpFileName).reshape(-1, dim, 2)
FloquetExp = FloquetExp[:, :, 0] + 1j * FloquetExp[:, :, 1]
FloquetExp /= pdim['tadim2year']

# Read Floquet vectors
vecLeft = readFile(vecLeftFileName).reshape(-1, dim, dim, 2)
vecLeft = vecLeft[:, :, :, 0] + 1j * vecLeft[:, :, :, 1]
vecRight = readFile(vecRightFileName).reshape(-1, dim, dim, 2)
vecRight = vecRight[:, :, :, 0] + 1j * vecRight[:, :, :, 1]

# Select
scale = .4

nMax = 7
l1Max = 20
xlimEig = [-2., 0.01]
ylimEig = [-5., 5.]
xlabelEig = r'$\Re(\lambda_k)$'
ylabelEig = r'$\Im(\lambda_k)$'
phiRng /= 0.1 / p["sigmahInf2"]
contSelRng = contSelRng
phiRng = phiRng
nCont = contSelRng.shape[0]
FE1Rng = np.empty((nCont,))
eigValJac0Rng = np.empty((nCont,), dtype=complex)
for c in np.arange(nCont):
    argm = np.argmin(np.abs(contRng - contSelRng[c]))
    argmFP = np.argmin(np.abs(contRngFP - contSelRng[c]))
    p['mu'] = contRng[argm]
    po = poRng[argm]
    T = TDimRng[argm]
    omega = 2 * np.pi / T
    FE = np.real(FloquetExp[argm])
    FVL = np.real(vecLeft[argm])
    FVR = np.real(vecRight[argm])
    FE1Rng[c] = FE[1]
    phi = phiRng[c]
    eigValJac = eigValJacRng[argmFP]
    isort = np.argsort(np.abs(eigValJac.real))
    eigValJac = eigValJac[isort]
    eigValJac0Rng[c] = eigValJac[0]

    # Get eigenvalues associated with stationary point
    eigValFP = np.empty((l1Max, l1Max), complex)
    for l1 in np.arange(l1Max):
        mm = l1
        if eigValJac[0].real > 0:
            mm = -(l1 + 1)
        for l2 in np.arange(l1Max):
            nn = l2
            if eigValJac[1].real > 0:
                nn = -(l2 + 1)
            eigValFP[l1, l2] = mm * eigValJac[0] + nn * eigValJac[1]
    eigValFP = eigValFP.flatten()
            
    # Get eigenvalues associated with limit cycle
    eigVal = np.empty((2*nMax+1, l1Max), complex)
    for n in np.arange(-nMax, nMax + 1):
        for l1 in np.arange(l1Max):
            eigVal[n, l1] = l1 * FE[1] - phi * n**2 + 1j * n * omega
    eigVal = eigVal.flatten()

    fig = ergoPlot.plotEig(eigVal, eigValFP, xlim=xlimEig, ylim=ylimEig,
                           xlabel=xlabelEig, ylabel=ylabelEig,
                           condition='k')
    fig.savefig('%s/spectrum/eigval/eigval_mu%03d%s.%s'\
                % (cfg.general.plotDir, int(p['mu'] * 100 + 0.1),
                   dstPostfix, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.plot(contSelRng[1:-2], FE1Rng[1:-2], '-b')
ax2.plot(contSelRng[1:-2], phiRng[1:-2], '-r')
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.setp(ax2.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\alpha_1$', fontsize=ergoPlot.fs_latex)
ax2.set_ylabel(r'$\phi$', fontsize=ergoPlot.fs_latex)

# Plot stationary point
nContFP = contRngFP.shape[0]
step = 10
for c in np.arange(0, nContFP, step):
    p['mu'] = contRngFP[c]
    eigValJac = eigValJacRng[c]
    isort = np.argsort(np.abs(eigValJac.real))
    eigValJac = eigValJac[isort]

    # Get eigenvalues associated with stationary point
    eigValFP = np.empty((l1Max, l1Max), complex)
    for l1 in np.arange(l1Max):
        mm = l1
        if eigValJac[0].real > 0:
            mm = -(l1 + 1)
        for l2 in np.arange(l1Max):
            nn = l2
            if eigValJac[1].real > 0:
                nn = -(l2 + 1)
            eigValFP[l1, l2] = mm * eigValJac[0] + nn * eigValJac[1]
    eigValFP = eigValFP.flatten()
            
    fig = ergoPlot.plotEig(eigValFP, xlim=xlimEig, ylim=ylimEig,
                           xlabel=xlabelEig, ylabel=ylabelEig,
                           condition='k', marker='x')
    fig.savefig('%s/spectrum/eigval/eigval_mu%03d%s.%s'\
                % (cfg.general.plotDir, int(p['mu'] * 100 + 0.1),
                   dstPostfixFP, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
