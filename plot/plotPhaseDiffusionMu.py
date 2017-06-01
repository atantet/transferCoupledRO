import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import sys
sys.path.append('../cfg/')
from coupledRO import *
from ergoInt import *
from ergoCont import *
import ergoPlot
import traceback
from scipy.interpolate import interp1d

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

# Parameter ranges
eta2Min = 0.5; eta2Max = 0.7; neta2 = 21
rMin = 0.1; rMax = 0.3; nr = 21
gammaMin = 0.3; gammaMax = 0.5; ngamma = 21;

contAbs = np.sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
contPostfix = "_cont%04d_contStep%de%d" \
              % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                 (int(exp*1.01)))

# Loop over parameters 
eta2Rng = np.linspace(eta2Min, eta2Max, neta2)
rRng = np.linspace(rMin, rMax, nr)
gammaRng = np.linspace(gammaMin, gammaMax, ngamma)
data = np.empty((6, 0))
contInterp = np.linspace(0., 7., 100)
for ieta2 in np.arange(eta2Rng.shape[0]):
    p['eta2'] = eta2Rng[ieta2]
    for ir in np.arange(rRng.shape[0]):
        p['r'] = rRng[ir]
        for igamma in np.arange(gammaRng.shape[0]):
            p['gamma'] = gammaRng[igamma]
	    print "Getting phase diffusion for eta2 = ", p["eta2"], \
	        ", r = ", p["r"], ", gamma = ", p["gamma"]
            dstPostfix = "%s_eta2%04d_r%04d_gamma%04d%s_dt%d" \
                         % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                            int(p["r"] * 1000 + 0.1),
                            int(p["gamma"] * 1000 + 0.1), contPostfix,
                            int(np.round(-np.log10(dt)) + 0.1))
            poFileName = '%s/poState/poState%s.%s' \
                         % (contDir, dstPostfix, fileFormat)
            phiFileName = '%s/phase/phaseDiffusion%s.%s' \
                          % (contDir, dstPostfix, fileFormat)
            FloquetExpFileName = '%s/poExp/poExp%s.%s' \
                                 % (contDir, dstPostfix, fileFormat)

            try:
                X = readFile(phiFileName).reshape(-1, 2)
                contRngSel = X[:, 0]
                phiRng = X[:, 1]
                if len(phiRng) == 0:
                    continue
                # Read orbit information
                state = readFile(poFileName)
                state = state.reshape(-1, dim+2)
                contRng = state[:, dim]
                # po = state[:, :dim]
                # TRng = state[:, dim+1]
                # Convert to relaxation rate in years
                # TDimRng = TRng * pdim['tadim2year']
                # Read Floquet exponents
                FloquetExp = readFile(FloquetExpFileName)
                FloquetExp = FloquetExp.reshape(-1, dim, 2)
                FloquetExp = FloquetExp[:, :, 0] + 1j * FloquetExp[:, :, 1]
                # Remove nans
                FloquetExp[np.isnan(FloquetExp)] \
                    = np.min(FloquetExp.real[~np.isnan(FloquetExp)])
                # Select parameters
                FE2 = np.empty(contRngSel.shape)
                for c in np.arange(contRngSel.shape[0]):
                    argm = np.argmin(np.abs(contRng - contRngSel[c]))
                    FE2[c] = np.real(FloquetExp[argm][1])
                # Scale with time
                phiRng /= pdim['tadim2year']
                FE2 /= pdim['tadim2year']
                fphi = interp1d(contRngSel, phiRng)
                fFE2 = interp1d(contRngSel, FE2)
                datai = np.concatenate((np.ones((contRngSel.shape[0],)) \
                                        * p['eta2'],
                                        np.ones((contRngSel.shape[0],)) \
                                        * p['r'],
                                        np.ones((contRngSel.shape[0],)) \
                                        * p['gamma'],
                                        contRngSel, phiRng,
                                        FE2)).reshape(6, -1)
                data = np.concatenate((data, datai), 1)
            except IOError:
                traceback.print_exc()
                pass


            
eta2Sel = 0.6
rSel = 0.17
gammaSel = 0.39
nLev = 20
vmax = 5.
lwTriangle = 0.1

# Plot (mu, eta2)
# Select data points
sel = ((data[1] - rSel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[0]
contRngSel = dataSel[3]
phiRngSel = dataSel[4]
phiRngSel[phiRngSel > vmax] = vmax
dstPostfixPlot = "%s_eta2%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
#levels = np.linspace(np.min(phiRngSel)*1.01, np.max(phiRngSel)*1.01, nLev)
levels = np.linspace(0.1,1,10)
levels = np.concatenate((levels[:-1],np.linspace(1.,10, 10)))
#levels = np.concatenate((levels[:-1],np.linspace(10,100,10)))
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r,
                    norm = LogNorm())
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$\phi$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\eta_2$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, r)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[1]
contRngSel = dataSel[3]
phiRngSel = dataSel[4]
phiRngSel[phiRngSel > vmax] = vmax
dstPostfixPlot = "%s_r%04d%s" \
                 % (srcPostfix, int(p["r"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
#levels = np.linspace(np.min(phiRngSel)*1.01, np.max(phiRngSel)*1.01, nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r,
                    norm = LogNorm())
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$\phi$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$r$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, gamma)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[1] - rSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[2]
contRngSel = dataSel[3]
phiRngSel = dataSel[4]
phiRngSel[phiRngSel > vmax] = vmax
dstPostfixPlot = "%s_gamma%04d%s" \
                 % (srcPostfix, int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
#levels = np.linspace(np.min(phiRngSel)*1.01, np.max(phiRngSel)*1.01, nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r,
                    norm = LogNorm())
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$\phi$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\gamma$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


# FE2
# Plot (mu, eta2)
# Select data points
sel = ((data[1] - rSel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[0]
contRngSel = dataSel[3]
phiRngSel = -dataSel[5]
dstPostfixPlot = "%s_eta2%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(phiRngSel), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r)
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$|\lambda_{-}|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\eta_2$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/FE2%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, r)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[1]
contRngSel = dataSel[3]
phiRngSel = -dataSel[5]
dstPostfixPlot = "%s_r%04d%s" \
                 % (srcPostfix, int(p["r"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(phiRngSel), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r)
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$|\lambda_{-}|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$r$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/FE2%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, gamma)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[1] - rSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[2]
contRngSel = dataSel[3]
phiRngSel = -dataSel[5]
dstPostfixPlot = "%s_gamma%04d%s" \
                 % (srcPostfix, int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(phiRngSel), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=lwTriangle, color='white')
cf = ax.tricontourf(triang, phiRngSel, levels, cmap=cm.magma_r)
cbar = plt.colorbar(cf)
cbar.ax.set_ylabel(r'$|\lambda_{-}|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\gamma$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/FE2%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
