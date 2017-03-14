import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append('../cfg/')
from coupledRO import *
from ergoInt import *
from ergoCont import *
import ergoPlot
import traceback

dim = cfg.model.dim
fileFormat = cfg.general.fileFormat

# Define diffusion matrix (constant)
# Stochastic forcing configuration
V1 = fieldNoise(np.empty((dim,)), p)
V1Mat = np.tile(V1, (dim, 1))
Q = V1Mat * V1Mat.T

# List of continuations to plot
initCont = 0.
contStep = 0.001

# Parameter ranges
eta2Min = 0.5; eta2Max = 0.7; neta2 = 11
rMin = 0.1; rMax = 0.3; nr = 11
gammaMin = 0.3; gammaMax = 0.5; ngamma = 11;

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
for ieta2 in np.arange(eta2Rng.shape[0]):
    p['eta2'] = eta2Rng[ieta2]
    for ir in np.arange(rRng.shape[0]):
        p['r'] = rRng[ir]
        for igamma in np.arange(gammaRng.shape[0]):
            p['gamma'] = gammaRng[igamma]
	    print "Getting phase diffusion for eta2 = ", p["eta2"], \
	        ", r = ", p["r"], ", gamma = ", p["gamma"]
            dstPostfix = "%s_eta2%04d_r%04d_gamma%04d%s" \
                         % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                            int(p["r"] * 1000 + 0.1),
                            int(p["gamma"] * 1000 + 0.1), contPostfix)
            poFileName = '%s/poCont/poState%s.%s' \
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
rSel = 0.2
gammaSel = 0.4
nLev = 10

# Plot (mu, eta2)
# Select data points
sel = ((data[1] - rSel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[0]
contRngSel = dataSel[3]
phiRng = dataSel[4]
dstPostfixPlot = "%s_r%04d_gamma%04d%s" \
                 % (srcPostfix, int(p["r"] * 1000 + 0.1),
                    int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
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
phiRng = dataSel[4]
dstPostfixPlot = "%s_eta2%04d_gamma%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                    int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
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
phiRng = dataSel[4]
dstPostfixPlot = "%s_eta2%04d_r%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                    int(p["r"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
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


# Scaled plots
# Plot (mu, eta2)
# Select data points
sel = ((data[1] - rSel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[0]
contRngSel = dataSel[3]
phiRng = dataSel[4] * dataSel[5] / p['sigmah']**2
dstPostfixPlot = "%s_r%04d_gamma%04d%s" \
                 % (srcPostfix, int(p["r"] * 1000 + 0.1),
                    int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\eta_2$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusionScale%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, r)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[2] - gammaSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[1]
contRngSel = dataSel[3]
phiRng = dataSel[4] * dataSel[5] / p['sigmah']**2
dstPostfixPlot = "%s_eta2%04d_gamma%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                    int(p["gamma"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$r$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusionScale%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot (mu, gamma)
# Select data points
sel = ((data[0] - eta2Sel)**2 < 1.e-8) & ((data[1] - rSel)**2 < 1.e-8)
dataSel = data[:, sel]
paramRng = dataSel[2]
contRngSel = dataSel[3]
phiRng = dataSel[4] * dataSel[5] / p['sigmah']**2
dstPostfixPlot = "%s_eta2%04d_r%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                    int(p["r"] * 1000 + 0.1), contPostfix)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
levels = np.linspace(0., np.max(np.abs(phiRng)), nLev)
triang = tri.Triangulation(contRngSel, paramRng)
plt.triplot(triang, lw=0.5, color='white')
cf = ax.tricontourf(triang, np.abs(phiRng), levels, cmap=cm.hot_r)
cbar = plt.colorbar(cf, ticks=np.linspace(0., int(levels[-1]),
                               int(levels[-1]) + 1, dtype=int))
cbar.ax.set_ylabel(r'$|\phi|$', fontsize=ergoPlot.fs_latex)
plt.setp(cbar.ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\gamma$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRngSel), np.max(contRngSel))
ax.set_ylim(np.min(paramRng), np.max(paramRng))
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusionScale%s.%s' \
            % (plotDir, dstPostfixPlot, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
