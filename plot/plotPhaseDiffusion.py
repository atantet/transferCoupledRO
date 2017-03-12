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
p["eta2"] = 0.5
p["r"] = 0.1
p["gamma"] = 0.3

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
phiFileName = '%s/phase/phaseDiffusion%s.%s' \
              % (contDir, dstPostfix, fileFormat)

# Read periodic orbit
print 'Reading states...'
# state = readFile(poFileName)
# state = state.reshape(-1, dim+2)
# po = state[:, :dim]
# TRng = state[:, dim+1]
# contRng = state[:, dim]
# Convert to relaxation rate in years
# TDimRng = TRng * pdim['tadim2year']
print 'Phase diffusion coefficient...'
data = readFile(phiFileName).reshape(-1, 2)
contSelRng = data[:, 0]
phiRng = data[:, 1]
print 'phi = ', np.abs(phiRng)
            
# Plot phase diffusion coefficient
lw=2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(contSelRng, np.abs(phiRng), linewidth=lw)
ax.set_xlabel(r'$\mu$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Phi$', fontsize=ergoPlot.fs_latex)
ax.set_xlim(np.min(contRng), np.max(contRng[-1]))
#ax.set_ylim(0., 5.)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/continuation/phase/phaseDiffusion%s.%s' \
            % (plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
