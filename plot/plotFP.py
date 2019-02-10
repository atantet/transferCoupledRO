import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/tantet/Pro/dev/Climate/Transfer/transferCoupledRO/cfg/')
from coupledRO import *
from ergoPack.ergoInt import *
from ergoPack import ergoPlot

# Reload modules with execution of any code, to avoid having to restart
# the kernel after editing timeseries_scripts
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

p["eta2"] = 0.6
p["r"] = 0.17
p["gamma"] = 0.39

dim = cfg.model.dim
fileFormat = cfg.general.fileFormat

# List of continuations to plot
initContRng = [0.]
contStepRng = [0.002]
nCont = len(initContRng)

srcPostfix = "_%s" % (cfg.model.caseName,)

# Prepare plot
fig = plt.figure(figsize=(8, 10))
xmin = cfg.continuation.contMin
xmax = cfg.continuation.contMax
ax = []
nPan = 100*(1+2*nCont) + 10 + 1
ax.append(fig.add_subplot(nPan))
for k in np.arange(nCont):
    nPan += 1
    ax.append(fig.add_subplot(nPan))
    nPan += 1
    ax.append(fig.add_subplot(nPan))

fpL = []
eigValL = []
contL = []
lsStable = ['--', '-']
lw = 2
color = 'k'
comp = 'TE'
#comp = 'hE'
for k in np.arange(nCont):
    initCont = initContRng[k]
    contStep = contStepRng[k]
    
    contAbs = np.sqrt(contStep*contStep)
    sign = contStep / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    contPostfix = "_cont%04d_contStep%de%d" \
                  % (int(initCont * 1000 + 0.1), int(mantis*1.01),
                     (int(exp*1.01)))
    dstPostfix = "%s_eta2%04d_r%04d_gamma%04d_tauExt%04d%s" \
                 % (srcPostfix, int(p["eta2"] * 1000 + 0.1),
                    int(p["r"] * 1000 + 0.1), int(p["gamma"] * 1000 + 0.1),
                    int(p["tauExt"] * 1000 + 0.1), contPostfix)
    fpFileName = '%s/fpState/fpState%s.%s' % (contDir, dstPostfix, fileFormat)
    eigValFileName = '%s/fpEigVal/fpEigValCont%s.%s' \
                     % (contDir, dstPostfix, fileFormat)

    # Read fixed point and cont
    state = readFile(fpFileName).reshape(-1, dim+1)
    fp = state[:, :dim]
    contRng = state[:, dim]
    # Read eigValenvalues
    eigVal = readFile(eigValFileName).reshape(-1, dim, 2)
    eigVal = eigVal[:, :, 0] + 1j * eigVal[:, :, 1]
    # Convert to relaxation rate in years^-1
    eigVal /= pdim['tadim2year']

    # # Bound
    # isig = contRng < 1.
    # contRng = contRng[isig]
    # fp = fp[isig]
    # eigVal = eigVal[isig]

    fpL.append(fp)
    eigValL.append(eigVal)
    contL.append(contRng)
    
    
    isStable = np.max(eigVal.real, 1) < 0
    change = np.nonzero(isStable[1:] ^ isStable[:-1])[0] + 1
    print('Change of stability at cont = ', contRng[change])
    print('Fixed point at change of instability: ', fp[change])
    print('Characteristic exponents at instability: ', eigVal[change])

    # Diagnose
    pRng = []
    for c in np.arange(contRng.shape[0]):
        p['mu'] = contRng[c]
        pRng.append(p)
    diag = diagnoseMulti(fp, pRng)

    # Plot diagnostic
    plotDiagnosticVersusMu(diag, contRng, isStable)
    plt.savefig('%s/continuation/fp/fpDiag%s.%s' \
                % (plotDir, dstPostfix, ergoPlot.figFormat),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

    # Plot diagram
    change = np.concatenate(([0], change, [contRng.shape[0]]))
    for c in np.arange(change.shape[0]-1):
        ax[0].plot(contRng[change[c]:change[c+1]],
                   diag[comp][change[c]:change[c+1]],
                   linestyle=lsStable[isStable[change[c]]],
                   color=color, linewidth=lw)

    # Plot real parts
    ax[1+2*k].plot(contRng, np.zeros((contRng.shape[0],)), '--k')
    ax[1+2*k].plot(contRng, eigVal.real, linewidth=2)
    ax[1+2*k].set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax[1+2*k].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax[1+2*k].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax[1+2*k].set_xlim(xmin, xmax)
    ax[1+2*k].set_ylim(-7., 1.)

    # Plot imaginary parts
    ax[1+2*k+1].plot(contRng, eigVal.imag, linewidth=2)
    ax[1+2*k+1].set_ylabel(r'$\Im(\lambda_i)$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax[1+2*k+1].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax[1+2*k+1].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax[1+2*k+1].set_xlim(xmin, xmax)
ax[0].set_ylabel(varName[comp], fontsize=ergoPlot.fs_latex)
ax[0].set_xlim(xmin, xmax)
plt.setp(ax[0].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax[0].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax[-1].set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)

fig.savefig('%s/continuation/fp/fpCont%s.%s' \
            % (plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

