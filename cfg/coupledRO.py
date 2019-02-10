import os
import numpy as np
import pylibconfig2
import matplotlib.pyplot as plt
from ergoPack import ergoPlot


pdim = {}
p = {}

# Time conversion
day2sec = 24. * 60 * 60
year2day = 365.

# Variable names
varName = {}
varName['TE'] = r'$T_E$'
varName['Ts'] = r'$T_s$'
varName['hW'] = r'$h_W$'
varName['hE'] = r'$h_E$'
varName['tau'] = r'$\tau$'
varName['w'] = r'$w$'
varName['v'] = r'$v$'


def getModelParam():
  # Scales
  pdim['L'] = cfg.model.L # (m) Width of the basin
  pdim['Ly'] = cfg.model.Ly # (m) Meridional length
  pdim['Hs'] = cfg.model.Hs # (m s^-2) External wind stress
  pdim['c0'] = cfg.model.c0 # (m s^-1) Vel. of the 1st baroclin Kelvin mode
  pdim['DeltaT'] = cfg.model.DeltaT # (K) Reference temperature difference
  pdim['T0'] = cfg.model.T0 # (K) Radiative equilibrium temperature
  pdim['tau0'] = cfg.model.tau0 # (K) Radiative equilibrium temperature
  pdim['t'] = pdim['L'] / pdim['c0']

  if cfg.model.adim:
    # Adimensional parameters
    p['alpha'] = cfg.model.alpha
    p['deltas'] = cfg.model.deltas
    p['eta1'] = cfg.model.eta1
    p['eta2'] = cfg.model.eta2
    p['gamma'] = cfg.model.gamma
    p['r'] = cfg.model.r
    p['xs0'] = cfg.model.xs0
    p['tauExt'] = cfg.model.tauExt
    p['w0'] = cfg.model.w0
    p['epsh'] = cfg.model.epsh
    p['sigmahInf2'] = cfg.model.sigmahInf2
    p['sigmah'] = np.sqrt(cfg.model.sigmahInf2 * 2 * p['epsh'])
         
    # Dimensional parameters
    pdim['Ts0'] = p['xs0'] * pdim['DeltaT'] + pdim['T0']
    pdim['epsT'] = p['alpha'] / pdim['t']
    pdim['Hm'] = p['eta1'] * pdim['Hs']
    pdim['h0'] = p['eta2'] * pdim['Hs']
    pdim['r'] = p['r'] / pdim['t']
    pdim['b'] = p['gamma'] / pdim['L'] / pdim['tau0'] * pdim['Hm']
    pdim['tauExt'] = p['tauExt'] * pdim['tau0']
    pdim['w0'] = p['w0'] / pdim['t'] * pdim['Hm']
    pdim['epsh'] = p['epsh'] / pdim['t']
  else:
    # Dimensional parameters
    pdim['Ts0'] = cfg.model.Ts0 # (K) Thermocline reference temperature
    pdim['Hm'] = cfg.model.Hm # (m) Mixed-layer depth
    pdim['h0'] = cfg.model.h0
    pdim['tau0'] = cfg.model.tau0 # (m s^-2) Reference wind stress
    pdim['tauExt'] = cfg.model.tauExt # (m s^-2) External wind stress
    pdim['epsT'] = cfg.model.epsT 
    pdim['r'] = cfg.model.r
    pdim['b'] = cfg.model.b
    pdim['w0'] = cfg.model.w0
    pdim['epsh'] = cfg.model.epsh
    
    # Adimensional parameters
    p['alpha'] = pdim['epsT'] * pdim['t']
    p['eta1'] = pdim['Hm'] / pdim['Hs']
    p['eta2'] = pdim['h0'] / pdim['Hs']
    p['r'] = pdim['r'] * pdim['t']
    p['gamma'] = pdim['b'] * pdim['L'] * pdim['tau0'] / pdim['Hm']
    p['xs0'] = (pdim['Ts0'] - pdim['T0']) / pdim['DeltaT']
    p['tauExt'] = pdim['tauExt'] / pdim['tau0']
    p['w0'] = pdim['w0'] * pdim['t'] / pdim['Hm']
    p['epsh'] = pdim['epsh'] * pdim['t']
    p['deltas'] = 1.
    
  if (hasattr(cfg.model, 'deltas')):
    p['deltas'] = cfg.model.deltas
  if (hasattr(cfg.model, 'mu')):
    p['mu'] = cfg.model.mu
  if (hasattr(cfg.model, 'sigmahInf2')):
    p['sigmah'] = np.sqrt(cfg.model.sigmahInf2 * 2 * p['epsh'])
    
  pdim['tadim2year'] = pdim['t'] / day2sec / year2day


configFile = '../cfg/coupledRO.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
getModelParam()
if (cfg.general.fileFormat == 'bin'):
    readFile = np.fromfile
else:
    readFile = np.loadtxt
srcPostfix = "_%s" % (cfg.model.caseName,)

# Directories
resDir = cfg.general.resDir
contDir = '%s/continuation' % resDir
plotDir = cfg.general.plotDir
os.system("mkdir %s/continuation/fp/ 2> /dev/null" % plotDir)
os.system("mkdir %s/continuation/po/ 2> /dev/null" % plotDir)
os.system("mkdir %s/continuation/phase/ 2> /dev/null" % plotDir)


# Nonlinearity
# def nl(x):
#     return (x - x**3 / 3)
# def dnl(x):
#     return 1. - x**2
def nl(x):
  return np.tanh(x)
def dnl(x):
  return 1. - np.tanh(x)**2

def H(x):
    return ((np.sign(x) + 1) / 2)


def fieldRO2D(X, p, dummy=None):
    (x, y, z) = X
    tau = p['tauExt'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0']
    mv = p['deltas'] * tau
    yp = p['eta1'] * z + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))
    
    f = np.array([-p['alpha'] * x - H(w) * w * (x - xs) - H(mv) * mv * x,
                  -p['r'] * (y + p['gamma'] / 2 * tau),
                  p['epsh'] * (y + p['gamma'] * tau - z)])
    
    return f

def fieldNoise(X, p, dummy=None):
    return np.array([0., 0., p['sigmah']])
    

def JacobianRO2D(X, p):
    (x, y, z) = X
    tau = p['tauExt'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0']
    mv = p['deltas'] * tau
    yp = p['eta1'] * z + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))

    # Derivatives of the wind-stress
    dtaudx = p['mu']
    dtaudy = 0.
    dtaudz = 0.
    # Derivatives of the upwelling
    dwdx = -p['deltas'] * dtaudx
    dwdy = -p['deltas'] * dtaudy # (= 0.)
    dwdz = -p['deltas'] * dtaudz # (= 0.)
    # Derivatives of the meridional velocity
    dmvdx = p['deltas'] * dtaudx
    dmvdy = p['deltas'] * dtaudy # (= 0.)
    dmvdz = p['deltas'] * dtaudz # (= 0.)
    # Derivatives of the argument inside f
    dypdx = 0
    dypdy = 0
    dypdz = p['eta1']
    dxsdx = 0.
    dxsdy = 0.
    dxsdz = -p['xs0'] * dypdz * dnl(yp)
    J = np.array([[-p['alpha'] \
                   - H(w) * (dwdx * (x - xs) + w * (1. - dxsdx)) \
                   - H(mv) * (dmvdx * x + mv),
                   - H(w) * (dwdy * (x - xs) - w * dxsdy) \
                   - H(mv) * dmvdy * x,
                   H(w) * w * dxsdz],
                  [-p['r'] * p['gamma'] / 2 * dtaudx,
                   -p['r'] * (1. + p['gamma'] / 2 * dtaudy),
                   -p['r'] * p['gamma'] / 2 * dtaudz],
                  [p['epsh'] * p['gamma'] * dtaudx,
                   p['epsh'] * (1. + p['gamma'] * dtaudy),
                   p['epsh'] * (p['gamma'] * dtaudz - 1.)]])
    
    return J


def JacFieldRO2D(dX, p, X):
    return np.dot(JacobianRO2D(X, p), dX)


def diagnose(X, p):
    # Get adimensional variables
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    tau = p['tauExt'] + p['mu'] * x
    w = -p['deltas'] * tau + p['w0']
    mv = p['deltas'] * tau
    yp = p['eta1'] * z + p['eta2']
    xs = p['xs0'] * (1. - nl(yp))

    # Get dimensional variables
    diagnostic = {}
    diagnostic['TE'] = x * pdim['DeltaT'] + pdim['T0']
    diagnostic['Ts'] = xs * pdim['DeltaT'] + pdim['T0']
    diagnostic['hW'] = y * pdim['Hm']
    diagnostic['hE'] = z * pdim['Hm']
    diagnostic['tau'] = tau * pdim['tau0']
    diagnostic['w'] = w * pdim['Hm'] * pdim['c0'] / pdim['L']
    diagnostic['v'] = -mv * pdim['Ly'] * pdim['c0'] / pdim['L'] / 2

    return diagnostic

    
def diagnoseMulti(X, pMulti):
    n = X.shape[0]

    # Allocate
    diagnostic = {}
    diagnostic['TE'] = np.empty((n,))
    diagnostic['Ts'] = np.empty((n,))
    diagnostic['hW'] = np.empty((n,))
    diagnostic['hE'] = np.empty((n,))
    diagnostic['tau'] = np.empty((n,))
    diagnostic['w'] = np.empty((n,))
    diagnostic['v'] = np.empty((n,))
    for k in np.arange(n):
        p = pMulti[k]

        # Get adimensional variables
        x = X[k, 0]
        y = X[k, 1]
        z = X[k, 2]
        tau = p['tauExt'] + p['mu'] * x
        w = -p['deltas'] * tau + p['w0']
        mv = p['deltas'] * tau
        yp = p['eta1'] * z + p['eta2']
        xs = p['xs0'] * (1. - nl(yp))

        # Get dimensional variables
        diagnostic['TE'][k] = x * pdim['DeltaT'] + pdim['T0']
        diagnostic['Ts'][k] = xs * pdim['DeltaT'] + pdim['T0']
        diagnostic['hW'][k] = y * pdim['Hm']
        diagnostic['hE'][k] = z * pdim['Hm']
        diagnostic['tau'][k] = tau * pdim['tau0']
        diagnostic['w'][k] = w * pdim['Hm'] * pdim['c0'] / pdim['L']
        diagnostic['v'][k] = -mv * pdim['Ly'] * pdim['c0'] / pdim['L'] / 2

    return diagnostic


def plotOrbit(diagnostic, p, limTE=None, limhW=None):
  os.system("mkdir %s/continuation/po/orbit/ 2> /dev/null" % plotDir)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(diagnostic['TE'], diagnostic['hW'], '-k')
  ax.scatter(diagnostic['TE'][0], diagnostic['hW'][0], s=40, c='k')
  if limTE is not None:
    ax.set_xlim(limTE[0], limTE[1])
  if limhW is not None:
    ax.set_ylim(limhW[0], limhW[1])
  ax.set_xlabel(r'$TE$', fontsize=ergoPlot.fs_latex)
  ax.set_ylabel(r'$hW$', fontsize=ergoPlot.fs_latex)
  plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
  plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)


def plotFloquetVec(xt, p, FE, FVL, FVR, comps, scale=1, colors=None,
                   compLabels=None):
  os.system("mkdir %s/continuation/po/orbit/ 2> /dev/null" % plotDir)
  (i, j) = comps
  po = xt[0]
  dim = po.shape[0]
  if colors is None:
    colors = ['r', 'g', 'b']
  if compLabels is None:
    compLabels = [r'$x$', r'$y$', r'$z$']
  # Plot (x, y) plane
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(xt[:, i], xt[:, j], '-k')
  ax.scatter(po[i], po[j], s=40, c='k')
  labels = ['0', '1', '2']
  for d in np.arange(dim):
    # Normalize only with respect to components
    vr = FVR[:, d].copy()
    vr /= np.sqrt(vr[i]**2 + vr[j]**2)
    ax.plot([po[i], po[i] + scale*vr[i]],
            [po[j], po[j] + scale*vr[j]], color=colors[d],
            linestyle='-', linewidth=2, label=r'$e^{%s}$' % labels[d])
    vl = FVL[:, d].copy()
    vl /= np.sqrt(vl[i]**2 + vl[j]**2)
    ax.plot([po[i], po[i] + scale*vl[i]],
            [po[j], po[j] + scale*vl[j]], color=colors[d],
            linestyle='--', linewidth=2, label=r'$f^{%s}$' % labels[d])
    ax.legend(fontsize=ergoPlot.fs_latex)
  ax.set_xlabel(compLabels[i], fontsize=ergoPlot.fs_latex)
  ax.set_ylabel(compLabels[j], fontsize=ergoPlot.fs_latex)
  plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
  plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)


def plotDiagnosticVersusMu(diagnostic, muRng, stable=None):
  lw = 2
  xlim = [np.min(muRng), np.max(muRng)]
  fig = plt.figure(figsize=[8, 10])
  ax1 = fig.add_subplot(511)
  ax1.plot(muRng, pdim['T0'] * np.ones((muRng.shape[0],)), '-r', linewidth=1)
  ax1.plot(muRng, pdim['Ts0'] * np.ones((muRng.shape[0],)), '-b', linewidth=1)
  ax1.plot(muRng, diagnostic['TE'], '-r', linewidth=lw)
  ax1.plot(muRng, diagnostic['Ts'], '-b', linewidth=lw)
  ylim = ax1.get_ylim()
  if stable is not None:
    for s in np.arange(1, stable.shape[0]):
      if (stable[s-1] ^ stable[s]):
        ax1.plot([muRng[s], muRng[s]], ylim, '--k')
  ax1.set_ylabel(varName['TE'] + ', ' + varName['Ts'], fontsize='xx-large')
  ax1.set_xlim(xlim)
  
  ax2 = fig.add_subplot(512)
  ax2.plot(muRng, diagnostic['hW'], '-r', linewidth=lw)
  ax2.plot(muRng, diagnostic['hE'], '-b', linewidth=lw)
  ax2.set_ylabel(varName['hW'] + ', ' + varName['hE'], fontsize='xx-large')
  ylim = ax2.get_ylim()
  if stable is not None:
    for s in np.arange(1, stable.shape[0]):
      if (stable[s-1] ^ stable[s]):
        ax2.plot([muRng[s], muRng[s]], ylim, '--k')
  ax2.set_xlim(xlim)
      
  ax3 = fig.add_subplot(513)
  ax3.plot(muRng, pdim['tauExt'] * np.ones((muRng.shape[0],)), '-k',
           linewidth=1)
  ax3.plot(muRng, diagnostic['tau'], '-k', linewidth=lw)
  ax3.set_ylabel(varName['tau'], fontsize='xx-large')
  ylim = ax3.get_ylim()
  if stable is not None:
    for s in np.arange(1, stable.shape[0]):
      if (stable[s-1] ^ stable[s]):
        ax3.plot([muRng[s], muRng[s]], ylim, '--k')
  ax3.set_xlim(xlim)
      
  ax4 = fig.add_subplot(514)
  ax4.plot(muRng, diagnostic['w'], '-b', linewidth=lw)
  ax4.set_ylabel(varName['w'], fontsize='xx-large')
  ylim = ax4.get_ylim()
  if stable is not None:
    for s in np.arange(1, stable.shape[0]):
      if (stable[s-1] ^ stable[s]):
        ax4.plot([muRng[s], muRng[s]], ylim, '--k')
  ax4.set_xlim(xlim)
      
  ax5 = fig.add_subplot(515)
  ax5.plot(muRng, diagnostic['v'], '-k', linewidth=lw)
  ax5.set_ylabel(varName['v'], fontsize='xx-large')
  ax5.set_xlabel(r'$\mu$', fontsize='xx-large')
  ylim = ax5.get_ylim()
  if stable is not None:
    for s in np.arange(1, stable.shape[0]):
      if (stable[s-1] ^ stable[s]):
        ax5.plot([muRng[s], muRng[s]], ylim, '--k')
  ax5.set_xlim(xlim)
      
  
