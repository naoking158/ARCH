#!/usr/bin/env python3
from collections import deque
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class CMAES:
    def __init__(self, xmean0, sigma0, lam=None):

        """
        Parameters
        ----------
        xmean0 : 1d array-like
            initial mean vector
        sigma0 : 1d array-like
            initial diagonal decoding
        lam : int, optional (default = None)
            population size
        beta_eig : float, optional (default = None)
            coefficient to control the frequency of matrix decomposition
        """

        self.N     = len(xmean0)
        self.chiN  = np.sqrt(self.N) * (1. - 1. / (4. * self.N) + 1. / (21. * self.N **2))

        # parameters for recombination and step-size adaptation
        self.lam   = lam if lam else 4 + int(3 * math.log(self.N))
        assert self.lam > 2
        w          = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam+1))
        w[w > 0]  /= np.sum(w[w > 0])
        w[w < 0]   = 0.
        self.w     = w
        self.mu    = self.lam // 2
        self.mueff = 1. / np.sum(self.w**2)
        self.cm    = 1.
        self.cs    = (self.mueff + 2.) / (self.N + self.mueff + 5.)
        self.ds    = 1. + self.cs + 2.*max(0., math.sqrt((self.mueff - 1.) / (self.N + 1.)) - 1.)

        # parameters for covariance matrix adaptation
        self.cone  = 2. / ((self.N + 1.3)**2 + self.mueff)
        self.cmu   = min(1. - self.cone,
                       2.*(self.mueff - 2. + 1./self.mueff) / ((self.N + 2.)**2 + self.mueff))
        self.cc    = (4. + self.mueff/self.N) / (self.N + 4. + 2.*self.mueff / self.N)

        # others
        self.neval = 0
        self.niter = 0

        # dynamic parameters
        self.xmean = np.array(xmean0, copy=True)
        self.sigma = np.array(sigma0, copy=True)
        self.D     = np.ones(self.N)
        self.cov   = np.diag(self.D ** 2)
        self._decompose()

        self.ps        = np.zeros(self.N)
        self.ps_factor = 0.0
        self.pc        = np.zeros(self.N)
        self.pc_factor = 0.0

        # storage for checker and logger
        self.arx = np.zeros((self.lam, self.N)) * np.nan
        self.arf = np.zeros(self.lam) * np.nan

        # decomposition performed after given #f-calls
        # For more detail, see
        # Y. Akimoto and N. Hansen., "Diagonal Acceleration for Covariance Matrix Adaptation Evolution Strategies"
        self.eigen_frequency = int(self.lam / (10 * self.N * (self.cone + self.cmu)))
        self.eigneval = 0


    def transform(self, z):
        y = np.dot(z, self.sqrtC)
        return y * (self.D * self.sigma)

    def transform_inverse(self, y):
        z = y / (self.D * self.sigma)
        return np.dot(z, self.invsqrtC)

    def sample_candidate(self):
        arz = np.random.randn(self.lam, self.N)
        ary = np.dot(arz, self.sqrtC.T)
        arx = self.xmean + self.sigma * ary
        return arx

    def _decompose(self):
        """Decompose the covariance matrix and update relevant parameters"""
        DD, self.B = np.linalg.eigh(self.cov)
        self.D = np.sqrt(DD)
        if not (self.D.max() < self.D.min() * 1e7):
            raise RuntimeError('Condition number > 1e7 or nan appears.')
        self.sqrtC = np.dot(self.B * self.D, self.B.T)
        self.invsqrtC = np.dot(self.B / self.D, self.B.T)
        self.eigneval = self.neval

    def update(self, idx, arx):
        # shortcut
        sary = (arx[idx] - self.xmean) / self.sigma
        sarx = arx[idx]

        # recombination
        dy = np.dot(self.w, sary)
        self.xmean += self.sigma * dy


        # step-size adaptation
        self.ps_factor = (1. - self.cs)**2 * self.ps_factor + self.cs * (2. - self.cs)
        self.ps *= (1. - self.cs)
        self.ps += math.sqrt(self.cs * (2. - self.cs) * self.mueff) * np.dot(dy, self.invsqrtC)
        normsquared = np.sum(self.ps * self.ps)
        hsig = (normsquared <
                ((1.4 + 2.0 / (self.N + 1.)) * self.chiN * math.sqrt(self.ps_factor))**2)
        self.sigma *= math.exp((math.sqrt(normsquared) / self.chiN
                                - math.sqrt(self.ps_factor)) * self.cs / self.ds)

        # covariance matrix adaptation
        # Rank-mu
        rank_mu   = np.dot(sary.T * self.w, sary) - self.cov

        # Rank-one
        self.pc  = (1. - self.cc) * self.pc
        self.pc += hsig * math.sqrt(self.cc * (2. - self.cc) * self.mueff) * dy
        self.pc_factor = (1. - self.cc)**2 * self.pc_factor + hsig * self.cc * (2. - self.cc)
        rank_one  = np.outer(self.pc, self.pc) - self.pc_factor * self.cov

        # Update
        self.cov += self.cmu * rank_mu + self.cone * rank_one

        # update the square root of the covariance matrix
        if (self.neval - self.eigneval) > self.eigen_frequency:
            self._decompose()

    def onestep(self, func):
        """
        Parameter
        ---------
        func : callable
            parameter : 2d array-like with candidate solutions (x) as elements
            return    : 1d array-like with f(x) as elements
        """
        # sampling
        arx = self.sample_candidate()  # (lam, dim)-array

        # evaluation
        arf = func(arx)
        self.neval += len(arf)

        # sort
        idx = np.argsort(arf)

        # update
        self.update(idx, arx)

        # finalize
        self.arx = arx
        self.arf = arf
        self.niter += 1

    @property
    def coordinate_std(self):
        return self.sigma * np.sqrt(np.diag(self.cov))

class Checker(object):
    """Termination Checker for BBOB

    Termination condition implemented in [Hansen 2009] (BI-POP-CMA-ES on BBOB 2019)
    is implemented here.

    """
    def __init__(self, cma):
        assert isinstance(cma, CMAES)
        assert hasattr(cma, 'sigma')
        self._cma = cma
        self._init_std = self._cma.sigma
        self._N = self._cma.N
        self._lam = self._cma.lam
        self._hist_fbest = deque(maxlen=10 + int(np.ceil(30 * self._N / self._lam)))
        self._hist_feq_flag = deque(maxlen=self._N)
        self._hist_fmin = deque()
        self._hist_fmed = deque()

    def __call__(self):
        return self.bbob_check()

    def check_maxiter(self):
        return self._cma.niter > 100 + 50 * (self._N + 3) ** 2 / np.sqrt(self._lam)

    def check_tolhistfun(self):
        self._hist_fbest.append(np.min(self._cma.arf))
        return (self._cma.niter >= 10 + int(np.ceil(30 * self._N / self._lam)) and
                np.max(self._hist_fbest) - np.min(self._hist_fbest) < 1e-12)

    def check_equalfunvals(self):
        k = int(math.ceil(0.1 + self._lam / 4))
        sarf = np.sort(self._cma.arf)
        self._hist_feq_flag.append(sarf[0] == sarf[k])
        return 3 * sum(self._hist_feq_flag) > self._N

    def check_tolx(self):
        assert hasattr(self._cma, 'pc')
        assert hasattr(self._cma, 'sigma')
        return (np.all(np.abs(self._cma.pc) * (self._cma.sigma / self._init_std) < 1e-12) and
                np.all(self._cma.coordinate_std / self._init_std) < 1e-12)

    def check_tolupsigma(self):
        assert hasattr(self._cma, 'sigma')
        assert hasattr(self._cma, 'D')
        return np.any((self._cma.sigma / self._init_std) > 1e20 * np.max(self._cma.D))

    def check_stagnation(self):
        self._hist_fmin.append(np.min(self._cma.arf))
        self._hist_fmed.append(np.median(self._cma.arf))
        _len = int(np.ceil(self._cma.niter / 5 + 120 + 30 * self._N / self._lam))
        if len(self._hist_fmin) > _len:
            self._hist_fmin.popleft()
            self._hist_fmed.popleft()
        fmin_med = np.median(np.asarray(self._hist_fmin)[-20:])
        fmed_med = np.median(np.asarray(self._hist_fmed)[:20])
        return self._cma.niter >= _len and fmin_med >= fmed_med

    def check_conditioncov(self):
        assert hasattr(self._cma, 'D')
        return np.max(self._cma.D) / np.min(self._cma.D) > 1e7

    def check_noeffectaxis(self):
        assert hasattr(self._cma, 'sigma')
        assert hasattr(self._cma, 'D')
        assert hasattr(self._cma, 'B')
        t = self._cma.niter % self._N
        test = 0.1 * self._cma.sigma * self._cma.D[t] * self._cma.B[:, t]
        return np.all(self._cma.xmean == self._cma.xmean + test)

    def check_noeffectcoor(self):
        return np.all(self._cma.xmean == self._cma.xmean + 0.2 * self._cma.coordinate_std)

    def check_flat(self):
        return np.max(self._cma.arf) == np.min(self._cma.arf)

    def bbob_check(self):
        if self.check_maxiter():
            return True, 'bbob_maxiter'
        if self.check_tolhistfun():
            return True, 'bbob_tolhistfun'
        if self.check_equalfunvals():
            return True, 'bbob_equalfunvals'
        if self.check_tolx():
            return True, 'bbob_tolx'
        if self.check_tolupsigma():
            return True, 'bbob_tolupsigma'
        if self.check_stagnation():
            return True, 'bbob_stagnation'
        if self.check_conditioncov():
            return True, 'bbob_conditioncov'
        if self.check_noeffectaxis():
            return True, 'bbob_noeffectaxis'
        if self.check_noeffectcoor():
            return True, 'bbob_noeffectcoor'
        if self.check_flat():
            return True, 'bbob_flat'
        return False, ''


class Logger:
    """Logger for CMAES"""
    def __init__(self, cma, prefix='log', variable_list=['xmean', 'D', 'sigma']):
        """
        Parameters
        ----------
        cma : CMAES instance
        prefix : string
            prefix for the log file path
        variable_list : list of string
            list of names of attributes of `cma` to be monitored
        """
        self._cma = cma
        self.prefix = prefix
        self.variable_list = variable_list
        self.logger = dict()
        self.fmin_logger = self.prefix + '_fmin.dat'
        with open(self.fmin_logger, 'w') as f:
            f.write('#' + type(self).__name__ + "\n")
        for key in self.variable_list:
            self.logger[key] = self.prefix + '_' + key + '.dat'
            with open(self.logger[key], 'w') as f:
                f.write('#' + type(self).__name__ + "\n")

    def __call__(self, condition=''):
        self.log(condition)

    def log(self, condition=''):
        with open(self.fmin_logger, 'a') as f:
            f.write("{} {} {}\n".format(self._cma.niter, self._cma.neval, np.min(self._cma.arf)))
            if condition:
                f.write('# End with condition = ' + condition)
        for key, log in self.logger.items():
            key_split = key.split('.')
            key = key_split.pop(0)
            var = getattr(self._cma, key)
            for i in key_split:
                var = getattr(var, i)
            if isinstance(var, np.ndarray) and len(var.shape) > 1:
                var = var.flatten()
            varlist = np.hstack((self._cma.niter, self._cma.neval, var))
            with open(log, 'a') as f:
                f.write(' '.join(map(repr, varlist)) + "\n")

    def my_formatter(self, x, pos):
        """Float Number Format for Axes"""
        float_str = "{0:2.1e}".format(x)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0}e{1}".format(base, int(exponent))
        else:
            return r"" + float_str + ""

    def plot(self,
             xaxis=0,
             ncols=None,
             figsize=None,
             cmap_='Spectral'):

        """Plot the result

        Parameters
        ----------
        xaxis : int, optional (default = 0)
            0. vs iterations
            1. vs function evaluations
        ncols : int, optional (default = None)
            number of columns
        figsize : tuple, optional (default = None)
            figure size
        cmap_ : string, optional (default = 'spectral')
            cmap

        Returns
        -------
        fig : figure object.
            figure object
        axdict : dictionary of axes
            the keys are the names of variables given in `variable_list`
        """
        mpl.rc('lines', linewidth=2, markersize=8)
        mpl.rc('font', size=12)
        mpl.rc('grid', color='0.75', linestyle=':')
        mpl.rc('ps', useafm=True)  # Force to use
        mpl.rc('pdf', use14corefonts=True)  # only Type 1 fonts
        mpl.rc('text', usetex=True)  # for a paper submision

        prefix = self.prefix
        variable_list = self.variable_list

        # Default settings
        nfigs = 1 + len(variable_list)
        if ncols is None:
            ncols = int(np.ceil(np.sqrt(nfigs)))
        nrows = int(np.ceil(nfigs / ncols))
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        axdict = dict()

        # Figure
        fig = plt.figure(figsize=figsize)
        # The first figure
        x = np.loadtxt(prefix + '_fmin.dat')
        x = x[~np.isnan(x[:, xaxis]), :]  # remove columns where xaxis is nan
        # Axis
        ax = plt.subplot(nrows, ncols, 1)
        ax.set_title('fmin')
        ax.grid(True)
        ax.grid(which='major', linewidth=0.50)
        ax.grid(which='minor', linewidth=0.25)
        plt.plot(x[:, xaxis], x[:, 2:])
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(self.my_formatter))
        axdict['fmin'] = ax

        # The other figures
        idx = 1
        for key in variable_list:
            idx += 1
            x = np.loadtxt(prefix + '_' + key + '.dat')
            x = x[~np.isnan(
                x[:, xaxis]), :]  # remove columns where xaxis is nan
            ax = plt.subplot(nrows, ncols, idx)
            ax.set_title(r'\detokenize{' + key + '}')
            ax.grid(True)
            ax.grid(which='major', linewidth=0.50)
            ax.grid(which='minor', linewidth=0.25)
            cmap = plt.get_cmap(cmap_)
            cNorm = mpl.colors.Normalize(vmin=0, vmax=x.shape[1] - 2)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
            for i in range(x.shape[1] - 2):
                plt.plot(
                    x[:, xaxis], x[:, 2 + i], color=scalarMap.to_rgba(i))
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self.my_formatter))
            axdict[key] = ax

        plt.tight_layout() # NOTE: not sure if it works fine
        return fig, axdict



def main():

    def sphere(x):
        return 0.5 * np.sum(x ** 2, axis=-1)

    # Main loop
    N = 20
    cma = CMAES(xmean0=np.random.randn(N), sigma0=np.ones(N))
    checker = Checker(cma)
    logger = Logger(cma)
    issatisfied = False
    fbestsofar = np.inf
    while not issatisfied:
        cma.onestep(func=sphere)
        fbest = np.min(cma.arf)
        fbestsofar = min(fbest, fbestsofar)
        if fbest < 1e-8:
            issatisfied, condition = True, 'ftarget'
        else:
            issatisfied, condition = checker()
        if cma.niter % 10 == 0:
            print(cma.niter, cma.neval, fbest, fbestsofar)
            logger()
    print(cma.niter, cma.neval, fbest, fbestsofar)
    print("Terminated with condition: " + condition)
    logger(condition)

    # Produce a figure
    fig, axdict = logger.plot()
    for key in axdict:
        if key not in ('xmean'):
            axdict[key].set_yscale('log')
    plt.savefig(logger.prefix + '.pdf', tight_layout=True)

if __name__ == '__main__':
    main()
