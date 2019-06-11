import numpy as np
import math


class CMAES:
    def __init__(self, func, xmean0, sigma0, **kwargs):

        self.N     = len(xmean0)
        self.lam   = kwargs.get('lam', int(4 + np.floor(3 * np.log(self.N))))
        self.func  = func
        self.neval = 0
        self.niter = 0

        wtemp = np.array([
            np.log(np.float(self.lam + 1) / 2.0) - np.log(1 + i)
            for i in range(self.lam // 2)
        ])
        self.w     = kwargs.get('w', wtemp / np.sum(wtemp))
        self.mueff = 1.0 / (self.w**2).sum()
        self.mu    = self.w.shape[0]

        self.cs   = (self.mueff + 2.) / (self.N + self.mueff + 5.)
        self.cc   = (4. + self.mueff/self.N) / (self.N + 4. + 2.*self.mueff/self.N)
        self.cone = 2. / ((self.N + 1.3)**2 + self.mueff)
        self.cmu  = min(1. - self.cone, 
                       2.*(self.mueff - 2. + 1./self.mueff) / ((self.N + 2.)**2 + self.mueff))

        self.xmean = xmean0
        self.sigma = sigma0
        self.D     = np.ones(self.N)
        self.cov   = np.diag(self.D**2)
        self._decompose()

        self.ds        = 1. + self.cs + 2.*max(0., np.sqrt((self.mueff - 1.) / (self.N + 1.)) - 1.)
        self.ps        = np.zeros(self.N)
        self.ps_factor = 0.0
        self.pc        = np.zeros(self.N)
        self.pc_factor = 0.0
        self.chiN      = np.sqrt(self.N) * (1. - 1. / (4. * self.N) + 1. / (21. * self.N **2))

        self.arx        = np.zeros((self.lam, self.N)) * np.nan
        self.dx         = np.zeros(self.N) * np.nan
        self.arf        = np.zeros(self.lam) * np.nan
        self.sorted_idx = np.zeros(self.lam) * np.nan
        self.fbest      = self.func(self.xmean)
        
        # decomposition performed after given #f-calls
        self.eigen_frequency = self.lam / (10 * self.N * (self.cone + self.cmu)) 
        
        # Termination Conditions
        self.ftarget = kwargs.get('ftarget', 1e-8)
        self.maxeval = kwargs.get('maxeval', 5e4 * self.N)
        self.maxiter = kwargs.get('maxiter', 5e3 * self.N)
        
        self.__dict__.update(kwargs)

        
    def run(self):
        itr = 0
        satisfied = False
        
        print('{:>4s}   {:>6s}   {:>11s}   {:>11s}'.format(
            '#iter', '#eval', 'criterion', 'sigma'))
        print('|-----------------------------------------------|')
        while not satisfied:
            self._onestep()
            satisfied, condition = self._check()
            if self.niter % 10 == 0 or condition:
                print('{:4d}    {:5d}    {:8e}    {:8e}'.format(
                    self.niter, self.neval, self.criterion, self.sigma))
            if satisfied:
                print(condition)
                break
        return self.xmean


    def _onestep(self):
        self.arx = self.sample_candidate()  # (lam, dim)-array
        self.arf, self.sorted_idx = self.evaluate_candidate(self.arx)
        self.update(self.arf, self.sorted_idx)
        self.niter += 1

    
    @property
    def criterion(self):
        return self.arf.min()

    
    def _check(self):
        """Check the termination criteria
        Returns
        -------
        flag : bool
            True if one of the termination condition is satisfied, False otherwise
        condition : str
            String of condition, empty string '' if no condition is satisfied
        """
        if self.criterion < self.ftarget:
            return True, 'target'
        elif self.niter >= self.maxiter:
            return True, 'maxiter'
        elif self.neval >= self.maxeval:
            return True, 'maxeval'
        else:
            return False, ''
        
        
    def _decompose(self):
        """Decompose the covariance matrix and update relevant parameters"""
        DD, self.B = np.linalg.eigh(self.cov)
        self.D = np.sqrt(DD)
        if not (self.D.max() < self.D.min() * 1e7):
            raise RuntimeError('Condition number > 1e7 or nan appears.')
        self.sqrtC = np.dot(self.B * self.D, self.B.T)
        self.invsqrtC = np.dot(self.B / self.D, self.B.T)
        self.eigneval = self.neval


    def sample_candidate(self):
        arz = np.random.randn(self.lam, self.N)
        ary = np.dot(arz, self.sqrtC.T)
        arx = self.xmean + self.sigma * ary
        return arx
    
    
    def evaluate_candidate(self, arx):
        """
        Parameters
        ----------
        arx : 2D-array (lam, dim)
                candidate solutions
                
        Returns
        -------
        arf : 1D-array (lam)
                the objective function values
                
        idx : 1D-array (lam)
                ascending orderd index based on the `arf`

        """
        
        arf = self.func(arx)
        idx = np.argsort(arf)

        self.neval += self.lam

        return arf, idx
    
    
    def update(self, arf, idx):
        sary = (self.arx[idx] - self.xmean) / self.sigma
        sarx = self.arx[idx]
        
        # Update sigma
        self.ps_factor = (1. - self.cs)**2 * self.ps_factor + self.cs * (2. - self.cs)
        self.ps *= (1. - self.cs)
        self.ps += math.sqrt(self.cs * (2. - self.cs) * self.mueff) * \
                    np.dot(np.dot(self.w, sary[:self.mu]), self.invsqrtC)
        normsquared = np.sum(self.ps * self.ps)
        self.sigma *= math.exp((math.sqrt(normsquared)/self.chiN - math.sqrt(self.ps_factor))
                               * self.cs / self.ds)
        
        hsig = (normsquared < 
                ((1.4 + 2.0 / (self.N + 1.)) * self.chiN * math.sqrt(self.ps_factor))**2)
        
        # Update xmean
        self.dx     = np.dot(self.w, sarx[:self.mu]) - self.xmean
        self.xmean += self.dx
        
        # Cumulation
        self.pc  = (1. - self.cc) * self.pc 
        self.pc += hsig * math.sqrt(self.cc * (2. - self.cc) * self.mueff) * np.dot(self.w, sary[:self.mu])
        self.pc_factor = (1. - self.cc)**2 * self.pc_factor + hsig * self.cc * (2. - self.cc)
        
        rank_mu   = np.dot(sary[:self.mu].T * self.w, sary[:self.mu]) - self.cov
        rank_one  = np.outer(self.pc, self.pc) - self.pc_factor * self.cov
        self.cov += self.cmu * rank_mu + self.cone * rank_one
        
        # update the square root of the covariance matrix
        if (self.neval - self.eigneval) > self.eigen_frequency:
            self._decompose()


def main():

    def func(x):
        return 0.5 * np.sum(x ** 2, axis=-1)

    N = 20
    sigma0 = 1.
    xmean0 = np.ones(N)

    algo = CMAES(func=func,
                 sigma0=sigma0,
                 xmean0=xmean0,
                 maxiter=500,
                 target=1e-8)

    algo.run()


if __name__ == '__main__':
    main()
