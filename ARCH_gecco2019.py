import numpy as np
import math
from collections import Counter
from CMAES import CMAES
from optimalstepsize import sigma_normalization_factor


class ARCH(CMAES):
    def __init__(self, func, xmean0, sigma0, matA, vecb, eps=1e-13, tol_for_ineqcons=0.0, **kwargs):
        
        super(ARCH, self).__init__(func=func, xmean0=xmean0, sigma0=sigma0, **kwargs)

        self.init_eps = eps
        self.eps = self.init_eps
        self.tol_for_ineqcons = tol_for_ineqcons
        self.A = matA.reshape(1, len(matA)).astype(float) if matA.ndim == 1 else matA.astype(float)
        self.b = vecb.ravel().astype(float) if vecb.ndim == 2 else vecb.astype(float)
        self.num_of_constraint = self.b.shape[0]
        self.bool_idx = np.ones(self.num_of_constraint, dtype=bool)
        self.num_of_fail = 0

        self.weight = np.hstack((self.w, np.zeros(self.lam - len(self.w))))
        self.sqreSNF      = sigma_normalization_factor(self.N, self.weight) ** 2
        self.d_alpha      = 1.0 / self.N
        self.alpha        = 1.0
        self.dm_threshold = 1.0
        self.arffeas      = np.empty(self.lam)
        self.arpen        = np.empty(self.lam)
        self.arxfeas      = np.empty((self.lam, self.N))


    @property
    def criterion(self):
        if hasattr(self, 'fopt'):
            return self.fopt - self.arffeas.min()
        elif hasattr(self, 'xopt'):
            if not hasattr(self, 'hess'):
                self.hess = np.eye(N)
            return np.dot(np.dot(self.xmean - self.xopt, self.hess), self.xmean - self.xopt)
        else:
            return self.arffeas.min()

        
    def evaluate_candidate(self, arx):
        """
        Parameters
        ----------
        arx : 2D-array (lam, dim)
                candidate solutions
                
        Returns
        -------
        arf : 1D-array (lam)
                the total ranking

        idx : 1D-array (lam)
                ascending orderd index based on the `arf`

        """

        arxfeas = self.repair(arx)
        Cinv = np.dot(self.invsqrtC.T, self.invsqrtC) / (self.sigma**2)
        self.arpen = self.compute_penalty(x_infeas=arx, x_feas=arxfeas, Cinv=Cinv)
        self.arffeas = np.asarray([self.func(arxfeas[i]) 
                                   if np.all(self.violation_of_(arxfeas[i]) <= 0) else np.inf
                                   for i in range(self.lam)])
        self.neval += self.lam

        arfrank = self.compute_ranking(self.arffeas)
        argrank = self.compute_ranking(self.arpen)

        self.update_alpha(Cinv)
        arf = arfrank + self.alpha * argrank
        idx = np.argsort(arf)
        return arf, idx

        
    @staticmethod
    def compute_ranking(values):
        idx = np.argsort(values)
        res = np.zeros(len(values)) * np.nan
        res[idx] = np.arange(len(values))
        for key, cnt in Counter(values).items():
            if cnt > 1:
                res[values == key] = np.mean(res[values == key])
        return res

    
    def violation_of_(self, x):
        """
        Parameters
        ----------
        x : ndarray (1D or 2D)
        
        Returns
        -------
        violation
        """
        violation = np.dot(x, self.A.T) - self.b
        return violation


    def repair(self, x, info=False, is_xmean=False):
        if info:
            if np.ndim(x) >= 2:
                tmp = [self.repair(x[i], info=True, is_xmean=is_xmean)
                       for i in range(x.shape[0])]

                X = np.array([tmp[i][0] for i in range(len(tmp))])
                J_done = np.array([tmp[i][1] for i in range(len(tmp))])
                violation = [tmp[i][2] for i in range(len(tmp))]

                return X, J_done, violation

            elif np.ndim(x) == 1:
                if np.all(self.violation_of_(x) <= self.tol_for_ineqcons):
                    return x, [], []
                else:
                    res = self.nearest_feasible(x)
                    if (np.any(self.violation_of_(res[0]) > self.tol_for_ineqcons) 
                        and not is_xmean):
                        self.num_of_fail += 1
                    return res
        else:
            return self.repair(x, info=True, is_xmean=is_xmean)[0]


    def nearest_feasible(self, x):
        """Find the nearest feasible solution in terms of the Mahalanobis distance

        Parameters
        ----------
        x : ndarray (1D)

        Returns
        -------
        xnear : ndarray
            Repaired x. If `out` is passed, its reference is returned.

        J_done : list
            list of index of active constraint

        A : ndarray (2D)
            the rows of A are normal vectors of active constraints

        pinvA : ndarray (2D)
            Moore-Penrose inverse of A, which is a right inverse of A
        """

        assert (np.ndim(x) == 1)
        J_list = list(range(self.num_of_constraint))
        J_done = []
        
        ynear = np.dot(self.invsqrtC, x) / self.sigma

        A = np.dot(self.A, self.sqrtC) * self.sigma
        b = self.b - self.eps
        pinvA = np.empty((0, self.N))

        for _ in range(self.num_of_constraint):
            violation = np.dot(A, ynear) - b

            # Termination check
            if np.all(violation <= self.eps + self.tol_for_ineqcons):
                break

            # index of violated or handled constraints
            J = [j for j in range(self.num_of_constraint)
                 if j in J_done or violation[j] >= self.eps+self.tol_for_ineqcons]

            pinvA  = np.linalg.pinv(A[J])
            ynear -= np.dot(pinvA, violation[J])
            J_done = J.copy()

        return self.sigma*np.dot(self.sqrtC, ynear), J_done, np.dot(A, ynear) - b

    
    @staticmethod
    def compute_penalty(x_infeas, x_feas, Cinv):
        assert x_infeas.shape == x_feas.shape
        dx = x_infeas - x_feas

        if np.ndim(x_infeas) == 2:
            return np.sum(np.dot(dx, Cinv) * dx, axis=-1)
        elif np.ndim(x_infeas) == 1:
            return np.dot(np.dot(dx, Cinv), dx)


    def update_alpha(self, Cinv):
        mfeas, J_done, _  = self.repair(x=self.xmean, info=True, is_xmean=True)
        if np.array_equal(mfeas, self.xmean):
            self.dm = 0.0
        else:
            num_of_act = len(J_done)
            self.dm  = self.compute_penalty(x_infeas=self.xmean, x_feas=mfeas, Cinv=Cinv)
            self.dm *= self.sqreSNF  # sqreSNF = (optimal normalized step-size / n)^2
            self.dm *= 2.*self.N / (self.N + 2.*num_of_act)
        
        if not hasattr(self, 'dm_old'):
            # Initialization
            self.dm_old = 0.0

        if np.sign(self.dm - self.dm_old) == np.sign(self.dm - self.dm_threshold) or self.dm == 0.0:
            self.alpha *= np.exp(np.sign(self.dm - self.dm_threshold) * self.d_alpha)

        self.dm_old = self.dm
        self.alpha = max(1.0 / self.lam, min(self.alpha, self.lam))

        # for repair procedure at next iteration
        if self.num_of_fail <= np.ceil(self.lam / 10):
            self.eps *= 0.5
        else:
            self.eps *= 10.
        
        self.eps = max(self.init_eps, min(1e-4, self.eps))
        self.num_of_fail = 0


def main():

    def func(x):
        return 0.5 * np.sum(x ** 2, axis=-1)

    N = 20
    sigma0 = 1.
    xmean0 = np.ones(N) * 3.
    A = np.vstack((np.eye(N)*-1, np.eye(N)))
    b = np.array([-1., 1.] * (N//2))
    b = np.hstack((b, b + 5.))

    xopt = b[:N] * (-1)
    xopt[xopt < 0] = 0.0
    algo = ARCH(func=func, 
                sigma0=sigma0, 
                xmean0=xmean0, 
                matA=A, vecb=b,
                maxiter=2000,
                target=1e-8,
                xopt=xopt, hess=np.eye(N))

    algo.run()


if __name__ == '__main__':
    main()
