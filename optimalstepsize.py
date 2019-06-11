import math
import numpy as np
from scipy.stats import norm


class NormalOrderStatistics(object):
    """Compute Moments of Normal Order Statistics

    Requires
    --------
    numpy
    scipy.stats.norm
    """

    def __init__(self, n):
        """Normal Order Statistics from `n` populations

        Normal order statistics of population size `n` are the ordered 
        random variables
            N_{1:n} < N_{2:n} < ... < N_{n:n}
        that are drawn from the standard normal distribution N(0, 1)
        independently.

        Parameters
        ----------
        n : int
            population size
        """
        self._n = n
        self._pr = np.arange(1, n + 1, dtype=float) / (n + 1)
        self._q0r = norm.ppf(self._pr)
        self._q1r = 1.0 / norm.pdf(self._q0r)
        self._q2r = self._q0r * self._q1r**2
        self._q3r = (1.0 + 2.0 * self._q0r**2) * self._q1r**3
        self._q4r = self._q0r * (7.0 + 6.0 * self._q0r**2) * self._q1r**4

    def exp(self):
        """Expectation of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._q0r
        result += self._pr * (1 - self._pr) * self._q2r / (2 * self._n + 4)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * self._q3r / (3 * (self._n + 2)**2)
        result += (self._pr *
                   (1 - self._pr))**2 * self._q4r / (8 * (self._n + 2)**2)
        return result

    def var(self):
        """Variance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        1D ndarray : array of variance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = self._pr * (1 - self._pr) * self._q1r**2 / (self._n + 2)
        result += self._pr * (1 - self._pr) * (
            1 - 2 * self._pr) * 2 * self._q1r * self._q2r / ((self._n + 2)**2)
        result += (self._pr * (1 - self._pr))**2 * (
            self._q1r * self._q3r + self._q2r**2 / 2) / (self._n + 2)**2
        return result

    def cov(self):
        """Covariance of the normal order statistics, using Taylor Expansion.

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        Algorithm
        ---------
        Eq. (4.6.3)--(4.6.5) combined with Example 4.6 in "Order Statistics".
        """
        result = np.outer(self._pr**2 * self._q2r, (1 - self._pr)
                          **2 * self._q2r) / 2
        result += np.outer(self._pr * (1 - 2 * self._pr) * self._q2r,
                           (1 - self._pr) * self._q1r)
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * (1 - 2 * self._pr) * self._q2r)
        result += np.outer(self._pr**2 * (1 - self._pr) * self._q3r,
                           (1 - self._pr) * self._q1r) / 2
        result += np.outer(self._pr * self._q1r, self._pr * (1 - self._pr)
                           **2 * self._q3r) / 2
        result /= (self._n + 2)**2
        result += np.outer(self._pr * self._q1r,
                           (1 - self._pr) * self._q1r) / (self._n + 2)
        return np.triu(result) + np.triu(result, k=1).T

    def blom(self):
        """Blom's Approximation of the Expectation of Normal Order Statistics

        Returns
        -------
        1D ndarray : array of expectation of the normal order statistics
        """
        alpha = 0.375
        pir = (np.arange(1, self._n + 1) - alpha) / (self._n + 1 - 2 * alpha)
        return norm.ppf(pir)

    def davis_stephens(self):
        """Refinement of Covariance Matrix by Algorithm 128

        Returns
        -------
        2D ndarray : array of covariance of the normal order statistics

        See
        ---
        https://statistics.stanford.edu/sites/default/files/SOL%20ONR%20254.pdf
        """
        result = self.cov()
        n = self._n
        for i in range((n + 1) // 2):
            rowsum = np.sum(result[i])
            free = np.sum(result[i, i:n - i])
            result[i, i:n - i] *= 1 + (1 - rowsum) / free
            result[i:n - i, i] = result[i, i:n - i]
            result[n - i - 1, i:n - i] = (result[i, i:n - i])[::-1]
            result[i:n - i, n - i - 1] = (result[i, i:n - i])[::-1]
        return result


# def sigma_normalization_factor(dim, weights, cm=1.):
#     nos = NormalOrderStatistics(len(weights))
#     nlam = nos.blom()
#     beta = -np.dot(nlam, weights)
#     muw = np.dot(weights, weights)
#     return beta * muw / (dim - 1 + beta * beta * muw) / cm

# Modified at May 12, 2017
def sigma_normalization_factor(dim, weights, cm=1.):
    """sigma = snf * ||xmean|| """
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    if len(weights) < 50:
        nnlam = nos.davis_stephens()
        gamma = beta**2 + np.dot(np.dot(nnlam, weights), weights)
    else:
        gamma = beta**2
    muw = np.sum(np.abs(weights)) ** 2 / np.dot(weights, weights)
    return beta * muw / (dim - 1 + gamma * muw) / cm


def quadratic_optimal_sigma(hess, xmean, weights, cm=1.):
    """Optimal Sigma for Quadratic
    If hess is proportional to [1, ..., 1] or the identity matrix,
    the result should be the same as 
    sigma_normalization_factor * ||xmean||.
    """    
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    if len(weights) < 50:
        nnlam = nos.davis_stephens()
        gamma = beta**2 + np.dot(np.dot(nnlam, weights), weights)
    else:
        gamma = beta**2
    muw = np.sum(np.abs(weights)) ** 2 / np.dot(weights, weights)
    if np.ndim(hess) == 1:
        e = hess * xmean
        eae = np.dot(hess, e * e) / np.dot(e, e) / np.sum(hess)
        g = np.linalg.norm(e) / np.sum(hess)
    else:
        e = np.dot(hess, xmean)
        eae = np.dot(e, np.dot(hess, e)) / np.dot(e, e) / np.trace(hess)
        g = np.linalg.norm(e) / np.trace(hess)
    return (beta * muw / (1 - eae + eae * gamma * muw) / cm) * g

def quadratic_optimal_normalized_sigma(hess, weights, cm=1.):
    """Estimate the optimal sigma and normalized quality gain
    The mean vector is supposed to be on the long axis of the ellipsoid.
    """
    if np.ndim(hess) == 1:
        idx = np.argmin(hess)
        xmean = np.zeros(hess.shape[0])
        xmean[idx] = 1.0
        g = hess[idx] / np.sum(hess)
    else:
        d, b = np.linalg.eigh(hess)
        idx = np.argmin(d)
        xmean = b[:, idx]
        g = d[idx] / np.sum(d)
    nos = NormalOrderStatistics(len(weights))
    nlam = nos.blom()
    beta = -np.dot(nlam, weights)
    optns = quadratic_optimal_sigma(hess, xmean, weights, cm=cm) * (cm / g)
    optnqg = beta * optns / 2.0
    return optns, optnqg, g


def truncation_weights(lam, mu):
    w_arr = np.zeros(lam)
    w_arr[:mu] = 1.0 / mu
    return w_arr

def cmatype_weights(lam):
    w_arr = np.zeros(lam)
    w_arr[:(lam // 2)] = math.log(
        (lam + 1) / 2.0) - np.log(np.arange(lam // 2) + 1)
    w_arr /= np.sum(w_arr[:(lam // 2)])
    return w_arr

def optimal_weights(lam):
    nos = NormalOrderStatistics(lam)
    w_arr = -nos.exp()
    w_arr /= np.sum(np.abs(w_arr))
    return w_arr
