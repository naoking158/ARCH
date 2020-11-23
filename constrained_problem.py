#!/usr/bin/env python3

import numpy as np
import warnings
try:
    from numba import jit
except ImportError:
    print("`numba` not found")


class CEC2006ConstrainedBenchmarkSet(object):
    ''' Constrained problems of CEC2006 constrained competition

    See also,
        ``Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization'',
    Liang, J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello, C. C., and Deb, K. (2006)

    How to use
    ----------
    >> cec2006 = CEC2006ConstrainedBenchmarkSet()
    >> prob = cec2006.constrained_problem(prob_idx)  # prob_idx can be 1 to 24.

    `prob` is dict type and it contains following keywords,
        f, cons_dict, lb, ub, fopt, mact, ni, ne,

    `mact` is number of active constraints at the optimum,
    `ni` and `ne` are number of inequality and equality constraint, respectively.

    `cons_dict` contains two elements:
        cons_dict['eq'] : list of equality constraint fucntions
        cons_dict['ineq'] : list of inequality constraint fucntions


    If you want to include box constraints in `cons_dict['ineq']` in function form, do bellow,

    >> cons_dict['ineq'] += transform_bounds2ineq_cons(lb, ub)

    '''

    def __init__(self, tol_for_eqcons=1e-4):
        self.tol = tol_for_eqcons  # it is default value in CEC2006


    @staticmethod
    def transform_eq2ineq_cons(eqcons_list, tol):
        ''' transform equality constraints to inequality one

        Parameters
        ----------
        eqcons_list : list
            equality constraint funcsions

        Returns
        -------
        ineqcons_list : list

        '''
        def get_ineqcons(idx):
            def gj(x):
                return np.abs(eqcons_list[idx](x)) - tol
            return gj

        return [get_ineqcons(idx) for idx in range(len(eqcons_list))]


    @staticmethod
    def transform_bounds2ineq_cons(lb, ub):
        ''' transform bounds to inequality constraints

        Parameters
        ----------
        lb : 1D-array
            lower bound

        ub : 1D-array
            upper bound

        Returns
        -------
        list
        '''
        def get_gj(idx):

            def gj_lb(x):
                return -x[idx] + lb[idx]

            def gj_ub(x):
                return x[idx] - ub[idx]

            return [gj_lb, gj_ub]

        return np.asarray([get_gj(idx) for idx in range(len(lb))]).ravel().tolist()


    def constrained_problem(self, idx):
        '''
        Parameters
        ----------
        idx : int (1 -- 24)
            index of the problem c1 -- c24

        Returns
        -------
        prob : dict
            it constaints following elements,

        /-------------------/
        f : method
            the objetive function

        cons_dict : dict
            the dict contains two lists:
                cons_dict['eq'] : equality constraint functions list
                cons_dict['ineq'] : inequality constraint functions list

        lb : 1d-array
            lower bound

        ub : 1d-array
            upper bound

        fopt : float
            the optimum value reported in CEC2006

        mact : int
            number of active constraints at the optimum

        ni : int
            number of inequality constraints

        ne : int
            number of equality constraints
        /-------------------/

        '''
        return eval('self.c' + str(idx) + '()')


    def c1(self):
        dim = 13
        ni = 9
        ne = 0
        lb = np.zeros(dim)
        ub = np.asarray([1.]*9 + [100.]*3 + [1.])
        fopt = -15.
        mact = 6

        @jit('f8(f8[:])')
        def f(x):
            return 5.*(np.sum(x[:4]) - np.sum(x[:4]**2)) - np.sum(x[4:])

        @jit('f8(f8[:])')
        def g1(x):
            return 2.0 * x[0] + 2.0 * x[1] + x[9] + x[10] - 10.

        @jit('f8(f8[:])')
        def g2(x):
            return 2.0 * x[0] + 2.0 * x[2] + x[9] + x[11] - 10.

        @jit('f8(f8[:])')
        def g3(x):
            return 2.0 * x[1] + 2.0 * x[2] + x[10] + x[11] - 10.

        @jit('f8(f8[:])')
        def g4(x):
            return -8.0 * x[0] + x[9]

        @jit('f8(f8[:])')
        def g5(x):
            return -8.0 * x[1] + x[10]

        @jit('f8(f8[:])')
        def g6(x):
            return -8.0 * x[2] + x[11]

        @jit('f8(f8[:])')
        def g7(x):
            return -2.0 * x[3] - x[4] + x[9]

        @jit('f8(f8[:])')
        def g8(x):
            return -2.0 * x[5] - x[6] + x[10]

        @jit('f8(f8[:])')
        def g9(x):
            return -2.0 * x[7] - x[8] + x[11]

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c2(self):
        dim = 20
        ni = 2
        ne = 0
        lb = np.zeros(dim) + 1e-15
        ub = np.ones(dim) * 10.
        fopt = - 0.80361910412559
        mact = 1

        @jit('f8(f8[:])')
        def f(x):
            res = np.sum(np.cos(x)**4) - 2.*np.prod(np.cos(x)**2)
            res /= np.sqrt(np.sum(np.arange(1, dim+1, dtype=np.float64) * x**2))
            return - np.abs(res)

        @jit('f8(f8[:])')
        def g1(x):
            return 0.75 - np.prod(x)

        @jit('f8(f8[:])')
        def g2(x):
            return np.sum(x) - 7.5 * dim

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c3(self):
        dim = 10
        ni = 0
        ne = 1
        lb = np.zeros(dim)
        ub = np.ones(dim)
        fopt = - 1.00050010001000
        mact = 1

        @jit('f8(f8[:])')
        def f(x):
            return - (np.sqrt(dim))**dim * np.prod(x)

        @jit('f8(f8[:])')
        def h1(x):
            return np.sum(x ** 2) - 1.

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c4(self):
        dim = 5
        ni = 6
        ne = 0
        lb = np.asarray([78.] + [33.] + [27.]*3)
        ub = np.asarray([102.] + [45.]*4)
        fopt = - 3.066553867178332e4
        mact = 2

        @jit('f8(f8[:])')
        def f(x):
            return 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

        @jit('f8(f8[:])')
        def g1(x):
            return 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92.

        @jit('f8(f8[:])')
        def g2(x):
            return -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4]

        @jit('f8(f8[:])')
        def g3(x):
            return 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] * x[2] - 110.

        @jit('f8(f8[:])')
        def g4(x):
            return -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] * x[2] + 90.

        @jit('f8(f8[:])')
        def g5(x):
            return 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25.

        @jit('f8(f8[:])')
        def g6(x):
            return -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20.

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c5(self):
        dim = 4
        ni = 2
        ne = 3
        lb = np.asarray([0.]*2 + [-0.55]*2)
        ub = np.asarray([1200.]*2 + [0.55]*2)
        fopt = 5126.4967140071
        mact = 3

        @jit('f8(f8[:])')
        def f(x):
            return 3.0 * x[0] + 0.000001 * x[0]**3 + 2.0 * x[1] + (0.000002 / 3.0) * x[1]**3

        @jit('f8(f8[:])')
        def g1(x):
            return -x[3] + x[2] - 0.55

        @jit('f8(f8[:])')
        def g2(x):
            return -x[2] + x[3] - 0.55

        @jit('f8(f8[:])')
        def h1(x):
            return 1e3 * np.sum(np.sin(-x[2:] - 0.25)) + 894.8 - x[0]

        @jit('f8(f8[:])')
        def h2(x):
            return 1e3 * (np.sin(x[2] - 0.25) + np.sin(x[2] - x[3] - 0.25)) + 894.8 - x[1]

        @jit('f8(f8[:])')
        def h3(x):
            return 1e3 * (np.sin(x[3] - 0.25) + np.sin(x[3] - x[2] - 0.25)) + 1294.8

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': ineqcons_list, 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c6(self):
        dim = 2
        ni = 2
        ne = 0
        lb = np.asarray([13., 0.])
        ub = np.asarray([100., 100.])
        fopt = - 6961.81387558015
        mact = 2

        @jit('f8(f8[:])')
        def f(x):
            tmp = x - 10.0
            tmp[1] -= 10.0
            return np.sum(tmp**3)

        @jit('f8(f8[:])')
        def g1(x):
            return - np.sum((x - 5.) ** 2) + 100.

        @jit('f8(f8[:])')
        def g2(x):
            return (x[0] - 6.)**2 + (x[1] - 5.)**2 - 82.81

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c7(self):
        dim = 10
        ni = 8
        ne = 0
        lb = np.ones(dim) * -10.
        ub = np.ones(dim) * 10.
        fopt = 24.30620906818
        mact = 6

        @jit('f8(f8[:])')
        def f(x):
            res = np.sum(x[:2] ** 2) + x[0] * x[1] - 14.0 * x[0] - 16.0 * x[1] + (x[2] - 10.0)**2
            res += 4.0 * (x[3] - 5.0)**2 + (x[4] - 3.0)**2 + 2.0 * (x[5] - 1.0)**2 + 5.0 * x[6]**2
            res += 7.0 * (x[7] - 11)**2 + 2.0 * (x[8] - 10.0)**2 + (x[9] - 7.0)**2 + 45.
            return res

        @jit('f8(f8[:])')
        def g1(x):
            return -105.0 + 4.0 * x[0] + 5.0 * x[1] - 3.0 * x[6] + 9.0 * x[7]

        @jit('f8(f8[:])')
        def g2(x):
            return 10.0 * x[0] - 8.0 * x[1] - 17.0 * x[6] + 2.0 * x[7]

        @jit('f8(f8[:])')
        def g3(x):
            return -8.0 * x[0] + 2.0 * x[1] + 5.0 * x[8] - 2.0 * x[9] - 12.0

        @jit('f8(f8[:])')
        def g4(x):
            return 3.0 * (x[0] - 2.0) * (x[0] - 2.0) + 4.0 * (x[1] - 3.0) * (x[1] - 3.0) + 2.0 * x[2] * x[2] - 7.0 * x[3] - 120.0

        @jit('f8(f8[:])')
        def g5(x):
            return 5.0 * x[0] * x[0] + 8.0 * x[1] + (x[2] - 6.0) * (x[2] - 6.0) - 2.0 * x[3] - 40.0

        @jit('f8(f8[:])')
        def g6(x):
            return x[0] * x[0] + 2.0 * (x[1] - 2.0) * (x[1] - 2.0) - 2.0 * x[0] * x[1] + 14.0 * x[4] - 6.0 * x[5]

        @jit('f8(f8[:])')
        def g7(x):
            return 0.5 * (x[0] - 8.0) * (x[0] - 8.0) + 2.0 * (x[1] - 4.0) * (x[1] - 4.0) + 3.0 * x[4] * x[4] - x[5] - 30.0

        @jit('f8(f8[:])')
        def g8(x):
            return -3.0 * x[0] + 6.0 * x[1] + 12.0 * (x[8] - 8.0) * (x[8] - 8.0) - 7.0 * x[9]

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c8(self):
        dim = 2
        ni = 2
        ne = 0
        lb = np.zeros(dim)
        ub = np.ones(dim) * 10.
        fopt = - 0.0958250414180359
        mact = 0

        @jit('f8(f8[:])')
        def f(x):
            return - np.sin(2. * np.pi * x[0])**3 * np.sin(2. * np.pi * x[1]) / (x[0]**3 * np.sum(x))

        @jit('f8(f8[:])')
        def g1(x):
            return x[0]**2 - x[1] + 1.

        @jit('f8(f8[:])')
        def g2(x):
            return 1. - x[0] + (x[1] - 4.)**2

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c9(self):
        dim = 7
        ni = 4
        ne = 0
        lb = np.ones(dim) * -10.
        ub = np.ones(dim) * 10.
        fopt = 680.630057374402
        mact = 2

        @jit('f8(f8[:])')
        def f(x):
            res = (x[0] - 10.)**2 + 5. * (x[1] - 12.)**2 + x[2]**4 + 3. * (x[3] - 11.)**2
            res += 10. * x[4]**6 + 7. * x[5]**2 + x[6]**4 - 4. * x[5] * x[6] - 10. * x[5] - 8. * x[6]
            return res

        @jit('f8(f8[:])')
        def g1(x):
            return -127.0 + 2 * x[0]**2 + 3.0 * x[1]**4 + x[2] + 4.0 * x[3]**2 + 5.0 * x[4]

        @jit('f8(f8[:])')
        def g2(x):
            return -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2]**2 + x[3] - x[4]

        @jit('f8(f8[:])')
        def g3(x):
            return -196.0 + 23.0 * x[0] + x[1]**2 + 6.0 * x[5]**2 - 8.0 * x[6]

        @jit('f8(f8[:])')
        def g4(x):
            return 4.0 * x[0]**2 + x[1]**2 - 3.0 * x[0] * x[1] + 2.0 * x[2]**2 + 5.0 * x[5] - 11.0 * x[6]

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c10(self):
        dim = 8
        ni = 6
        ne = 0
        lb = np.asarray([100.] + [1000.]*2 + [10.]*5)
        ub = np.asarray([10000.]*3 + [1000]*5)
        fopt = 7049.24802052867
        mact = 6

        @jit('f8(f8[:])')
        def f(x):
            return np.sum(x[:3])

        @jit('f8(f8[:])')
        def g1(x):
            return -1.0 + 0.0025 * (x[3] + x[5])

        @jit('f8(f8[:])')
        def g2(x):
            return -1.0 + 0.0025 * (x[4] + x[6] - x[3])

        @jit('f8(f8[:])')
        def g3(x):
            return -1.0 + 0.01 * (x[7] - x[4])

        @jit('f8(f8[:])')
        def g4(x):
            return -x[0] * x[5] + 833.33252 * x[3] + 100.0 * x[0] - 83333.333

        @jit('f8(f8[:])')
        def g5(x):
            return -x[1] * x[6] + 1250.0 * x[4] + x[1] * x[3] - 1250.0 * x[3]

        @jit('f8(f8[:])')
        def g6(x):
            return -x[2] * x[7] + 1250000.0 + x[2] * x[4] - 2500.0 * x[4]

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c11(self):
        dim = 2
        ni = 0
        ne = 1
        lb = np.ones(dim) * -1.
        ub = np.ones(dim) * 1.
        fopt = 0.7499
        mact = 1

        @jit('f8(f8[:])')
        def f(x):
            return x[0]**2 + (x[1] - 1.)**2

        @jit('f8(f8[:])')
        def h1(x):
            return x[1] - x[0]**2

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c12(self):
        dim = 3
        ni = 1
        ne = 0
        lb = np.zeros(dim)
        ub = np.ones(dim) * 10.
        fopt = -1.0
        mact = 0

        @jit('f8(f8[:])')
        def f(x):
            return - (100. - np.sum((x - 5.)**2)) / 100.

        @jit('f8(f8[:])')
        def g1(x):
            tmp = (- np.asarray([np.arange(1.0, 10.0)] * 3).T + x)**2  # (9, dim)-array
            return np.min((np.outer(tmp[:, 2], np.ones(9)) + tmp[:, 1]).T
                          + np.asarray([np.outer(tmp[:, 0], np.ones(9))] * 9)) - 0.0625

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c13(self):
        dim = 5
        ni = 0
        ne = 3
        lb = np.asarray([-2.3]*2 + [-3.2]*3)
        ub = np.asarray([2.3]*2 + [3.2]*3)
        fopt = 0.053941514041898
        mact = 3

        @jit('f8(f8[:])')
        def f(x):
            return np.exp(np.prod(x))

        @jit('f8(f8[:])')
        def h1(x):
            return np.sum(x ** 2) - 10.

        @jit('f8(f8[:])')
        def h2(x):
            return np.prod(x[1:3]) - 5. * np.prod(x[3:])

        @jit('f8(f8[:])')
        def h3(x):
            return np.sum(x[:2] ** 3) + 1

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c14(self):
        dim = 10
        ni = 0
        ne = 3
        lb = np.zeros(dim) + 1e-15
        ub = np.ones(dim) * 10.
        fopt = - 47.7648884594915
        mact = 3

        c = np.asarray([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179])

        @jit('f8(f8[:])')
        def f(x):
            return np.sum( x * (c + np.log(x / np.sum(x))) )

        @jit('f8(f8[:])')
        def h1(x):
            return x[0] + 2.0 * x[1] + 2.0 * x[2] + x[5] + x[9] - 2.0

        @jit('f8(f8[:])')
        def h2(x):
            return x[3] + 2.0 * x[4] + x[5] + x[6] - 1.0

        @jit('f8(f8[:])')
        def h3(x):
            return x[2] + x[6] + x[7] + 2.0 * x[8] + x[9] - 1.0

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c15(self):
        dim = 3
        ni = 0
        ne = 2
        lb = np.zeros(dim)
        ub = np.ones(dim) * 10.
        fopt = 961.715022289961
        mact = 2

        @jit('f8(f8[:])')
        def f(x):
            return 1000. - x[0]**2 - 2. * x[1]**2 - x[2]**2 - np.prod(x[:2]) - np.prod(x[::2])

        @jit('f8(f8[:])')
        def h1(x):
            return np.sum(x ** 2) - 25.

        @jit('f8(f8[:])')
        def h2(x):
            return 8. * x[0] + 14. * x[1] + 7. * x[2] - 56

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c16(self):
        dim = 5
        ni = 38
        ne = 0
        lb = np.asarray([704.4148, 68.6, 0., 193., 25.])
        ub = np.asarray([906.3855, 288.88, 134.75, 287.0966, 84.1988])
        fopt = - 1.90515525853479
        mact = 4

        @jit('f8(f8[:])')
        def f(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            c12 = 0.995 * y9 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y13 - 5095.0
            y14 = y12 / c12
            y15 = 148000.0 - 331000.0 * y14 + 40.0 * y12 - 61.0 * y14 * y12
            y10 = 1.71 * x[0] - 0.452 * y3 + 0.580 * y2
            c13 = 2324.0 * y9 - 28740000.0 * y1
            y16 = 14130000.0 - 1328.0 * y9 - 531.0 * y10 + (c13 / c11)
            c14 = (y12 / y14) - (y12 / 0.52)
            c15 = 1.104 - 0.72 * y14
            c10 = 1.75 * y1 * 0.995 * x[0]
            y11 = x[0] * 12.3 / 752.3 + (c10 / c11)

            return - (0.0000005843 * y16 - 0.000117 * y13 - 0.1365 - 0.00002358 * y12 - 0.000001502 * y15 - 0.0321 * y11 - 0.004324 * y4 - 0.0001 * (c14 / c15) - 37.48 * (y1 / c11))

        @jit('f8(f8[:])')
        def g1(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6

            return - y3 + (0.28 / 0.72) * y4


        @jit('f8(f8[:])')
        def g2(x):
            return -1.5 * x[1] + x[2]

        @jit('f8(f8[:])')
        def g3(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0

            return -21.0 + 3496.0 * y1 / c11

        @jit('f8(f8[:])')
        def g4(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            c16 = y8 + x[4]

            return -(62212.0 / c16) + 110.6 + y0

        @jit('f8(f8[:])')
        def g5(x):
            return 213.1 - (x[1] + x[2] + 41.6)

        @jit('f8(f8[:])')
        def g6(x):
            return (x[1] + x[2] + 41.6) - 405.23

        @jit('f8(f8[:])')
        def g7(x):
            return 17.505 - (12.5 / (0.024 * x[3] - 4.62) + 12.0)

        @jit('f8(f8[:])')
        def g8(x):
            return (12.5 / (0.024 * x[3] - 4.62) + 12.0) - 1053.6667

        @jit('f8(f8[:])')
        def g9(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            return - c1 / c2 + 11.275

        @jit('f8(f8[:])')
        def g10(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            return c1 / c2 - 35.03

        @jit('f8(f8[:])')
        def g11(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            return - 19.0 * y2 + 214.228

        @jit('f8(f8[:])')
        def g12(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            return 19.0 * y2 - 665.585

        @jit('f8(f8[:])')
        def g13(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            return - c5 * c6 + 7.458

        @jit('f8(f8[:])')
        def g14(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            return c5 * c6 - 584.463

        @jit('f8(f8[:])')
        def g15(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            return 0.961 - y5

        @jit('f8(f8[:])')
        def g16(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            return y5 - 265.916

        @jit('f8(f8[:])')
        def g17(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            return 1.612 - y6

        @jit('f8(f8[:])')
        def g18(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            return y6 - 7.046

        @jit('f8(f8[:])')
        def g19(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            return 0.146 - y7

        @jit('f8(f8[:])')
        def g20(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            return y7 - 0.222

        @jit('f8(f8[:])')
        def g21(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            return 107.99 - y8

        @jit('f8(f8[:])')
        def g22(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            return y8 - 273.366

        @jit('f8(f8[:])')
        def g23(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            return 922.693 - y9

        @jit('f8(f8[:])')
        def g24(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            return y9 - 1286.105

        @jit('f8(f8[:])')
        def g25(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            y10 = 1.71 * x[0] - 0.452 * y3 + 0.580 * y2
            return 926.832 - y10

        @jit('f8(f8[:])')
        def g26(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            y10 = 1.71 * x[0] - 0.452 * y3 + 0.580 * y2
            return y10 - 1444.046

        @jit('f8(f8[:])')
        def g27(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            c10 = 1.75 * y1 * 0.995 * x[0]
            y11 = x[0] * 12.3 / 752.3 + (c10 / c11)
            return 18.766 - y11

        @jit('f8(f8[:])')
        def g28(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            c10 = 1.75 * y1 * 0.995 * x[0]
            y11 = 12.3 / 752.3 * x[0] + (c10 / c11)
            return y11 - 537.141

        @jit('f8(f8[:])')
        def g29(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            return 1072.163 - y12

        @jit('f8(f8[:])')
        def g30(x):
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            return y12 - 3247.039

        @jit('f8(f8[:])')
        def g31(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            return 8961.448 - y13

        @jit('f8(f8[:])')
        def g32(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            return y13 - 26844.086

        @jit('f8(f8[:])')
        def g33(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            c12 = 0.995 * y9 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y13 - 5095.0
            y14 = y12 / c12
            return 0.063 - y14

        @jit('f8(f8[:])')
        def g34(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            c12 = 0.995 * y9 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y13 - 5095.0
            y14 = y12 / c12
            return y14 - 0.386

        @jit('f8(f8[:])')
        def g35(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            c12 = 0.995 * y9 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y13 - 5095.0
            y14 = y12 / c12
            y15 = 148000.0 - 331000.0 * y14 + 40.0 * y12 - 61.0 * y14 * y12
            return 71084.33 - y15

        @jit('f8(f8[:])')
        def g36(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            c8 = y6 - (0.0663 * y6 / y7) - 0.3153
            y8 = (96.82 / c8) + 0.321 * y0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            y12 = c11 - 1.75 * y1
            y13 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + (146312.0 / (y8 + x[4]))
            c12 = 0.995 * y9 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y13 - 5095.0
            y14 = y12 / c12
            y15 = 148000.0 - 331000.0 * y14 + 40.0 * y12 - 61.0 * y14 * y12
            return y15 - 140000.0

        @jit('f8(f8[:])')
        def g37(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            c13 = 2324.0 * y9 - 28740000.0 * y1
            y10 = 1.71 * x[0] - 0.452 * y3 + 0.580 * y2
            y16 = 14130000.0 - 1328.0 * y9 - 531.0 * y10 + (c13 / c11)
            return 2802713.0 - y16

        @jit('f8(f8[:])')
        def g38(x):
            y0 = x[1] + x[2] + 41.6
            y1 = (12.5 / (0.024 * x[3] - 4.62)) + 12.0
            c1 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y1 * x[0]
            c2 = 0.052 * x[0] + 78.0 + 0.002377 * y1 * x[0]
            y2 = c1 / c2
            y3 = 19.0 * y2
            if abs(x[1]) < 1e-14:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) * 1e15 * np.sign(x[1])) + 0.6376 * y3 + 1.594 * y2
            else:
                c3 = 0.04782 * (x[0] - y2) + ((0.1956 * (x[0] - y2)**2) / x[1]) + 0.6376 * y3 + 1.594 * y2
            c4 = 100.0 * x[1]
            c5 = x[0] - y2 - y3
            c6 = 0.950 - (c3 / c4)
            y4 = c5 * c6
            c7 = (y4 + y3) * 0.995
            y6 = c7 / y0
            y7 = c7 / 3798.0
            y5 = x[0] - y4 - y3 - y2
            y9 = 1.29 * y4 + 1.258 * y3 + 2.29 * y2 + 1.71 * y5
            c11 = 0.995 * y9 + 1998.0
            c13 = 2324.0 * y9 - 28740000.0 * y1
            y10 = 1.71 * x[0] - 0.452 * y3 + 0.580 * y2
            y16 = 14130000.0 - 1328.0 * y9 - 531.0 * y10 + (c13 / c11)
            return y16 - 12146108.0

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c17(self):
        dim = 6
        ni = 0
        ne = 4
        lb = np.asarray([0.]*2 + [340.]*2 + [-1000.] + [0.])
        ub = np.asarray([400., 1000., 420., 420., 1000., 0.5236])
        fopt = 8853.53967480648
        mact = 4

        cos_147588 = np.cos(1.47588)
        sin_147588 = np.sin(1.47588)

        @jit('f8(f8[:])')
        def f(x):
            f1 = 30. * x[0] if 0 <= x[0] < 300 else 31. * x[0]
            f2 = 28. * x[1] if 0 <= x[1] < 100 else 29. * x[1] if 100 <= x[1] < 200 else 30. * x[1]
            return f1 + f2

        @jit('f8(f8[:])')
        def h1(x):
            return -x[0] + 300. + ( -np.prod(x[2:4]) * np.cos(1.48477 - x[5]) + 0.90798 * x[2]**2 * cos_147588) / 131.078

        @jit('f8(f8[:])')
        def h2(x):
            return - x[1] + ( -np.prod(x[2:4]) * np.cos(1.48477 + x[5]) + 0.90798 * x[3]**2 * cos_147588) / 131.078

        @jit('f8(f8[:])')
        def h3(x):
            return - x[4] + ( -np.prod(x[2:4]) * np.sin(1.48477 + x[5]) + 0.90798 * x[3]**2 * sin_147588) / 131.078

        @jit('f8(f8[:])')
        def h4(x):
            return 200 + ( -np.prod(x[2:4]) * np.sin(1.48477 - x[5]) + 0.90798 * x[2]**2 * sin_147588) / 131.078

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]

        cons_dict = {'ineq': [], 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c18(self):
        dim = 9
        ni = 13
        ne = 0
        lb = np.asarray([-10.]*8 + [0.])
        ub = np.asarray([10.]*8 + [20.])
        fopt = - 0.866025403784439
        mact = 6

        @jit('f8(f8[:])')
        def f(x):
            return - 0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])

        @jit('f8(f8[:])')
        def g1(x):
            return np.sum(x[2:4] ** 2) - 1

        @jit('f8(f8[:])')
        def g2(x):
            return x[8]**2 - 1

        @jit('f8(f8[:])')
        def g3(x):
            return np.sum(x[4:6] ** 2) - 1

        @jit('f8(f8[:])')
        def g4(x):
            return x[0]**2 + (x[1] - x[-1])**2 - 1

        @jit('f8(f8[:])')
        def g5(x):
            return (x[0] - x[4])**2 + (x[1] - x[5])**2 - 1

        @jit('f8(f8[:])')
        def g6(x):
            return (x[0] - x[6])**2 + (x[1] - x[7])**2 - 1

        @jit('f8(f8[:])')
        def g7(x):
            return (x[2] - x[4])**2 + (x[3] - x[5])**2 - 1

        @jit('f8(f8[:])')
        def g8(x):
            return (x[2] - x[6])**2 + (x[3] - x[7])**2 - 1

        @jit('f8(f8[:])')
        def g9(x):
            return x[6]**2 + (x[7] - x[8])**2 - 1

        @jit('f8(f8[:])')
        def g10(x):
            return np.prod(x[1:3]) - x[0] * x[3]

        @jit('f8(f8[:])')
        def g11(x):
            return - x[2] * x[8]

        @jit('f8(f8[:])')
        def g12(x):
            return x[4] * x[8]

        @jit('f8(f8[:])')
        def g13(x):
            return x[5] * x[6] - x[4] * x[7]

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c19(self):
        dim = 15
        ni = 5
        ne = 0
        lb = np.zeros(dim)
        ub = np.ones(dim) * 10.
        fopt = 32.6555929502463
        mact = 0

        A = np.asarray([[-16.0, 2.0, 0.0, 1.0, 0.0],
                        [0.0, -2.0, 0.0, 0.4, 2.0],
                        [-3.5, 0.0, 2.0, 0.0, 0.0],
                        [0.0, -2.0, 0.0, -4.0, -1.0],
                        [0.0, -9.0, -2.0, 1.0, -2.8],
                        [2.0, 0.0, -4.0, 0.0, 0.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -2.0, -3.0, -2.0, -1.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0]])
        B = np.asarray([-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0])
        C = np.asarray([[30.0, -20.0, -10.0, 32.0, -10.0],
                        [-20.0, 39.0, -6.0, -31.0, 32.0],
                        [-10.0, -6.0, 10.0, -6.0, -10.0],
                        [32.0, -31.0, -6.0, 39.0, -20.0],
                        [-10.0, 32.0, -10.0, -20.0, 30.0]])
        D = np.asarray([4.0, 8.0, 10.0, 6.0, 2.0])
        E = np.asarray([-15.0, -27.0, -36.0, -18.0, -12.0])

        @jit('f8(f8[:])')
        def f(x):
            return np.sum(C * np.outer(x[10:], x[10:])) + 2. * np.sum(D * x[10:]**3) - np.dot(B, x[:10])

        @jit('f8(f8[:])')
        def g1(x):
            return -2. * np.dot(C[:, 0], x[10:]) - 3.* D[0] * x[10+0]**2 - E[0] + np.dot(A[:, 0], x[:10])

        @jit('f8(f8[:])')
        def g2(x):
            return -2. * np.dot(C[:, 1], x[10:]) - 3.* D[1] * x[10+1]**2 - E[1] + np.dot(A[:, 1], x[:10])

        @jit('f8(f8[:])')
        def g3(x):
            return -2. * np.dot(C[:, 2], x[10:]) - 3.* D[2] * x[10+2]**2 - E[2] + np.dot(A[:, 2], x[:10])

        @jit('f8(f8[:])')
        def g4(x):
            return -2. * np.dot(C[:, 3], x[10:]) - 3.* D[3] * x[10+3]**2 - E[3] + np.dot(A[:, 3], x[:10])

        @jit('f8(f8[:])')
        def g5(x):
            return -2. * np.dot(C[:, 4], x[10:]) - 3.* D[4] * x[10+4]**2 - E[4] + np.dot(A[:, 4], x[:10])

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c20(self):
        dim = 24
        ni = 6
        ne = 14
        lb = np.zeros(dim)
        ub = np.ones(dim) * 10.
        fopt = 0.2049794002  # infeasible solution
        mact = 0
        warnings.warn("fopt is the value of infeasible solution and no feasible solution is found sofar.")

        A = np.asarray([0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09, 0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09])
        B = np.asarray([44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097])
        C = np.asarray([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64])
        D = np.asarray([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1])
        E = np.asarray([0.1, 0.3, 0.4, 0.3, 0.6, 0.3])
        K = 0.7302 * 530. * 14.7 / 40.

        @jit('f8(f8[:])')
        def f(x):
            return np.dot(A, x)

        @jit('f8(f8[:])')
        def g1(x):
            return (x[0] + x[0+12]) / (np.sum(x) + E[0])

        @jit('f8(f8[:])')
        def g2(x):
            return (x[1] + x[1+12]) / (np.sum(x) + E[1])

        @jit('f8(f8[:])')
        def g3(x):
            return (x[2] + x[2+12]) / (np.sum(x) + E[2])

        @jit('f8(f8[:])')
        def g4(x):
            return (x[3+3] + x[3+15]) / (np.sum(x) + E[3])

        @jit('f8(f8[:])')
        def g5(x):
            return (x[4+3] + x[4+15]) / (np.sum(x) + E[4])

        @jit('f8(f8[:])')
        def g6(x):
            return (x[5+3] + x[5+15]) / (np.sum(x) + E[5])

        @jit('f8(f8[:])')
        def h1(x):
            return x[0+12] / (B[0+12]*np.sum(x[12:] / B[12:])) - C[0]*x[0] / (40.* B[0]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h2(x):
            return x[1+12] / (B[1+12]*np.sum(x[12:] / B[12:])) - C[1]*x[1] / (40.* B[1]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h3(x):
            return x[2+12] / (B[2+12]*np.sum(x[12:] / B[12:])) - C[2]*x[2] / (40.* B[2]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h4(x):
            return x[3+12] / (B[3+12]*np.sum(x[12:] / B[12:])) - C[3]*x[3] / (40.* B[3]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h5(x):
            return x[4+12] / (B[4+12]*np.sum(x[12:] / B[12:])) - C[4]*x[4] / (40.* B[4]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h6(x):
            return x[5+12] / (B[5+12]*np.sum(x[12:] / B[12:])) - C[5]*x[5] / (40.* B[5]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h7(x):
            return x[6+12] / (B[6+12]*np.sum(x[12:] / B[12:])) - C[6]*x[6] / (40.* B[6]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h8(x):
            return x[7+12] / (B[7+12]*np.sum(x[12:] / B[12:])) - C[7]*x[7] / (40.* B[7]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h9(x):
            return x[8+12] / (B[8+12]*np.sum(x[12:] / B[12:])) - C[8]*x[8] / (40.* B[8]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h10(x):
            return x[9+12] / (B[9+12]*np.sum(x[12:] / B[12:])) - C[9]*x[9] / (40.* B[9]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h11(x):
            return x[10+12] / (B[10+12]*np.sum(x[12:] / B[12:])) - C[10]*x[10] / (40.* B[10]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h12(x):
            return x[11+12] / (B[11+12]*np.sum(x[12:] / B[12:])) - C[11]*x[11] / (40.* B[11]*np.sum(x[:12] / B[:12]))

        @jit('f8(f8[:])')
        def h13(x):
            return np.sum(x) - 1

        @jit('f8(f8[:])')
        def h14(x):
            return np.sum(x[:12] / D[:12]) + K * np.sum(x[12:] / B[12:]) - 1.671

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c21(self):
        dim = 7
        ni = 1
        ne = 5
        lb = np.asarray([0., 0., 0., 100., 6.3, 5.9, 4.5])
        ub = np.asarray([1000., 40., 40., 300., 6.7, 6.4, 6.25])
        fopt = 193.724510070035
        mact = 5

        @jit('f8(f8[:])')
        def f(x):
            return x[0]

        @jit('f8(f8[:])')
        def g1(x):
            return -x[0] + 35. * np.sum(x[1:3] ** 0.6)

        @jit('f8(f8[:])')
        def h1(x):
            return -300. * x[2] + 7500. * (x[4] - x[5]) + 25. * x[3] * (-x[4] + x[5]) + x[2] * x[3]

        @jit('f8(f8[:])')
        def h2(x):
            return 100. * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25. * x[3] * x[6] - 15536.5

        @jit('f8(f8[:])')
        def h3(x):
            return -x[4] + np.log(-x[3] + 900.)

        @jit('f8(f8[:])')
        def h4(x):
            return -x[5] + np.log(x[3] + 300.)

        @jit('f8(f8[:])')
        def h5(x):
            return -x[6] + np.log(-2. * x[3] + 700.)

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c22(self):
        dim = 22
        ni = 1
        ne = 19
        lb = np.asarray([0.]*7 + [100.]*2 + [100.01] + [100.]*2 + [0.]*3 + [0.01]*2 + [-4.7]*5)
        ub = np.asarray([20000.] + [1e6]*3 + [4e7]*3 + [299.99] + [399.99] + [300.] + [400.] + [600.] + [500.]*3 + [300.] + [400.] + [6.25]*5)
        fopt = 236.430975504001
        mact = 19

        @jit('f8(f8[:])')
        def f(x):
            return x[0]

        @jit('f8(f8[:])')
        def g1(x):
            return -x[0] + np.sum(x[1:4] ** 0.6)

        @jit('f8(f8[:])')
        def h1(x):
            return x[4] - 100000.0 * x[7] + 10000000.0

        @jit('f8(f8[:])')
        def h2(x):
            return x[5] + 100000.0 * x[7] - 100000.0 * x[8]

        @jit('f8(f8[:])')
        def h3(x):
            return x[6] + 100000.0 * x[8] - 50000000.0

        @jit('f8(f8[:])')
        def h4(x):
            return x[4] + 100000.0 * x[9] - 33000000.0

        @jit('f8(f8[:])')
        def h5(x):
            return x[5] + 100000 * x[10] - 44000000.0

        @jit('f8(f8[:])')
        def h6(x):
            return x[6] + 100000 * x[11] - 66000000.0

        @jit('f8(f8[:])')
        def h7(x):
            return x[4] - 120.0 * x[1] * x[12]

        @jit('f8(f8[:])')
        def h8(x):
            return x[5] - 80.0 * x[2] * x[13]

        @jit('f8(f8[:])')
        def h9(x):
            return x[6] - 40.0 * x[3] * x[14]

        @jit('f8(f8[:])')
        def h10(x):
            return x[7] - x[10] + x[15]

        @jit('f8(f8[:])')
        def h11(x):
            return x[8] - x[11] + x[16]

        @jit('f8(f8[:])')
        def h12(x):
            return -x[17] + np.log(x[9] - 100.0)

        @jit('f8(f8[:])')
        def h13(x):
            return -x[18] + np.log(-x[7] + 300.0)

        @jit('f8(f8[:])')
        def h14(x):
            return -x[19] + np.log(x[15])

        @jit('f8(f8[:])')
        def h15(x):
            return -x[20] + np.log (-x[8] + 400.0)

        @jit('f8(f8[:])')
        def h16(x):
            return -x[21] + np.log(x[16])

        @jit('f8(f8[:])')
        def h17(x):
            return -x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400.0

        @jit('f8(f8[:])')
        def h18(x):
            return x[7] - x[8] - x[10] + x[13] * x[19] - x[13] * x[20] + 400.0

        @jit('f8(f8[:])')
        def h19(x):
            return x[8] - x[11] - 4.60517 * x[14] + x[14] * x[21] + 100.0

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c23(self):
        dim = 9
        ni = 2
        ne = 4
        lb = np.asarray([0.]*8 + [0.01])
        ub = np.asarray([300.]*2 + [100.] + [200.] + [100.] + [300.] + [100.] + [200.] + [0.03])
        fopt = - 400.055099999999584
        mact = 4

        @jit('f8(f8[:])')
        def f(x):
            return -9.0 * x[4] - 15.0 * x[7] + 6.0 * x[0] + 16.0 * x[1] + 10.0 * (x[5] + x[6])

        @jit('f8(f8[:])')
        def g1(x):
            return x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4]

        @jit('f8(f8[:])')
        def g2(x):
            return x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7]

        @jit('f8(f8[:])')
        def h1(x):
            return x[0] + x[1] - x[2] - x[3]

        @jit('f8(f8[:])')
        def h2(x):
            return 0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3])

        @jit('f8(f8[:])')
        def h3(x):
            return x[2] + x[5] - x[4]

        @jit('f8(f8[:])')
        def h4(x):
            return x[3] + x[6] - x[7]

        loc_vals = locals()
        eqcons_list = [loc_vals['h'+str(j+1)] for j in range(ne)]
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': eqcons_list}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}


    def c24(self):
        dim = 2
        ni = 2
        ne = 0
        lb = np.zeros(dim)
        ub = np.asarray([3., 4.])
        fopt = - 5.50801327159536
        mact = 2

        @jit('f8(f8[:])')
        def f(x):
            return -x[0] - x[1]

        @jit('f8(f8[:])')
        def g1(x):
            return -2.0 * x[0]**4 + 8.0 * x[0]**3 - 8.0 * x[0]**2.0 + x[1] - 2.0

        @jit('f8(f8[:])')
        def g2(x):
            return -4.0 * x[0]**4 + 32.0 * x[0]**3 - 88.0 * x[0]**2.0 + 96.0 * x[0] + x[1] - 36.0

        loc_vals = locals()
        ineqcons_list = [loc_vals['g'+str(j+1)] for j in range(ni)]

        cons_dict = {'ineq': ineqcons_list, 'eq': []}

        return {'f': f, 'cons_dict': cons_dict, 'lb': lb, 'ub': ub, 'fopt': fopt, 'mact': mact, 'ni': ni, 'ne': ne}
