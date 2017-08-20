"""

Differential evolution (DE)
===========================

This implements only a single class: :class:`DiffEvolOptimizer` that follows the
differential evolution optimization method by Storn & Price (Storn, R., Price,
K., Journal of Global Optimization 11: 341--359, 1997)

**Main assuption**: the heuristic suppose a continuous parameter space.
"""
from __future__ import division

import numpy as np
from numpy.random import random as _random, randint as _randint

__all__ = ['DiffEvolOptimizer']

try:
    from builtins import xrange
except ImportError:
    xrange = range


class DiffEvolOptimizer(object):
    """
    Implements the differential evolution optimization method by Storn & Price
    (Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)

    Attributes
    ----------
    fun: callable
       the function to be minimized

    bounds: sequence of tuples
        parameter bounds as [ndim, 2] sequence

    npop: int
        the size of the population
        5 * ndim - 10 * ndim are usual values

    F: float, optional (default=0.5)
        the difference amplification factor.
        Values of 0.5-0.8 are good in most cases.

    C: float, optional (default=0.5)
        The cross-over probability. Use 0.9 to test for fast convergence, and smaller
        values (~0.1) for a more elaborate search.

    seed: int, optional (default=0)
        Random seed, for reproductible results

    maximize: bool, optional (default=False)
        Switch setting whether to maximize or minimize the function.
        Defaults to minimization.

    population: ndarray
        The population parameter vector
    """
    def __init__(self, fun, bounds, npop, F=0.5, C=0.5, seed=None, maximize=False):
        """ Constructor

        Parameters
        ----------
        fun: callable
        the function to be minimized

        bounds: sequence of tuples
            parameter bounds as [ndim, 2] sequence

        npop: int
            the size of the population
            5 * ndim - 10 * ndim are usual values

        F: float, optional (default=0.5)
            the difference amplification factor.
            Values of 0.5-0.8 are good in most cases.

        C: float, optional (default=0.5)
            The cross-over probability. Use 0.9 to test for fast convergence, and smaller
            values (~0.1) for a more elaborate search.

        seed: int, optional (default=None)
            Random seed, for reproductible results

        maximize: bool, optional (default=False)
            Switch setting whether to maximize or minimize the function.
            Defaults to minimization.
        """
        if seed is not None:
            np.random.seed(seed)

        self.fun = fun
        self.bounds = np.asarray(bounds)
        self.npop = npop
        self.F = F
        self.C = C

        self.ndim  = (self.bounds).shape[0]
        self.m  = -1 if maximize else 1

        bl = self.bounds[:, 0]
        bw = self.bounds[:, 1] - self.bounds[:, 0]
        self.population = bl[None, :] + _random((self.npop, self.ndim)) * bw[None, :]
        self.fitness = np.empty(npop, dtype=float)
        self._minidx = None

    def step(self):
        """Take a step in the optimization"""
        rnd_cross = _random((self.npop, self.ndim))
        for i in xrange(self.npop):
            t0, t1, t2 = i, i, i
            while t0 == i:
                t0 = _randint(self.npop)
            while t1 == i or t1 == t0:
                t1 = _randint(self.npop)
            while t2 == i or t2 == t0 or t2 == t1:
                t2 = _randint(self.npop)

            v = self.population[t0,:] + self.F * (self.population[t1,:] - self.population[t2,:])

            crossover = rnd_cross[i] <= self.C
            u = np.where(crossover, v, self.population[i,:])

            ri = _randint(self.ndim)
            u[ri] = v[ri]

            ufit = self.m * self.fun(u)

            if ufit < self.fitness[i]:
                self.population[i,:] = u
                self.fitness[i] = ufit

    @property
    def value(self):
        """The best-fit value of the optimized function"""
        return self.fitness[self._minidx]

    @property
    def location(self):
        """The best-fit solution"""
        return self.population[self._minidx, :]

    @property
    def index(self):
        """Index of the best-fit solution"""
        return self._minidx

    def optimize(self, ngen=1):
        """Run the optimizer for ``ngen`` generations

        Parameters
        ----------
        ngen: int
            number of iterations

        Returns
        -------
        population: ndarray
            population locations, [Npop x Ndim]

        fitness: ndarray
            population values, [Npop]
        """
        for i in xrange(self.npop):
            self.fitness[i] = self.m * self.fun(self.population[i,:])

        for j in xrange(ngen):
            self.step()

        self._minidx = np.argmin(self.fitness)
        return self.population[self._minidx,:], self.fitness[self._minidx]

    def iteroptimize(self, ngen=1):
        """Iterator to the optimizer for ``ngen`` generations

        Parameters
        ----------
        ngen: int
            number of iterations

        Returns
        -------
        population: ndarray
            population locations, [Npop x Ndim]

        fitness: ndarray
            population values, [Npop]
        """

        for i in xrange(self.npop):
            self.fitness[i] = self.m * self.fun(self.population[i,:])

        for j in xrange(ngen):
            self.step()
            self._minidx = np.argmin(self.fitness)
            yield self.population[self._minidx,:], self.fitness[self._minidx]

    def __call__(self, ngen=1):
        return self.iteroptimize(ngen)
