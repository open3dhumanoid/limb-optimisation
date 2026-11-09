import copy
import random
import numpy

class Evolver:
    def __init__(self, tournamentSize, kFittest, mutationRate, sigma, areParamsDynamic, seed):
        self._tournamentSize      = 3
        self._baseMutationRate    = mutationRate
        self._currentMutationRate = mutationRate
        self._baseSigma           = sigma
        self._currentSigma        = sigma
        self._areParamsDynamic    = areParamsDynamic
        self._kFittest            = kFittest
        self._rng                 = numpy.random.default_rng(seed = seed)

    def _calcDiversity(self, pop:numpy.ndarray) -> tuple:
        def calcMean(vec:numpy.ndarray) -> float:
            rows, cols = pop.shape
            if cols != 1:
                raise RuntimeError(
                    f"Error occurred when calculating mean value of a population's genotype. Value is not 1"
                )
            else:
                return vec.mean()

        rows, cols    = pop.shape
        alphaVariance = 0.0
        pVariance     = 0.0
        rVariance     = 0.0
        for j in range(0, cols):
            sum        = 0.0
            mean = calcMean(pop[:, j])
            for i in range(0, rows):
                sum += (pop[i, j] - mean)**2

            if j == 0:
                alphaVariance = sum / rows
            elif j == 1:
                pVariance     = sum / rows
            elif j == 2:
                rVariance     = sum / rows
            else:
                raise RuntimeError(
                    f"Error occurred when calculating diverstiy of population: a value of {j} was encountered when a maximum of 2 were encountered\n    - Suggestion: more parameters than the 3 under study were declared" 
                )

        return alphaVariance, pVariance, rVariance

    def _gaussianMutate(self, genotype:numpy.ndarray) -> numpy.ndarray:
        noise        = numpy.random.rand(genotype.shape) * self._currentSigma
        mutationMask = (numpy.random.rand(genotype.shape) < self._currentMutationRate).float()
        return genotype + (noise * mutationMask)

    def _reproduce(self, parent1:numpy.ndarray, parent2:numpy.ndarray, reproductionRule:str) -> numpy.ndarray:
        if reproductionRule == "point_crossover":
            parentRows, parentCols = parent1.shape
            crossoverPoint = numpy.random.randint(
                low  = 0,
                high = parentCols,
                size = 1
            )
            child = numpy.zeros(
                shape = (parentRows, parentCols)
            )
            child[0:crossoverPoint] = parent1[0:crossoverPoint]
            child[crossoverPoint:]  = parent2[crossoverPoint:]
            return child
        elif reproductionRule == "blend_crossover":
            alpha = numpy.random.uniform(
                low  = 0.3,
                high = 0.7,
                size = 1
            ).float()
            return alpha * parent1 + (1 - alpha) * parent2
        else:
            raise RuntimeError(
                f"Error occurred when attempting to call _reproduce function: a non-existent reproductionRule ({reproductionRule}) was passed on call"
            )

    def _tournamentSelection(self, pop:numpy.ndarray, fitnesses:numpy.ndarray):
        popRows, popCols         = pop.shape
        fitnessRows, fitnessCols = fitnesses.shape
        if popRows != fitnessRows:
            raise RuntimeError(
                f"Error occurred when performing tournament selecton: popRows ({popRows}) does not correspond to fitnessRows ({fitnessRows})"
            )
        else:
            i = self._rng.choice(
                popRows,
                size = min(self._tournamentSize, popRows),
                replace = False
            )
            return pop[i[numpy.argmax(fitnesses[i])]]

