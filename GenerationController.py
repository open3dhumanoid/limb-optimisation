import pandas
import numpy
import os

class GenerationController:
    _GEN_PATH = "generations/"
    _STL_PATH = "stls/"
    def __init__(self, saveStep, popAmount, max, min, seed):
        self._saveStep = saveStep
        self._genCount = self._readGenerationCount()

        if self._genCount == 0:
            self._generation = self._initProgenitorGeneration(
                popAmount = popAmount,
                max       = max,
                min       = min,
                seed      = seed
            )
        else:
            self._generation = numpy.load(
                os.path.join(GenerationController._GEN_PATH, str(self._genCount))
            )

    def _readGenerationCount(self) -> int:
        with open(os.path.join(GenerationController._GEN_PATH, "generationCount.txt"), "r") as f:
            genCount = int(f.read())

        if genCount < 0:
            raise ValueError(
                f"! ERROR: generationCount has an impossible value ({genCount})"
            )
        else:
            return genCount

    def _initProgenitorGeneration(self, popAmount:int, max:float, min:float, seed:int) -> numpy.ndarray:
        """
        Function to initialise a progenitor generation for the Vitruvian porous bone topological optimisation genetic
        algorithm. The function generates a matrix with the dimensions of (popAmount, 3), where each row is initialised
        with random values using numpy default_rng generator. The function also requires a seed parameter to initialise the
        aforementioned generator.
        """
        generator = numpy.random.default_rng(seed = seed)
        gen = numpy.zeros(
            shape = (popAmount, 3),
            dtype = numpy.float64
        )
        for i in range(popAmount):
            gen[i, 0] = generator.uniform(min, max)
            gen[i, 1] = generator.uniform(min, max)
            gen[i, 2] = generator.uniform(min, max)

        return gen

    def run(self):
        ...