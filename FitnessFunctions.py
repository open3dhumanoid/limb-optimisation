import numpy

def calcStressFitness(metrics, yieldStrength, targetSafetyWeight, penalty = 10.0) -> float:
    if "error" in metrics:
        return -numpy.inf
    else:
        safetyTerm = yieldStrength / metrics["worstStressP99"]
        shortFall  = max(0.0, targetSafetyWeight / safetyTerm - 1.0)
        return -(metrics["solidFraction"] + penalty * shortFall ** 2)