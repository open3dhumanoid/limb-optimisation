import Auxiliary
import numpy
import Optimisers
import scipy.ndimage

LENGTH  = 272.0
PITCH = 2.0


def fixture(coords):
    # whole bottom face, z = 0
    return coords[:, 0] <= 0.5 * PITCH


def loadSelector(coords):
    # 0.3 x 0.3 patch in the centre of the top face, z = SIDE
    return coords[:, 0] >= LENGTH - 0.5 * PITCH

matrix, _, _ = Auxiliary.voxelise("STLs/forearm.stl", PITCH)
passive = matrix & ~scipy.ndimage.binary_erosion(
    matrix,
    iterations = 2
)

controls = {
    "fixture"      : fixture,
    "loadSelector" : loadSelector,
    "loadForce"    : (0.0, 0.0, -1.0),
    "passive"      : passive,
    "E0"           : 3500.0,
    "nu"           : 0.36,
    "Emin"         : 1e-9 * 3500.0,
    "r"            : 1.5,
    "gamma"        : 3.0,
    "beta"         : 1.0,
    "betaEvery"    : 40,
    "betaMax"      : None,
    "maxIter"      : 300,
    "tolChange"    : 1e-3,
    "move"         : 0.2,
    "verbose"      : True
}
genotype = numpy.array(
    [0.3, 16.0, 4.0]
)
context = {
    "alpha" : 0,
    "p"     : 1,
    "R"     : 2
}

Auxiliary.view(
    genotype = genotype,
    stl      = "STLs/forearm.stl",
    pitch    = PITCH,
    context  = context,
    controls = controls,
    optimise = Optimisers.wuOptimise
)