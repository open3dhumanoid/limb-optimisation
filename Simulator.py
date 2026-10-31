import itertools
import os
import numpy

try:
    if os.environ.get("VOXEL_BACKEND") == "numpy":
        raise ImportError
    import cupy as xp
    import cupyx.scipy.ndimage as ndimage
    GPU = True
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndimage
    GPU = False

OFFSETS = numpy.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]
)


def toHost(array):
    return xp.asnumpy(array) if GPU else numpy.asarray(array)

def elementStiffness(nu, pitch, incompatible):
    def strainColumn(gradient, component):
        column = numpy.zeros(6)
        column[component] = gradient[component]
        shear = {0: ((3, 1), (5, 2)), 1: ((3, 0), (4, 2)), 2: ((4, 1), (5, 0))}
        for row, axis in shear[component]:
            column[row] = gradient[axis]
        return column

    signs = 2 * OFFSETS - 1
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu  = 1 / (2 * (1 + nu))
    D = numpy.zeros(
        shape = (6, 6)
    )
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[3:, 3:] = numpy.eye(3) * mu

    kuu = numpy.zeros((24, 24))
    kua = numpy.zeros((24, 9))
    kaa = numpy.zeros((9, 9))
    gaussPoint = 1 / numpy.sqrt(3)
    for point in itertools.product((-gaussPoint, gaussPoint), repeat = 3):
        point = numpy.array(point)
        one = 1 + signs * point
        dN = 0.125 * signs * numpy.column_stack(
            [
                one[:, 1] * one[:, 2],
                one[:, 0] * one[:, 2],
                one[:, 0] * one[:, 1]
            ]
        ) * (2 / pitch)
        B = numpy.column_stack([strainColumn(dN[a], d) for a in range(8) for d in range(3)])
        Ba = numpy.column_stack(
            [strainColumn(numpy.eye(3)[m] * (-2 * point[m] * 2 / pitch), d) for m in range(3) for d in range(3)]
        )
        weight = (pitch / 2) ** 3
        kuu += B.T @ D @ B * weight
        kua += B.T @ D @ Ba * weight
        kaa += Ba.T @ D @ Ba * weight

    k0 = kuu - kua @ numpy.linalg.solve(kaa, kua.T) if incompatible else kuu
    k0 = 0.5 * (k0 + k0.T)

    # Strain at the element centre (the incompatible modes have zero strain there)
    dNcentre = 0.125 * signs * (2 / pitch)
    centreStrain = numpy.column_stack([strainColumn(dNcentre[a], d) for a in range(8) for d in range(3)])
    return k0, centreStrain, D

def gatherElements(u, shape):
    nx, ny, nz = shape
    ue = xp.stack(
        [u[:, :, ox:ox + nx, oy:oy + ny, oz:oz + nz] for ox, oy, oz in OFFSETS],
        axis = 1
    )
    return ue.reshape(u.shape[0], 24, -1)


def applyStiffness(u, E, k0):
    batch = u.shape[0]
    shape = E.shape[1:]
    fe = xp.matmul(k0, gatherElements(u, shape)) * E.reshape(E.shape[0], 1, -1)
    fe = fe.reshape(batch, 8, 3, *shape)
    nx, ny, nz = shape
    out = xp.zeros_like(u)
    for a, (ox, oy, oz) in enumerate(OFFSETS):
        out[:, :, ox:ox + nx, oy:oy + ny, oz:oz + nz] += fe[:, a]
    return out


def applyMasked(u, level):
    mask = level["mask"]
    return mask * applyStiffness(mask * u, level["E"], level["k0"]) + (1 - mask) * u


def stiffnessDiagonal(E, k0Diagonal):
    batch, nx, ny, nz = E.shape
    diagonal = xp.zeros((batch, 3, nx + 1, ny + 1, nz + 1))
    for a, (ox, oy, oz) in enumerate(OFFSETS):
        for d in range(3):
            diagonal[:, d, ox:ox + nx, oy:oy + ny, oz:oz + nz] += E * k0Diagonal[3 * a + d]
    return diagonal


def elementEnergy(u, k0, shape):
    ue = gatherElements(u, shape)
    return (ue * xp.matmul(k0, ue)).sum(axis = 1).reshape(u.shape[0], *shape)


def nodeAttached(elements):
    batch, nx, ny, nz = elements.shape
    attached = xp.zeros((batch, nx + 1, ny + 1, nz + 1), dtype = bool)
    for ox, oy, oz in OFFSETS:
        attached[:, ox:ox + nx, oy:oy + ny, oz:oz + nz] |= elements
    return attached


def batchDot(a, b):
    product = a * b
    return product.reshape(product.shape[0], -1).sum(axis = 1)


def expand(values):
    return values.reshape(-1, 1, 1, 1, 1)

def prolongAxis(coarse, axis):
    shape = list(coarse.shape)
    nodes = shape[axis]
    shape[axis] = 2 * (nodes - 1) + 1
    fine = xp.zeros(shape)
    index = [slice(None)] * coarse.ndim
    index[axis] = slice(0, None, 2)
    fine[tuple(index)] = coarse
    left, right = [slice(None)] * coarse.ndim, [slice(None)] * coarse.ndim
    left[axis], right[axis] = slice(0, nodes - 1), slice(1, nodes)
    index[axis] = slice(1, None, 2)
    fine[tuple(index)] = 0.5 * (coarse[tuple(left)] + coarse[tuple(right)])
    return fine


def restrictAxis(fine, axis):
    even, odd = [slice(None)] * fine.ndim, [slice(None)] * fine.ndim
    even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
    coarse = fine[tuple(even)].copy()
    oddValues = fine[tuple(odd)]
    nodes = coarse.shape[axis]
    left, right = [slice(None)] * fine.ndim, [slice(None)] * fine.ndim
    left[axis], right[axis] = slice(0, nodes - 1), slice(1, nodes)
    coarse[tuple(left)] += 0.5 * oddValues
    coarse[tuple(right)] += 0.5 * oddValues
    return coarse


def prolongNodes(coarse):
    for axis in (2, 3, 4):
        coarse = prolongAxis(coarse, axis)
    return coarse


def restrictNodes(fine):
    for axis in (2, 3, 4):
        fine = restrictAxis(fine, axis)
    return fine


def coarsenElements(E):
    batch, nx, ny, nz = E.shape
    return E.reshape(batch, nx // 2, 2, ny // 2, 2, nz // 2, 2).mean(axis = (2, 4, 6))

def levelMasks(elements, fixedNodes, levels):
    masks = []
    fixed = fixedNodes[None, None].astype(float)
    for level in range(levels):
        free = nodeAttached(elements) & ~(fixed[:, 0] > 0)
        masks.append(xp.repeat(free[:, None], 3, axis = 1).astype(float))
        if level < levels - 1:
            batch, nx, ny, nz = elements.shape
            elements = elements.reshape(batch, nx // 2, 2, ny // 2, 2, nz // 2, 2).any(axis = (2, 4, 6))
            fixed = restrictNodes(fixed)
    return masks


def denseInverse(E, k0, mask):
    batch, nx, ny, nz = E.shape
    nodes = (nx + 1) * (ny + 1) * (nz + 1)
    n = 3 * nodes
    elements = numpy.argwhere(numpy.ones((nx, ny, nz), dtype = bool))
    corners = elements[:, None, :] + OFFSETS[None]
    node = (corners[..., 0] * (ny + 1) + corners[..., 1]) * (nz + 1) + corners[..., 2]
    dof = (numpy.arange(3)[None, None, :] * nodes + node[:, :, None]).reshape(len(elements), 24)
    flat = (dof[:, :, None] * n + dof[:, None, :]).reshape(-1)
    flat = xp.asarray(flat)
    weights = (E.reshape(batch, -1, 1) * k0.reshape(1, 1, 576)).reshape(batch, -1)
    offsets = xp.arange(batch)[:, None] * (n * n)
    K = xp.bincount(
        (flat[None, :] + offsets).reshape(-1),
        weights = weights.reshape(-1),
        minlength = batch * n * n
    ).reshape(batch, n, n)
    m = mask.reshape(mask.shape[0], n)
    K = K * m[:, :, None] * m[:, None, :]
    index = xp.arange(n)
    K[:, index, index] += 1 - m
    return xp.linalg.inv(K)


def buildHierarchy(E, k0, k0Diagonal, masks, bound):
    hierarchy = []
    for level, mask in enumerate(masks):
        diagonal = stiffnessDiagonal(E, k0Diagonal) * mask + (1 - mask)
        h = {
            "E"       : E,
            "k0"      : k0,
            "mask"    : mask,
            "diagInv" : 1.0 / diagonal
        }
        if level < len(masks) - 1:
            h["upper"] = xp.full(E.shape[0], bound)
        hierarchy.append(h)
        if level < len(masks) - 1:
            E = coarsenElements(E)
            k0 = 2.0 * k0
            k0Diagonal = 2.0 * k0Diagonal
    last = hierarchy[-1]
    last["inverse"] = denseInverse(last["E"], last["k0"], last["mask"])
    return hierarchy


def chebyshev(h, r, x, degree):
    upper = expand(h["upper"])
    lower = upper / 30.0
    theta, delta = 0.5 * (upper + lower), 0.5 * (upper - lower)
    sigma = theta / delta
    rho = 1.0 / sigma
    residual = r if x is None else r - applyMasked(x, h)
    d = h["diagInv"] * residual / theta
    x = d if x is None else x + d
    for k in range(1, degree):
        residual = residual - applyMasked(d, h)
        rhoNew = 1.0 / (2.0 * sigma - rho)
        d = rhoNew * rho * d + (2.0 * rhoNew / delta) * h["diagInv"] * residual
        rho = rhoNew
        x = x + d
    return x


def vcycle(hierarchy, level, r, degree = 3):
    h = hierarchy[level]
    if level == len(hierarchy) - 1:
        batch = r.shape[0]
        inverse = h["inverse"]
        if inverse.shape[0] != batch:
            inverse = xp.broadcast_to(inverse, (batch,) + inverse.shape[1:])
        return xp.matmul(inverse, r.reshape(batch, -1, 1)).reshape(r.shape)
    x = chebyshev(h, r, None, degree)
    coarse = restrictNodes(r - applyMasked(x, h)) * hierarchy[level + 1]["mask"]
    x = x + prolongNodes(vcycle(hierarchy, level + 1, coarse, degree)) * h["mask"]
    return chebyshev(h, r, x, degree)


def solve(hierarchy, f, x0, tolerance, maxIterations):
    fine = hierarchy[0]
    f = f * fine["mask"]
    x = x0 * fine["mask"]
    r = f - applyMasked(x, fine)
    z = vcycle(hierarchy, 0, r)
    p = z.copy()
    rz = batchDot(r, z)
    fNorm = xp.sqrt(batchDot(f, f))
    fNorm = xp.where(fNorm > 0, fNorm, 1.0)
    relative = xp.sqrt(batchDot(r, r)) / fNorm
    iteration = 0
    while iteration < maxIterations and not bool((relative < tolerance).all()):
        iteration += 1
        Ap = applyMasked(p, fine)
        pAp = batchDot(p, Ap)
        alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1.0), 0.0)
        x += expand(alpha) * p
        r -= expand(alpha) * Ap
        relative = xp.sqrt(batchDot(r, r)) / fNorm
        z = vcycle(hierarchy, 0, r)
        rzNew = batchDot(r, z)
        beta = xp.where(rz > 0, rzNew / xp.where(rz > 0, rz, 1.0), 0.0)
        p = z + expand(beta) * p
        rz = rzNew
    return x, iteration, float(relative.max())

def buildModel(mesh, controls, coarseLimit = 600):
    matrix, pitch, origin = mesh
    matrix = numpy.asarray(matrix, dtype = bool)
    origin = numpy.asarray(origin, dtype = float)
    shape = numpy.array(matrix.shape)

    levels = 1
    while True:
        coarse = numpy.ceil(shape / 2 ** (levels - 1)).astype(int)
        if 3 * numpy.prod(coarse + 1) <= coarseLimit or coarse.min() < 2:
            break
        levels += 1
    padded = coarse * 2 ** (levels - 1)

    part = numpy.zeros(padded, dtype = bool)
    part[:shape[0], :shape[1], :shape[2]] = matrix
    passive = numpy.zeros(padded, dtype = bool)
    if controls.get("passive") is not None:
        passive[:shape[0], :shape[1], :shape[2]] = numpy.asarray(controls["passive"], dtype = bool)
    passive &= part
    active = part & ~passive
    if not active.any():
        raise ValueError(
            "! ERROR: no active voxels; the passive region covers the whole part"
        )

    nodeIndex = numpy.indices(padded + 1).reshape(3, -1).T
    coords = origin + pitch * nodeIndex
    attached = toHost(nodeAttached(xp.asarray(part[None])))[0]
    fixedNodes = numpy.asarray(controls["fixture"](coords)).reshape(padded + 1) & attached
    loadNodes = numpy.asarray(controls["loadSelector"](coords)).reshape(padded + 1) & attached
    if not fixedNodes.any() or not loadNodes.any():
        raise ValueError(
            f"! ERROR: empty boundary set (fixed {int(fixedNodes.sum())}, loaded {int(loadNodes.sum())})"
        )
    if (fixedNodes & loadNodes).any():
        raise ValueError(
            "! ERROR: load nodes overlap the fixture: load would be dropped"
        )

    model = {
        "shape"      : tuple(int(s) for s in shape),
        "padded"     : tuple(int(s) for s in padded),
        "levels"     : levels,
        "pitch"      : float(pitch),
        "origin"     : origin,
        "part"       : xp.asarray(part),
        "passive"    : xp.asarray(passive),
        "active"     : xp.asarray(active),
        "activeIndex": xp.asarray(numpy.flatnonzero(active)),
        "partIndex"  : xp.asarray(numpy.flatnonzero(part)),
        "fixedNodes" : xp.asarray(fixedNodes),
        "loadNodes"  : xp.asarray(loadNodes)
    }
    for name, incompatible in (("C3D8", False), ("C3D8I", True)):
        k0, centreStrain, D = elementStiffness(controls["nu"], pitch, incompatible)
        scale = numpy.sqrt(numpy.diag(k0))
        model[name] = {
            "bound"        : float(numpy.linalg.eigvalsh(k0 / numpy.outer(scale, scale)).max()),
            "k0"           : xp.asarray(k0),
            "k0Diagonal"   : numpy.diag(k0).copy(),
            "centreStrain" : xp.asarray(centreStrain),
            "D"            : xp.asarray(D)
        }
    return model

def fastLength(n):
    length = n
    while True:
        rest = length
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest //= factor
        if rest == 1:
            return length
        length += 1


def kernelSpectrum(radius, hat, fftShape):
    reach = int(numpy.floor(radius))
    offsets = numpy.arange(-reach, reach + 1)
    x, y, z = numpy.meshgrid(offsets, offsets, offsets, indexing = "ij")
    distance = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
    kernel = numpy.maximum(0.0, 1.0 - distance / radius) if hat else (distance <= radius).astype(float)
    grid = numpy.zeros(fftShape)
    grid[x % fftShape[0], y % fftShape[1], z % fftShape[2]] = kernel
    return xp.fft.rfftn(xp.asarray(grid))


def convolve(values, spectrum, fftShape):
    batch, nx, ny, nz = values.shape
    padded = xp.zeros((batch,) + tuple(fftShape))
    padded[:, :nx, :ny, :nz] = values
    result = xp.fft.irfftn(
        xp.fft.rfftn(padded, axes = (1, 2, 3)) * spectrum,
        s = tuple(fftShape),
        axes = (1, 2, 3)
    )
    return result[:, :nx, :ny, :nz]


def toGrid(values, index, shape):
    grid = xp.zeros((values.shape[0], int(numpy.prod(shape))))
    grid[:, index] = values
    return grid.reshape((values.shape[0],) + tuple(shape))


def fromGrid(grid, index):
    return grid.reshape(grid.shape[0], -1)[:, index]

def mmaBatch(iteration, x, xold1, xold2, df0dx, fval, dfdx, low, upp, move):
    asyinit, asydecr, asyincr, asymin, asymax = 0.5, 0.7, 1.2, 0.01, 10.0
    raa0, albefa, c, d = 1e-5, 0.1, 1000.0, 1.0

    if iteration <= 2:
        low = x - asyinit
        upp = x + asyinit
    else:
        zzz = (x - xold1) * (xold1 - xold2)
        factor = xp.where(zzz > 0, asyincr, xp.where(zzz < 0, asydecr, 1.0))
        low = xp.clip(x - factor * (xold1 - low), x - asymax, x - asymin)
        upp = xp.clip(x + factor * (upp - xold1), x + asymin, x + asymax)
    alfa = xp.maximum(xp.maximum(low + albefa * (x - low), x - move), 0.0)
    beta = xp.minimum(xp.minimum(upp - albefa * (upp - x), x + move), 1.0)

    ux1, xl1 = upp - x, x - low
    p0, q0 = xp.maximum(df0dx, 0.0), xp.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0
    p0, q0 = (p0 + pq0) * ux1 ** 2, (q0 + pq0) * xl1 ** 2
    P, Q = xp.maximum(dfdx, 0.0), xp.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa0
    P, Q = (P + PQ) * ux1 ** 2, (Q + PQ) * xl1 ** 2
    b = (P / ux1 + Q / xl1).sum(axis = 1) - fval

    def primal(lam):
        sp = xp.sqrt(p0 + lam[:, None] * P)
        sq = xp.sqrt(q0 + lam[:, None] * Q)
        return xp.clip((low * sp + upp * sq) / (sp + sq), alfa, beta)

    def dualGradient(lam):
        xl = primal(lam)
        return (P / (upp - xl) + Q / (xl - low)).sum(axis = 1) - b - xp.maximum(0.0, (lam - c) / d)

    zero = xp.zeros(x.shape[0])
    positive = dualGradient(zero) > 0
    hi = xp.ones(x.shape[0])
    for _ in range(60):
        grow = positive & (dualGradient(hi) > 0)
        if not bool(grow.any()):
            break
        hi = xp.where(grow, 10.0 * hi, hi)
    lo = zero.copy()
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        above = dualGradient(mid) > 0
        lo = xp.where(above, mid, lo)
        hi = xp.where(above, hi, mid)
    lam = xp.where(positive, 0.5 * (lo + hi), 0.0)
    return primal(lam), low, upp

def wuOptimiseBatch(mesh, population, context, controls, model = None):
    model = model or buildModel(mesh, controls)
    population = numpy.asarray(population, dtype = float)
    members = population.shape[0]
    alpha = xp.asarray(population[:, context["alpha"]])
    p     = xp.asarray(population[:, context["p"]])
    radii = population[:, context["R"]]

    r, gamma = controls["r"], controls["gamma"]
    E0, Emin = controls["E0"], controls["Emin"]
    betaEvery, betaMax = controls["betaEvery"], controls["betaMax"]
    maxIter, tolChange = controls["maxIter"], controls["tolChange"]
    move, verbose = controls["move"], controls["verbose"]
    tolerance = controls.get("solverTolerance", 1e-6)
    solverMaxIter = controls.get("solverMaxIter", 200)

    shape = model["padded"]
    element = model["C3D8"]
    active, activeIndex, part = model["active"], model["activeIndex"], model["part"]
    n = int(activeIndex.size)

    masks = levelMasks(part[None], model["fixedNodes"], model["levels"])
    loadNodes = model["loadNodes"]
    f = xp.zeros((1, 3) + tuple(s + 1 for s in shape))
    loadForce = numpy.asarray(controls["loadForce"], dtype = float)
    nLoad = int(loadNodes.sum())
    for d in range(3):
        f[0, d] = xp.where(loadNodes, loadForce[d] / nLoad, 0.0)

    reach = int(numpy.ceil(max(r, radii.max())))
    fftShape = [fastLength(s + reach) for s in shape]
    filterSpectrum = kernelSpectrum(r, True, fftShape)[None]
    spectra = {R: kernelSpectrum(R, False, fftShape) for R in numpy.unique(radii)}
    localSpectra = xp.stack([spectra[R] for R in radii])

    activeGrid = active[None].astype(float)
    filterDen = fromGrid(convolve(activeGrid, filterSpectrum, fftShape), activeIndex)
    localDen = fromGrid(convolve(activeGrid, localSpectra, fftShape), activeIndex)

    phi = xp.repeat(alpha[:, None], n, axis = 1)
    xold1, xold2 = phi.copy(), phi.copy()
    low, upp = xp.zeros_like(phi), xp.zeros_like(phi)
    beta = xp.full(members, float(controls["beta"]))
    done = xp.zeros(members, dtype = bool)
    E = xp.zeros((members,) + tuple(shape))
    E.reshape(members, -1)[:, model["partIndex"]] = E0
    u = xp.zeros((members, 3) + tuple(s + 1 for s in shape))
    c0 = None
    history = []

    def project(phiTilde, beta):
        tanhHalf = xp.tanh(beta / 2)[:, None]
        return (tanhHalf + xp.tanh(beta[:, None] * (phiTilde - 0.5))) / (2 * tanhHalf)

    for iteration in range(1, maxIter + 1):
        phiT = fromGrid(convolve(toGrid(phi, activeIndex, shape), filterSpectrum, fftShape), activeIndex) / filterDen
        rho = project(phiT, beta)
        Ea = Emin + rho ** gamma * (E0 - Emin)
        E.reshape(members, -1)[:, activeIndex] = Ea

        hierarchy = buildHierarchy(E, element["k0"], element["k0Diagonal"], masks, element["bound"])
        u, solverIterations, residual = solve(hierarchy, f, u, tolerance, solverMaxIter)
        c = 0.5 * batchDot(xp.broadcast_to(f, u.shape), u)
        energy = fromGrid(elementEnergy(u, element["k0"], shape), activeIndex)
        dcdRho = -0.5 * gamma * rho ** (gamma - 1) * (E0 - Emin) * energy

        rhoBar = xp.maximum(fromGrid(convolve(toGrid(rho, activeIndex, shape), localSpectra, fftShape), activeIndex) / localDen, 1e-12)
        meanP = xp.mean(rhoBar ** p[:, None], axis = 1)
        g = meanP ** (1 / p) / alpha - 1
        dgdRhoBar = (meanP ** (1 / p - 1))[:, None] * rhoBar ** (p[:, None] - 1) / (n * alpha[:, None])
        dgdRho = fromGrid(convolve(toGrid(dgdRhoBar / localDen, activeIndex, shape), localSpectra, fftShape), activeIndex)

        tanhHalf = xp.tanh(beta / 2)[:, None]
        dProj = beta[:, None] * (1 - xp.tanh(beta[:, None] * (phiT - 0.5)) ** 2) / (2 * tanhHalf)
        dcdPhi = fromGrid(convolve(toGrid(dcdRho * dProj / filterDen, activeIndex, shape), filterSpectrum, fftShape), activeIndex)
        dgdPhi = fromGrid(convolve(toGrid(dgdRho * dProj / filterDen, activeIndex, shape), filterSpectrum, fftShape), activeIndex)

        if c0 is None:
            c0 = c.copy()
        xNew, low, upp = mmaBatch(
            iteration,
            phi,
            xold1,
            xold2,
            (10.0 / c0)[:, None] * dcdPhi,
            g,
            dgdPhi,
            low,
            upp,
            move
        )
        xNew = xp.where(done[:, None], phi, xNew)
        xold2, xold1 = xold1, phi
        phi = xNew
        change = xp.abs(phi - xold1).max(axis = 1)

        history.append(
            {
                "c"       : toHost(c),
                "g"       : toHost(g),
                "beta"    : toHost(beta),
                "change"  : toHost(change),
                "solver"  : (solverIterations, residual)
            }
        )
        if verbose:
            sharpness = 4.0 / n * (rho * (1 - rho)).sum(axis = 1)
            print(
                f"it {iteration:4d}  c/c0 {float((c / c0).mean()):.4f}  max g {float(g.max()):+.4f}  "
                f"max change {float(change.max()):.4f}  sharpness {float(sharpness.mean()):.3f}  "
                f"CG {solverIterations} ({residual:.1e})"
            )

        trigger = ((iteration % betaEvery) == 0) | (change < tolChange)
        if betaMax is None:
            beta = xp.where(trigger & ~done, 2.0 * beta, beta)
        else:
            capped = beta >= betaMax
            done = done | (trigger & capped & (change < tolChange))
            beta = xp.where(trigger & ~capped & ~done, 2.0 * beta, beta)
        if bool(done.all()):
            break

    phiT = fromGrid(convolve(toGrid(phi, activeIndex, shape), filterSpectrum, fftShape), activeIndex) / filterDen
    rhoGrid = xp.zeros((members,) + tuple(shape))
    rhoGrid.reshape(members, -1)[:, model["partIndex"]] = 1.0
    rhoGrid.reshape(members, -1)[:, activeIndex] = project(phiT, beta)
    nx, ny, nz = model["shape"]
    return rhoGrid[:, :nx, :ny, :nz], {"iterations": iteration, "history": history, "model": model}

def simulateBatch(mesh, rho, controls, cases, model = None, tolerance = 1e-8, maxIterations = 500):
    model = model or buildModel(mesh, controls)
    shape = model["padded"]
    nx, ny, nz = model["shape"]
    rho = xp.asarray(rho)
    designs = rho.shape[0]
    E0 = controls["E0"]
    element = model["C3D8I"]

    solid = xp.zeros((designs,) + tuple(shape), dtype = bool)
    solid[:, :nx, :ny, :nz] = rho >= 0.5
    solid &= model["part"][None]
    nThreshold = solid.reshape(designs, -1).sum(axis = 1)
    for b in range(designs):
        labels, count = ndimage.label(solid[b])
        if int(count) > 1:
            sizes = xp.bincount(labels.ravel())
            sizes[0] = 0
            solid[b] = labels == int(sizes.argmax())
    nSolid = solid.reshape(designs, -1).sum(axis = 1)

    attached = nodeAttached(solid)
    loadMask = model["loadNodes"][None] & attached
    fixMask = model["fixedNodes"][None] & attached
    nLoad = loadMask.reshape(designs, -1).sum(axis = 1)
    nFix = fixMask.reshape(designs, -1).sum(axis = 1)
    valid = toHost((nLoad > 0) & (nFix > 0) & (nSolid > 0))

    results = [None] * designs
    for b in numpy.flatnonzero(~valid):
        results[b] = {"error": "! ERROR: the thresholded design lost its load or fixture nodes"}
    keep = numpy.flatnonzero(valid)
    if keep.size == 0:
        return results
    keepDevice = xp.asarray(keep)
    solid, loadMask, nLoad, nFix = solid[keepDevice], loadMask[keepDevice], nLoad[keepDevice], nFix[keepDevice]
    batch = keep.size

    E = xp.repeat(E0 * solid.astype(float), 3, axis = 0)
    masks = levelMasks(solid, model["fixedNodes"], model["levels"])
    masks = [xp.repeat(mask, 3, axis = 0) for mask in masks]
    nodesShape = tuple(s + 1 for s in shape)
    f = xp.zeros((batch, 3, 3) + nodesShape)
    for j in range(3):
        f[:, j, j] = loadMask / expand(nLoad)[:, 0]
    f = f.reshape((3 * batch, 3) + nodesShape)

    hierarchy = buildHierarchy(E, element["k0"], element["k0Diagonal"], masks, element["bound"])
    u, iterations, residual = solve(hierarchy, f, xp.zeros_like(f), tolerance, maxIterations)

    Ku = applyStiffness(u, E, element["k0"])
    energy = 0.5 * batchDot(u * masks[0], Ku).reshape(batch, 3)
    u = u.reshape((batch, 3, 3) + nodesShape)
    weights = loadMask[:, None, None].astype(float) / expand(nLoad)[:, None]
    complianceMatrix = (u * weights).reshape(batch, 3, 3, -1).sum(axis = 3).transpose(0, 2, 1)
    reactions = (Ku.reshape((batch, 3, 3) + nodesShape) * fixMask[:, None, None]).reshape(batch, 3, 3, -1).sum(axis = 3)
    diagonal = xp.diagonal(complianceMatrix, axis1 = 1, axis2 = 2)
    energyError = xp.abs(energy / (0.5 * diagonal) - 1).max(axis = 1)
    symmetryError = xp.sqrt(((complianceMatrix - complianceMatrix.transpose(0, 2, 1)) ** 2).sum(axis = (1, 2))) / \
                    xp.sqrt((complianceMatrix ** 2).sum(axis = (1, 2)))
    complianceMatrix = 0.5 * (complianceMatrix + complianceMatrix.transpose(0, 2, 1))

    strain = xp.matmul(element["centreStrain"], gatherElements(u.reshape((3 * batch, 3) + nodesShape), shape))
    stress = E0 * xp.matmul(element["D"], strain).reshape(batch, 3, 6, -1)
    solidFlat = solid.reshape(batch, -1)
    nElements = solidFlat.shape[1]
    vonMises = {}

    for name, force in cases.items():
        s = xp.tensordot(xp.asarray(force), stress, axes = ([0], [1]))
        sxx, syy, szz, sxy, syz, szx = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
        vm = xp.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
        ordered = xp.sort(xp.where(solidFlat, vm, -xp.inf), axis = 1)
        start = (nElements - nSolid)[:, None].astype(float)
        summary = {}
        for label, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
            position = start + q * (nSolid[:, None] - 1)
            lower = xp.floor(position).astype(int)
            upper = xp.minimum(lower + 1, nElements - 1)
            fraction = position - lower
            low = xp.take_along_axis(ordered, lower, axis = 1)
            high = xp.take_along_axis(ordered, upper, axis = 1)
            summary[label] = toHost((low + fraction * (high - low))[:, 0])
        summary["max"] = toHost(ordered[:, -1])
        vonMises[name] = summary

    host = {
        "complianceMatrix" : toHost(complianceMatrix),
        "reactions"        : toHost(reactions),
        "energyError"      : toHost(energyError),
        "symmetryError"    : toHost(symmetryError),
        "nSolid"           : toHost(nSolid),
        "nThreshold"       : toHost(nThreshold[keepDevice]),
        "nLoad"            : toHost(nLoad),
        "nFix"             : toHost(nFix)
    }
    partVoxels = int(model["part"].sum())
    for i, b in enumerate(keep):
        results[b] = {
            "nElements"        : int(host["nSolid"][i]),
            "islandFraction"   : 1.0 - host["nSolid"][i] / max(int(host["nThreshold"][i]), 1),
            "solidFraction"    : host["nSolid"][i] / partVoxels,
            "volume"           : host["nSolid"][i] * model["pitch"] ** 3,
            "nodeSetSizes"     : {"FIX": int(host["nFix"][i]), "LOAD": int(host["nLoad"][i])},
            "complianceMatrix" : host["complianceMatrix"][i],
            "unitReactions"    : host["reactions"][i],
            "energyError"      : float(host["energyError"][i]),
            "symmetryError"    : float(host["symmetryError"][i]),
            "solverIterations" : iterations,
            "solverResidual"   : residual,
            "vonMises"         : {name: {k: float(v[i]) for k, v in summary.items()} for name, summary in vonMises.items()}
        }
    return results