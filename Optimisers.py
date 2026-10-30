import numpy
import scipy.sparse
from scipy.signal import fftconvolve
import mmapy
import pypardiso

def gyroidOptimise(mesh, params, context, controls):
    matrix, pitch, origin = mesh
    alpha = params[context["alpha"]]
    R = params[context["R"]]
    passive = controls.get("passive")

    period = max(
        4.0 * R,
        4.0
    ) * pitch
    k = 2.0 * numpy.pi / period

    index = numpy.indices(matrix.shape)
    x, y, z = origin[:, None, None, None] + (index + 0.5) * pitch

    g = (
            numpy.sin(k * x) * numpy.cos(k * y)
            + numpy.sin(k * y) * numpy.cos(k * z)
            + numpy.sin(k * z) * numpy.cos(k * x)
    )

    active = matrix if passive is None else matrix & ~passive
    threshold = numpy.quantile(
        numpy.abs(g[active]),
        alpha
    )

    rho = (numpy.abs(g) <= threshold).astype(float)
    rho[~matrix] = 0.0
    if passive is not None:
        rho[passive] = 1.0
    return rho

def wuOptimise(mesh, params, context, controls):
    def convolveActive(values, kernel, matrix, activeIndex):
        grid = numpy.zeros(matrix.shape)
        grid[activeIndex] = values
        return fftconvolve(
            grid,
            kernel,
            mode="same"
        )[activeIndex]

    matrix, pitch, origin = mesh
    alpha = params[context["alpha"]]
    R = params[context["R"]]
    p = params[context["p"]]

    r = controls["r"]
    gamma = controls["gamma"]
    E0 = controls["E0"]
    nu = controls["nu"]
    Emin = controls["Emin"]
    beta = controls["beta"]
    betaEvery = controls["betaEvery"]
    betaMax = controls["betaMax"]
    maxIter = controls["maxIter"]
    tolChange = controls["tolChange"]
    move = controls["move"]
    verbose = controls["verbose"]
    passive = controls["passive"]

    matrix = numpy.asarray(
        matrix,
        dtype=bool
    )
    origin = numpy.asarray(
        origin,
        dtype=float
    )
    if passive is None:
        passive = numpy.zeros_like(matrix)
    else:
        passive = numpy.asarray(
            passive,
            dtype=bool
        ) & matrix
    active = matrix & ~passive

    offsets = numpy.array(
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
    signs = 2 * offsets - 1
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1 / (2 * (1 + nu))

    D = numpy.zeros(
        shape=(6, 6)
    )
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[3:, 3:] = numpy.eye(3) * mu
    k0 = numpy.zeros(
        shape=(24, 24)
    )
    gaussPoint = 1 / numpy.sqrt(3)
    for xi in (-gaussPoint, gaussPoint):
        for eta in (-gaussPoint, gaussPoint):
            for zeta in (-gaussPoint, gaussPoint):
                one = 1 + signs * numpy.array([xi, eta, zeta])
                dN = 0.125 * signs * numpy.column_stack(
                    [
                        one[:, 1] * one[:, 2],
                        one[:, 0] * one[:, 2],
                        one[:, 0] * one[:, 1]
                    ]
                ) * (2 / pitch)
                B = numpy.zeros(
                    shape=(6, 24)
                )
                B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[:, 0], dN[:, 1], dN[:, 2]
                B[3, 0::3], B[3, 1::3] = dN[:, 1], dN[:, 0]
                B[4, 1::3], B[4, 2::3] = dN[:, 2], dN[:, 1]
                B[5, 0::3], B[5, 2::3] = dN[:, 2], dN[:, 0]

                k0 += B.T @ D @ B * (pitch / 2) ** 3

    nx, ny, nz = matrix.shape
    elements = numpy.argwhere(matrix)
    corners = elements[:, None, :] + offsets[None]
    gridNode = corners[..., 0] + (nx + 1) * (corners[..., 1] + (ny + 1) * corners[..., 2])
    gridNodes, localNode = numpy.unique(
        gridNode,
        return_inverse=True
    )
    localNode = localNode.reshape(gridNode.shape)
    nodeCoords = origin + pitch * numpy.column_stack(
        [
            gridNodes % (nx + 1),
            (gridNodes // (nx + 1)) % (ny + 1),
            gridNodes // ((nx + 1) * (ny + 1)),
        ]
    )

    dofN = 3 * len(gridNodes)
    dofE = (3 * localNode[:, :, None] + numpy.arange(3)).reshape(len(elements), 24)

    rows = numpy.repeat(
        dofE,
        24,
        axis=1
    ).ravel()
    cols = numpy.tile(
        dofE,
        (1, 24)
    ).ravel()
    keys, scatter = numpy.unique(
        rows.astype(numpy.int64) * dofN + cols,
        return_inverse=True
    )
    csrIndices = (keys % dofN).astype(numpy.int32)
    csrIndptr = numpy.searchsorted(
        keys // dofN,
        numpy.arange(dofN + 1)
    ).astype(numpy.int32)
    k0Flat = k0.ravel()

    fixNodes = numpy.flatnonzero(
        controls["fixture"](nodeCoords)
    )
    loadNodes = numpy.flatnonzero(
        controls["loadSelector"](nodeCoords)
    )
    if fixNodes.size == 0 or loadNodes.size == 0:
        raise ValueError(
            f"! ERROR: empty boundary set (fixed {fixNodes.size}, loaded {loadNodes.size})"
        )
    force = numpy.zeros(dofN)
    fixed = numpy.zeros(
        dofN,
        dtype=bool
    )
    loadForce = numpy.asarray(
        controls["loadForce"],
        dtype=float
    )
    for d in range(3):
        force[3 * loadNodes + d] = loadForce[d] / len(loadNodes)
        fixed[3 * fixNodes + d] = True
    free = numpy.flatnonzero(~fixed)

    centredCoords = nodeCoords - nodeCoords.mean(axis=0)
    rigId = numpy.zeros(
        shape=(dofN, 6)
    )
    for d in range(3):
        rigId[d::3, d] = 1.0
    rigId[0::3, 3], rigId[1::3, 3] = -centredCoords[:, 1], centredCoords[:, 0]
    rigId[1::3, 4], rigId[2::3, 4] = -centredCoords[:, 2], centredCoords[:, 1]
    rigId[0::3, 5], rigId[2::3, 5] = centredCoords[:, 2], -centredCoords[:, 0]
    rigId = rigId[free]

    kernels = []
    for radius, hat in ((r, True), (R, False)):
        reach = numpy.arange(
            -int(numpy.floor(radius)),
            int(numpy.floor(radius)) + 1
        )
        x, y, z = numpy.meshgrid(
            reach,
            reach,
            reach,
            indexing="ij"
        )
        dist = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
        if hat:
            kernels.append(
                numpy.maximum(
                    0.0,
                    1.0 - dist / radius
                )
            )
        else:
            kernels.append(
                (dist <= radius).astype(float)
            )
    filterKernel, localKernel = kernels
    activeIndex = tuple(
        numpy.argwhere(active).T
    )
    activePos = numpy.flatnonzero(active[tuple(elements.T)])
    n = len(activePos)

    filterDen = convolveActive(
        values=numpy.ones(n),
        kernel=filterKernel,
        matrix=matrix,
        activeIndex=activeIndex,
    )
    localDen = convolveActive(
        values=numpy.ones(n),
        kernel=localKernel,
        matrix=matrix,
        activeIndex=activeIndex,
    )

    phi = numpy.full(
        n,
        alpha
    )
    phi01d1, phi01d2 = phi.copy(), phi.copy()
    lowerBound = numpy.zeros(
        shape=(n, 1)
    )
    upperBound = numpy.zeros(
        shape=(n, 1)
    )
    rhoeE = numpy.ones(len(elements))
    uPrev, c0 = numpy.zeros(dofN), None

    for i in range(1, maxIter + 1):
        phiT = convolveActive(
            values=phi,
            kernel=filterKernel,
            matrix=matrix,
            activeIndex=activeIndex,
        ) / filterDen
        halfTanh = numpy.tanh(beta / 2)
        rho = (halfTanh + numpy.tanh(beta * (phiT - 0.5))) / (2 * halfTanh)
        rhoeE[activePos] = rho
        E = Emin + rhoeE ** gamma * (E0 - Emin)

        data = numpy.bincount(
            scatter,
            weights=(E[:, None] * k0Flat[None, :]).ravel(),
            minlength=len(csrIndices)
        )
        K = scipy.sparse.csr_matrix(
            (data, csrIndices, csrIndptr),
            shape=(dofN, dofN)
        )
        Kff = K[free][:, free].tocsr()
        u = numpy.zeros(dofN)

        u[free] = pypardiso.spsolve(
            Kff,
            force[free]
        )
        uPrev = u
        c = 0.5 * float(force @ u)
        ue = u[dofE[activePos]]
        dcdRho = -0.5 * gamma * rho ** (gamma - 1) * (E0 - Emin) * numpy.einsum("ij,jk,ik->i", ue, k0, ue)

        rhoBar = numpy.maximum(
            convolveActive(
                values=rho,
                kernel=localKernel,
                matrix=matrix,
                activeIndex=activeIndex,
            ) / localDen,
            1e-12
        )
        meanP = numpy.mean(rhoBar ** p)
        g = meanP ** (1 / p) / alpha - 1
        dgdRhoBar = meanP ** (1 / p - 1) * rhoBar ** (p - 1) / (n * alpha)
        dgdRho = convolveActive(
            values=dgdRhoBar / localDen,
            kernel=localKernel,
            matrix=matrix,
            activeIndex=activeIndex,
        )

        dProj = beta * (1 - numpy.tanh(beta * (phiT - 0.5)) ** 2) / (2 * halfTanh)
        dcdPhi = convolveActive(
            values=dcdRho * dProj / filterDen,
            kernel=filterKernel,
            matrix=matrix,
            activeIndex=activeIndex,
        )
        dgdPhi = convolveActive(
            values=dgdRho * dProj / filterDen,
            kernel=filterKernel,
            matrix=matrix,
            activeIndex=activeIndex,
        )

        if c0 is None:
            c0 = c
        xmma, *_, lowerBound, upperBound = mmapy.mmasub(
            1, n, i, phi[:, None], numpy.zeros((n, 1)), numpy.ones((n, 1)),
            phi01d1[:, None], phi01d2[:, None], 10.0 * c / c0,
            (10.0 / c0 * dcdPhi)[:, None],
            numpy.array([[g]]), dgdPhi[None, :],
            lowerBound, upperBound, 1.0, numpy.zeros((1, 1)), numpy.full((1, 1), 1000.0),
            numpy.ones((1, 1)),
            move=move
        )
        phi01d2, phi01d1 = phi01d1, phi
        phi = xmma.ravel()
        change = float(
            numpy.max(
                numpy.abs(phi - phi01d1)
            )
        )

        if verbose:
            sharpness = 4.0 / n * numpy.sum(rho * (1 - rho))
            print(f"it {i:4d}  beta {beta:6.1f}  c {c:.5g}  g {g:+.4f}  change {change:.4f}  sharpness {sharpness:.3f}")

        if i % betaEvery == 0 or change < tolChange:
            if betaMax is not None and beta >= betaMax:
                if change < tolChange:
                    break
            else:
                beta *= 2

    phiT = convolveActive(
        values=phi,
        kernel=filterKernel,
        matrix=matrix,
        activeIndex=activeIndex,
    ) / filterDen
    halfTanh = numpy.tanh(beta / 2)
    rhoGrid = numpy.zeros(matrix.shape)
    rhoGrid[matrix] = 1.0
    rhoGrid[activeIndex] = (halfTanh + numpy.tanh(beta * (phiT - 0.5))) / (2 * halfTanh)
    return rhoGrid

