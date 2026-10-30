import numpy
import trimesh
import pyvista

def voxelise(stl, pitch) -> tuple:
    mesh = trimesh.load(
        file_obj=stl,
        force="mesh"
    )
    if not mesh.is_watertight:
        raise ValueError(
            f"! ERROR: {stl} failed a watertight mesh check and, therefore, is not valid for simulation"
        )

    lowerMeshBound, upperMeshBound = mesh.bounds
    shape = numpy.ceil(
        (upperMeshBound - lowerMeshBound) / pitch - 1e-9
    ).astype(int)
    origin = lowerMeshBound

    xIndex, yIndex = numpy.meshgrid(
        numpy.arange(shape[0]),
        numpy.arange(shape[1]),
        indexing = "ij"
    )
    xy = origin[:2] + (numpy.column_stack([xIndex.ravel(), yIndex.ravel()]) + 0.5) * pitch
    oRay = numpy.column_stack(
        [xy, numpy.full(len(xy), lowerMeshBound[2] - pitch)]
    )
    dRay = numpy.tile(
        [0.0, 0.0, 1.0],
        (len(xy), 1)
    )

    hits, rayIndex, _ = mesh.ray.intersects_location(
        oRay,
        dRay,
        multiple_hits = True
    )

    order = numpy.argsort(
        rayIndex,
        kind = "stable"
    )
    zHits = hits[order, 2]
    starts = numpy.searchsorted(
        rayIndex[order],
        numpy.arange((len(xy) + 1))
    )

    zCentres = origin[2] + (numpy.arange(shape[2]) + 0.5) * pitch
    matrix = numpy.zeros(
        shape = shape,
        dtype = bool
    )
    for k, (i, j) in enumerate(zip(xIndex.ravel(), yIndex.ravel())):
        z = numpy.unique(numpy.round(zHits[starts[k]:starts[k+1]], 9))
        if z.size % 2:
            raise RuntimeError(
                f"! ERROR: ray in column {(i, j)} grazed an edge or vertex; parity is undefined"
            )
        matrix[i, j] = numpy.searchsorted(z, zCentres) % 2 == 1
    return matrix, pitch, origin

def view(genotype, stl, pitch, context, optimise, controls):
    mesh = voxelise(
        stl   = stl,
        pitch = pitch
    )
    matrix, pitch, origin = mesh
    phenotype = optimise(
        mesh     = mesh,
        params   = genotype,
        context  = context,
        controls = controls
    )

    grid = pyvista.ImageData(
        dimensions = numpy.array(matrix.shape) + 1,
        spacing    = (pitch, pitch, pitch),
        origin     = origin
    )
    values = numpy.asarray(phenotype, dtype = float).copy()
    values[~matrix] = numpy.nan
    grid.cell_data["phenotype"] = values.ravel(order = "F")
    solid = grid.threshold(
        0.5,
        scalars = "phenotype"
    )
    genes = ", ".join(
        f"{name} = {genotype[i]:g}" for name, i in context.items()
    )
    solidFraction = solid.n_cells / int(numpy.count_nonzero(matrix))

    plotter = pyvista.Plotter(
        shape = (1, 2),
        title = genes
    )

    plotter.subplot(0, 0)
    plotter.add_mesh_clip_plane(
        solid,
        normal = "xyz"[int(numpy.argmin(matrix.shape))],
        invert = True,
        crinkle = True,
        color = "lightgrey",
        show_edges = False
    )

    plotter.subplot(0, 1)
    plotter.add_mesh_slice_orthogonal(
        grid,
        scalars = "phenotype",
        cmap    = "viridis",
        clim    = (0.0, 1.0),
        nan_opacity = 0.0
    )

    plotter.link_views()
    plotter.show()
