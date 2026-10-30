import itertools

import ciclope
import numpy
import trimesh
import os
import subprocess
import meshio
import re
from Auxiliary import voxelise

class Simulator:
    def __init__(self, genotype, stl, pitch, context, optimise, controls, jobDir, offDesignCases, threads, writeVtu = False, keepRaw = False):
        self._genotype     = genotype
        self._fitnessTerms = {}
        self._controls = controls
        self._jobDir = jobDir
        self._offDesignCases = offDesignCases
        self._threads = threads
        self._writeVtu = writeVtu
        self._keepRaw = keepRaw

        # rather than voxelising per simulation, voxelise in GC and pass mesh to each simulator object?
        # - may reduce computation time
        self._phenotype = optimise(
            mesh      = voxelise(stl, pitch),
            params  = genotype,
            context   = context,
            control   = controls
        )

    def _readData(self, path):
        kinds = {
            "displacements"         : "U",
            "total force"           : "RF",
            "total internal energy" : "ENER"
        }
        blocks = {
            "U"                     : [],
            "RF"                    : [],
            "total internal energy" : []
        }
        kind = None
        with open(path) as f:
            for line in f:
                header = re.match(
                    r"^\s*(displacements|total force|total internal energy) ",
                    line
                )
                if header:
                    kind = kinds[header.group(1)]
                    blocks[kind].append([])
                    continue
                tokens = line.split()
                if kind is None or not tokens:
                    continue
                try:
                    blocks[kind][-1].append([float(token) for token in tokens])
                except ValueError:
                    kind = None
        return blocks

    def _readFrd(self, path):
        blocks = {
            "disp"   : [],
            "stress" : []
        }
        kind = None
        with open(path) as f:
            for line in f:
                if line.startswith("-4"):
                    kind = line[5:13].strip()
                    if kind in blocks:
                        blocks[kind].append(
                            ([], [])
                        )
                    else:
                        kind = None
                elif kind is None:
                    continue
                elif line.startswith("-1"):
                    body = line[13:].strip("\n")
                    blocks[kind][-1][0].append(int(line[3:13]))
                    blocks[kind][-1][1].append(
                        [float(body[k:k + 12]) for k in range(0, len(body) - 11, 12)]
                    )
                elif line.startswith(" -3"):
                    kind = None
        return blocks

    def run(self, threshold = 0.5, eltype = "C3D8I", ccxSolver = "ITERATIVE CHOLESKY", ccx = "ccx", jobName = "job"):
        ...

    def _initialise(self, threshold = 0.5, eltype = "C3D8I", ccxSolver = "ITERATIVE CHOLESKY", ccx = "ccx", jobName = "job") -> dict:
        matrix, pitch, origin = self._phenotype
        solid  = numpy.asarray(self._phenotype) >= threshold
        nSolid = int(solid.sum())
        vol    = ciclope.utils.preprocess.remove_unconnected(
            numpy.transpose(
                solid,
                (2, 1, 0)
            )
        )
        nElements = int(vol.sum())
        vol = numpy.pad(
            vol.astype(numpy.uint8),
            1
        )

        uGrid = ciclope.core.voxelFE.vol2ugrid(
            vol,
            voxelsize = [float(pitch)] * 3
        )
        cells = numpy.asarray(uGrid.cells_dict["hexahedron"])
        used  = numpy.unique(cells)
        world = numpy.asarray(
            uGrid.points,
            dtype = float
        )[used] - pitch + origin

        nodeSets = {
            "FIX"  : used[self._controls["fixture"](world)],
            "LOAD" : used[self._controls["loadSelector"](world)]
        }
        uGrid.point_sets = {name : ids.tolist() for name, ids in nodeSets.items()}
        uGrid.cell_sets  = {}

        loadForce = numpy.asarray(
            self._controls["loadForce"],
            dtype = float
        )
        mag = float(numpy.linalg.norm(loadForce))
        forces = {"DESIGN" : loadForce}
        for direction in itertools.product((-1, 0, 1), repeat = 3):
            for c, axis in zip(direction, "XYZ"):
                if c != 0:
                    if c > 0:
                        name = "".join("+") + axis
                    elif c < 0:
                        name = "".join("-") + axis

            forces[name] = mag * numpy.asarray(direction, dtype = float) / numpy.linalg.norm(direction)

        static   = "*STATIC" + (", SOLVER" + ccxSolver)
        template = [
            "*MATERIAL, NAME=PART",
            "*ELASTIC",
            f"{self._controls['E0']:.8g}, {self._controls['nu']:.8g}",
            "*SOLID SECTION, ELSET=SET1, MATERIAL=PART"
        ]

        for dof in (1, 2, 3):
            template += [
                "*STEP", static,
                "*BOUNDARY", "FIX 1, 3, 0.0",
                "*CLOAD, OP=NEW", f"LOAD, {dof}, {1.0 / nodeSets['LOAD'].size:.12e}",
                "*NODE PRINT, NSET = LOAD", "U",
                "*NODE PRINT, NSET = FIX, TOTALS=ONLY", "RF",
                "*EL PRINT, ELSET=SET1, TOTALS=ONLY", "ELSE",
                "*NODE FILE", "U",
                "*EL FILE", "S",
                "*END STEP"
            ]
        os.makedirs(
            self._jobDir,
            exist_ok = True
        )
        templatePath = os.path.join(self._jobDir, "template.inp")
        with open(templatePath, "w") as f:
            f.write("\n".join(template) + "\n")
        ciclope.core.voxelFE.mesh2voxelfe(
            uGrid,
            templatePath,
            os.path.join(self._jobDir, jobName + ".inp"),
            keywords = ["NSET"],
            eltype   = eltype
        )

        return {
            "jobName"   : jobName,
            "phenotype" : self._phenotype,
            "used"      : used,
            "world"     : world,
            "cells"     : cells,
            "forces"    : forces,
            "mag"       : mag,
            "nElements" : nElements,
            "islandFraction" : 1.0 - nElements / nSolid,
            "vol"       : nElements * float(pitch) ** 3,
            "solidFraction"  : nElements / int(numpy.count_nonzero(matrix)),
            "nodeSetSizes" : {name : int(ids.size) for name, ids in nodeSets.items()}
        }

    def run(self, model, ccx = "ccx"):
        jobName = model["jobName"]






