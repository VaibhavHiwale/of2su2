# of2su2 — OpenFOAM mesh → SU2 mesh

[![PyPI](https://img.shields.io/pypi/v/of2su2)](https://pypi.org/project/of2su2/) [![tests](https://github.com/VaibhavHiwale/of2su2/actions/workflows/tests.yml/badge.svg)](https://github.com/VaibhavHiwale/of2su2/actions/workflows/tests.yml) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/VaibhavHiwale/of2su2/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24127911.svg)](https://doi.org/10.5281/zenodo.24127911)

Converts an OpenFOAM mesh (`constant/polyMesh`, including snappyHexMesh meshes with polyhedra and hanging nodes) into a
valid, conformal SU2 mesh, and reports its quality exactly as SU2 computes it.

- **Single file**, Python 3 + numpy. No OpenFOAM or SU2 installation needed.
- **Only what SU2 cannot use is changed**: hex, prism, pyramid and tet cells are written as they are; polyhedra and
  hanging-node cells are split into standard elements, mostly without added points.
- **Checked output**: no inverted elements, every face matched, Σ volume = domain volume, SU2's own dual-CV quality.
- **Nothing dropped silently**: zones, interfaces, patch types and groups are carried over or listed.

## Installation

```bash
pip install of2su2                                          # from PyPI
pip install git+https://github.com/VaibhavHiwale/of2su2     # latest version from GitHub
of2su2 --help
```

Or download the single file `of2su2.py` and run `python3 of2su2.py` (needs only numpy).

## Usage

```bash
# convert (case folder or its constant/polyMesh folder)
of2su2  path/to/openfoamCase  mesh.su2

# convert and improve the worst added points against SU2's quality metrics (slower)
of2su2  path/to/openfoamCase  mesh.su2  --opt-apex 40 --opt-vr 1000

# score any .su2 file with SU2's quality metrics (optionally compare with SU2's MESH_QUALITY output)
of2su2  --check mesh.su2  [--vtk flow.vtk]
```

Output: `mesh.su2`, a full report `mesh.su2.json`, mesh information `mesh.su2.info.json`, and a summary on screen:

```text
zone rotating: 42,404 points, 61,106 elements, markers blade_wall, blade_root_cap, blade_tip_cap, AMI1
  valid: YES (non-positive elements 0, unmatched faces 0)
  SU2 orthogonality min 15.0° (points < 20°: 8, < 25°: 35), CV face AR max 173, sub-volume ratio max 357
  SU2 orientation check (as SU2 v8.5.0 does it): flips 0, distorted 0; boundary quads flips 0, distorted 2
zone rest: 13,092 points, 18,902 elements, markers farfield, AMI2
  ...
interface rotating | rest: markers AMI1 / AMI2 (coupled patches (cyclicAMI))
NOT carried over: pointZone 'frozenPoints' (0 points): the SU2 mesh format has no point sets
```

### Options

| Option | Meaning |
|---|---|
| `--opt-apex DEG` | move added centre points whose SU2 orthogonality is below DEG; never makes aspect ratio or volume ratio worse |
| `--opt-vr RATIO` | move added centre points of cells next to points with SU2 sub-volume ratio above RATIO |
| `--tri-interface` | triangulate zone-interface faces identically on both sides (for SU2 `WEIGHTED_AVERAGE`, see below) |
| `--single-zone` | one SU2 zone even if the mesh has separated cellZones |
| `--merge A B` | with `--single-zone`: glue this coupled patch pair (default: every cyclic/cyclicAMI pair whose points coincide is glued) |
| `--no-merge` | keep coupled patches as separate markers |
| `--no-patterns` | do not use the hanging-node transition patterns (centre-point split only) |
| `--nproc N` | worker processes (default: all CPUs; 1 = serial; the output does not depend on N) |
| `--check MESH.su2` | score an existing .su2 file instead of converting (`--vtk flow.vtk`: compare with SU2's output) |
| `--json` | print the full report |

## What it handles

| OpenFOAM | SU2 |
|---|---|
| hex, prism, pyramid, tet cells | same elements |
| hex with hanging nodes (2:1 refinement transitions) | mixed-element transition pattern (tets/pyramids/prisms/hexes, mostly no added point) when valid and not worse than the centre-point split |
| polygonal prisms (wall layers over split faces) | triangular prisms, no added point |
| other polyhedra, tet-wedges | pyramids + tets around one added centre point; polygon faces triangulated once → conformal |
| patches | markers (`MARKER_TAG=`), SU2 face winding |
| cellZones separated in OpenFOAM (`createBaffles` + `mergeOrSplitBaffles -split`) | one SU2 zone per cellZone + `rest` (`NZONE=` file); the coupled patches (e.g. cyclicAMI `AMI1`/`AMI2`) become the interface markers |
| cellZones with a conformal boundary (internal faceZone) | one SU2 zone; reported with the reason (of2su2 does not change the mesh topology) |
| patch types, groups, pointZones, sets | `<mesh>.su2.info.json` + "NOT carried over" list in the summary |
| ASCII or binary polyMesh, 32/64-bit labels | — |

Names are kept and made SU2-safe where needed; every rename is listed.

### Multizone meshes and `--tri-interface`

For a rotating zone (MRF) in SU2, separate the zone in OpenFOAM first:

```bash
createBaffles -overwrite                 # faceZone → patch pair (e.g. cyclicAMI AMI1/AMI2)
mergeOrSplitBaffles -split -overwrite    # duplicate the points → two separated cell regions
of2su2 . mesh.su2             # → 2 SU2 zones, interface markers AMI1/AMI2
```

In the SU2 config use `MARKER_ZONE_INTERFACE= ( AMI1, AMI2 )` and `KIND_INTERPOLATION`. The interface nodes coincide, so
`NEAREST_NEIGHBOR` is exact. SU2's conservative `WEIGHTED_AVERAGE` interpolation fails (NaN) on quadrilateral interface
elements; `--tri-interface` triangulates every interface face identically on both sides and splits only the cells
touching the interface (tets on the interface, pyramids elsewhere). It is the only option that changes cells SU2 could
use as they are.

## Known limitations

- 3D meshes only; periodic `cyclic` patches with a transform are not converted.
- Validated on the bundled test meshes and on one large snappyHexMesh rotor mesh (13.7 M cells, MRF zone, conformal
  and baffled); other meshers may show cell types not covered yet — the validity check reports any failure.
- Large meshes need memory (≈ 31 GB for 13.7 M cells) and time (≈ 7 min; ≈ 40 min with `--tri-interface`).
- At zone interfaces each SU2 zone sees only half of a node's control volume, so SU2's orthogonality there is lower
  than for the same mesh as one zone. This comes from SU2's dual mesh, not from the conversion.

## Method and validation

See [docs/METHOD.md](https://github.com/VaibhavHiwale/of2su2/blob/main/docs/METHOD.md): SU2 mesh requirements (from the SU2 source), the conversion steps, the quality
metrics, performance, validation results and credits.

## Tests

```bash
pip install pytest && python3 -m pytest -q tests/
```

| Path | Content |
|---|---|
| `tests/test_of2su2.py` | regression tests (pattern table, validity, quality, zones, interfaces) |
| `testcases/rotor_cut/` | 2,441 cells covering every unsupported cell shape of a 13.7 M-cell snappy rotor mesh (polyMesh included) |
| `testcases/blade_box/` | small snappyHexMesh blade case with refinement transitions, wall layers and an AMI pair (`Allrun` builds it; needs OpenFOAM) |
| `su2/su2_mesh_quality.cfg` | SU2 config that writes SU2's own mesh-quality output, to compare with `--check --vtk` |

## Licence

**GPL v3 or later** ([LICENSE](https://github.com/VaibhavHiwale/of2su2/blob/main/LICENSE)). of2su2 contains the hanging-node transition-pattern table of MixedOcTree
(github.com/clobosmesher/MixedOcTree, (C) 2013-2017 Claudio Lobos et al., GPL v3), based on E. González & C. Lobos,
"A set of mixed-element transition patterns for adaptive 3D meshing" (2014/15). Other prior art is credited in
[docs/METHOD.md](https://github.com/VaibhavHiwale/of2su2/blob/main/docs/METHOD.md).

## Citation

If you use of2su2 in published work, please cite it: DOI [10.5281/zenodo.24127911](https://doi.org/10.5281/zenodo.24127911) (all versions; each release has its own DOI on Zenodo). GitHub's "Cite this repository" gives the formatted reference.
