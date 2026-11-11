# of2su2 — method, validation and credits

## 1. Why a conversion (not just a format change) is needed

SU2 is a vertex-centred, median-dual finite-volume code; OpenFOAM is cell-centred and accepts any polyhedron. A
snappyHexMesh mesh therefore has to be **transformed**, not only rewritten. The requirements below are taken from the
SU2 source (v8.5.0 / v8.6.0).

| # | SU2 requirement | SU2 source | Consequence |
|---|---|---|---|
| R1 | Volume elements: tet, hex, prism, pyramid only | `CPhysicalGeometry::LoadVolumeElements`, `CCGNSElementType.cpp` | polyhedra (split-hex refinement cells, wall-layer cells over split faces) must be decomposed |
| R2 | No polyhedra in any input format (CGNS NGON/NFACE are rejected) | `CCGNSMeshReaderFVM.cpp`, `CCGNSElementType.cpp` | no CGNS shortcut |
| R3 | Node-conformal mesh; there is no hanging-node check | dual built per element (`SetControlVolume`) | a hanging node gives an open control volume without any error message → conformity must be guaranteed by the converter |
| R4 | Inverted elements are silently re-oriented, not rejected | `Check_IntElem_Orientation`, `Change_Orientation` | SU2 accepting a file proves nothing → the converter checks validity itself |
| R5 | No repeated nodes in one element | — | OpenFOAM tet-wedges are rewritten as standard elements |
| R6 | Hard stop on a zero-area dual face | `ComputeMeshQualityStatistics` | slivers must be removed |
| R7 | Quality is vertex-based (dual CV): orthogonality = area-weighted mean over a point's edges, "< 20° poor"; CV face aspect ratio; sub-volume ratio | `CPhysicalGeometry.cpp` (`ComputeMeshQualityStatistics`) | of2su2 computes exactly these metrics |
| R8 | ASCII format: `NDIME`, `NELEM`, `NPOIN`, `NMARK`, `MARKER_TAG=`, `MARKER_ELEMS=`; multizone: `NZONE=`, `IZONE=` | `CSU2ASCIIMeshReaderBase.cpp` | — |
| R9 | `ROTATING_FRAME` acts on a whole zone; partial rotation needs multizone + `MARKER_ZONE_INTERFACE` + `KIND_INTERPOLATION` | `CConfig.cpp`, `interface_interpolation/` | an OpenFOAM MRF cellZone becomes its own SU2 zone, if OpenFOAM separated it (§2.7) |

## 2. Conversion pipeline

1. **Read** `constant/polyMesh` directly (ASCII or binary, 32/64-bit labels), plus cellZones/faceZones/pointZones,
   patch types, `inGroups`, `neighbourPatch` and transforms. No OpenFOAM installation is needed.
2. **Classify** each cell. Hex, prism, pyramid and tet cells are written unchanged.
3. **Hexahedra with hanging nodes** (2:1 refinement transitions): the cell's faces are grouped into 6 "big" faces (by
   normal; by topology for warped thin cells), points are mapped onto MixedOcTree's 27-point cube numbering, and the
   matching transition pattern (325 patterns, 24 cube rotations; 260 of them add no point) is applied. A pattern is
   accepted only if every element is strictly valid, shared quads stay quads, triangulations of shared polygon faces
   agree with the neighbour, SU2 would not re-orient any element, and its smallest SU2 sub-volume is not smaller than
   that of the centre-point split. Otherwise the cell falls back to step 5.
4. **Polygonal prisms** (wall-layer cells over split faces, n-gon caps): split into n−2 triangular prisms **without an
   added point**; caps shared along a layer column get the same triangulation, so the mesh stays conformal.
5. **All other cells** (general polyhedra, tet-wedges): pyramids (on quad faces) and tets (on triangles) around one added
   centre point. The centre is the cell centroid; where the centroid does not see every face, a point in the cell's
   kernel is used for that cell only. Polygon faces are triangulated **once per face** (max-min quality dynamic
   programming), so both neighbours use the same triangles. A repair pass re-triangulates a free shared polygon face
   where a thin warped cell has no valid interior point, choosing the variant that keeps both neighbours valid.
6. **SU2 orientation view**: every element gets the orientation-preserving node order with the fewest failing SU2
   orientation tests, and boundary quads that SU2 would wrongly flip are fixed by relabelling the adjacent element.
   Geometry is unchanged.
7. **Zones and interfaces**: cellZones become SU2 zones only if OpenFOAM already separated them
   (`createBaffles` + `mergeOrSplitBaffles -split`, giving a coupled patch pair such as cyclicAMI `AMI1`/`AMI2`). The
   coupled patches become the interface markers. A conformal cellZone boundary (internal faceZone) is **not** split:
   the mesh stays one SU2 zone and the reason is reported. Names are kept and made SU2-safe if needed (the SU2 config
   tokenizer splits on spaces, brackets, `:` and `,`; max 64 characters). Every rename is listed.
8. **Optional point optimisation** (`--opt-apex`, `--opt-vr`): only the added centre points move. Each is improved on
   its local patch by a derivative-free search against SU2's own metric (orthogonality, or sub-volume ratio), accepting
   a move only if every sub-tet stays positive, SU2 would not flip any element, and the other two SU2 metrics do not get
   worse.
9. **`--tri-interface`** (opt-in): every quad/polygon of the zone interface is triangulated from a canonical,
   coordinate-based vertex order, so the two duplicated sides get identical triangles; each cell touching the interface
   is split around a centre point (tets on the interface, pyramids on its other faces). This is needed for SU2's
   conservative interface interpolation `KIND_INTERPOLATION= WEIGHTED_AVERAGE`, which fails on quadrilateral interface
   elements. Without the option the interface is written as in OpenFOAM.
10. **Write and check**: SU2 ASCII (single or multizone), then per zone: no element with volume ≤ 0, every element face
    matched, Σ volume = domain volume, and SU2's dual-CV quality.

## 3. Quality metrics

`su2_quality()` reproduces SU2's `ComputeMeshQualityStatistics` point by point (orthogonality, CV face aspect ratio,
sub-volume ratio); checked against SU2 v8.5.0 `MESH_QUALITY` volume output, max relative difference 5e-7 (ASCII
precision). `--check mesh.su2` scores any SU2 mesh the same way; `--vtk flow.vtk` compares against SU2's own output
(`su2/su2_mesh_quality.cfg` runs SU2 for that purpose).

Note for SU2 v8.5.0 users: the console line "Orthogonality Angle min" can be wrong. Exactly aligned edges give `nan` in
`acos` (no clamp in v8.5.0), and `std::min` with `nan` resets the running minimum. Use the per-point output or
`of2su2.py --check` instead.

Mesh-quality results on SU2's dual mesh at a zone interface: SU2 duplicates the interface points of each zone, so each
copy has only half a control volume and its orthogonality is lower than for the same mesh as one zone.

## 4. Performance

Per-face and per-cell steps run in forked worker processes (`--nproc`, default all CPUs); global decisions (layer
columns, shared-face conflicts) stay serial. Output is byte-identical for any number of processes.

| mesh | cells | time | peak memory |
|---|---|---|---|
| rotor test cut (`testcases/rotor_cut`) | 2,441 | 1.6 s | — |
| blade test mesh (`testcases/blade_box`) | 43,062 | 2.2 s | — |
| snappyHexMesh rotor, 1 zone | 13.74 M | 6.9 min | 30.7 GB |
| same, `--opt-vr 1000` | 13.74 M | 16.9 min | 32.2 GB |
| same, 2 zones, `--opt-vr 1000 --tri-interface` | 13.74 M | ~40 min | — |

(64 threads, AMD Threadripper PRO.)

## 5. Validation

Full rotor mesh (13,740,744 cells, snappyHexMesh, MRF zone; 540,535 cells not supported by SU2):

- 465,355 cells by MixedOcTree patterns, 64,130 polygonal prisms → triangular prisms, the rest centre-point split;
  15,202 added points.
- 0 non-positive elements, 0 unmatched faces, Σ volume = domain volume (11 digits).
- SU2 orthogonality min 15.6°, 4 points < 20° (all added centre points); median 89.9°; CV face AR max 234; sub-volume
  ratio max 75,944 (3,417 with `--opt-vr 1000`). For comparison, SU2's own Caradonna–Tung test mesh: AR max 434,
  sub-volume ratio max 7,997.
- SU2 v8.5.0 reads the result with no re-orientation of any volume or surface element, and its quality output equals
  of2su2's numbers. The remaining extremes sit where snappy's wall layers terminate (blade root, tip, trailing edge), in
  cells that are already badly shaped in the original mesh.
- Two-zone version (baffled MRF zone): both zones valid, interface nodes coincide exactly; SU2 runs steady rotating-frame
  RANS on it with `NEAREST_NEIGHBOR` and, with `--tri-interface`, with `WEIGHTED_AVERAGE`.

Regression tests (`tests/test_of2su2.py`, pytest): all 325 patterns valid on the unit cube; a single hex stays one hex;
the rotor cut converts valid, conformal and not worse than the recorded quality; `--check` reads back identical
quality; conformal cellZones stay one zone; baffled cellZones become SU2 zones; `--tri-interface` gives identical
triangles on both sides of an AMI pair.

## 6. Credits and prior art

| of2su2 part | source | use |
|---|---|---|
| hanging-node transition patterns | MixedOcTree (github.com/clobosmesher/MixedOcTree, `src/PatternFactory.cpp`, (C) 2013-2017 C. Lobos et al., **GPL v3**); E. González & C. Lobos, "A set of mixed-element transition patterns for adaptive 3D meshing" (2014/15) | pattern table embedded (hence of2su2 is GPL v3); rotation lookup as `TransitionTemplate.cpp` |
| centre point + pyramids/tets, consistent face splits | OpenFOAM `tetDecomposer` / `foamVtuSizing`; SALOME SMESH `SMESH_MeshEditor.cxx` (`SplitVolumes`, `getTetraSplitMethod`, LGPL) | method / idea, no code copied |
| polygonal prism → triangular prisms | SMESH extrusion split (hexa to prisms) | idea; layer-column consistency own |
| consistent interface triangulation | J. Dompierre, P. Labbé, M.-G. Vallet, R. Camarero, "How to subdivide pyramids, prisms and hexahedra into tetrahedra", IMR 1999 | consistent-order idea, applied to coordinates |
| point optimisation | Mesquite `NonSmoothDescent` / `NonGradient` (Freitag & Plassmann min-max smoothing; LGPL); CGAL Mesh_3 perturbation (step size) | structure (local patch, validity guard, accept-only-improving); objective = SU2's metric |
| SU2 metrics, orientation tests, mesh format | SU2 source (`CPhysicalGeometry.cpp`, `CSU2ASCIIMeshReaderBase.cpp`, `CConfig.cpp`; LGPL 2.1) | reproduced exactly |
| reading polyMesh directly | ConvertOpenFoam2SU2 (2013 script) | approach |
