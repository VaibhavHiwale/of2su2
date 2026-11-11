"""Regression tests for of2su2 (run: python3 -m pytest -q tests/).  Needs only numpy + pytest; uses the tracked rotor test cut."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import of2su2 as P  # noqa: E402

ROTOR = ROOT / "testcases" / "rotor_cut"
E2I = P._MOT_E2I
UNIT = np.array([[i % 3, i // 9, (i // 3) % 3] for i in E2I], float) / 2.0     # MixedOcTree 27 points on a unit cube


def test_pattern_table_complete_and_valid():
    """All 325 MixedOcTree patterns: every element strictly valid (VTK orientation), Σ volume = 1, conformal inside."""
    assert len(P.MOT_PATTERNS) == 325
    assert len(P._MOT_ROT) == 24
    for mask, els in P.MOT_PATTERNS.items():
        T = np.array([P._MOT_VTK[len(e)] for e in els])
        N = np.full((len(els), 8), -1, np.int64)
        for i, e in enumerate(els):
            N[i, :len(e)] = e
        assert P.element_min_subvol(UNIT, N, T).min() > 0, mask
        assert abs(P.element_volumes(UNIT, N, T).sum() - 1.0) < 1e-12, mask
        F, _ = P.element_faces(N, T)
        keys = [tuple(sorted(f[f >= 0])) for f in F]
        _, cnt = np.unique(np.array([k + (-1,) * (4 - len(k)) for k in keys]), axis=0, return_counts=True)
        assert cnt.max() <= 2, mask


def test_single_hex_unchanged(tmp_path):
    """A plain hex cell must come out as one native hex with the same 8 points."""
    pm = tmp_path / "constant" / "polyMesh"
    pm.mkdir(parents=True)
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    hdr = lambda cls, obj, note="": ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " + cls + ";\n"
                                     + (f'    note "{note}";\n' if note else "") + f"    object {obj};\n}}\n")
    (pm / "points").write_text(hdr("vectorField", "points") + "8\n(\n" + "".join(f"({x} {y} {z})\n" for x, y, z in pts) + ")\n")
    (pm / "faces").write_text(hdr("faceList", "faces") + "6\n(\n" + "".join(f"4({' '.join(map(str, f))})\n" for f in faces) + ")\n")
    (pm / "owner").write_text(hdr("labelList", "owner", "nPoints:8 nCells:1 nFaces:6 nInternalFaces:0") + "6\n(\n" + "0\n" * 6 + ")\n")
    (pm / "neighbour").write_text(hdr("labelList", "neighbour") + "0\n(\n)\n")
    (pm / "boundary").write_text(hdr("polyBoundaryMesh", "boundary") +
                                 "1\n(\n    walls\n    {\n        type wall;\n        nFaces 6;\n        startFace 0;\n    }\n)\n")
    rep = P.convert(tmp_path, tmp_path / "hex.su2", log=lambda *a: None)
    part = rep["parts"]["all"]
    assert rep["native"]["hex"] == 1 and rep["decomposed_cells"] == 0
    assert part["elements"] == 1 and part["points"] == 8 and part["n_nonpositive"] == 0
    assert abs(part["sum_volume"] - 1.0) < 1e-12
    assert part["su2_quality"]["orth_min_deg"] > 89.999


@pytest.fixture(scope="module")
def rotor(tmp_path_factory):
    out = tmp_path_factory.mktemp("rotor") / "rotor_cut.su2"
    return P.convert(ROTOR, out, log=lambda *a: None), out


def test_rotor_cut_valid_and_conformal(rotor):
    rep, _ = rotor
    part = rep["parts"]["all"]
    assert part["n_nonpositive"] == 0
    assert part["faces_used_more_than_twice"] == 0 and part["faces_once_not_on_marker_or_reverse"] == 0
    assert abs(part["sum_volume"] - part["boundary_volume"]) < 1e-9 * abs(part["boundary_volume"])
    assert rep["mixedoctree_pattern_cells"] > 500 and rep["polygonal_prisms_split"] > 300


def test_rotor_cut_quality_not_worse_than_recorded(rotor):
    """Recorded reference quality: orth min 8.8°, ≤ 7 points < 20°, ≤ 35 < 25°, AR ≤ 133, VR ≤ 1490."""
    q = rotor[0]["parts"]["all"]["su2_quality"]
    assert q["orth_min_deg"] >= 8.8 - 0.05
    assert q["points_orth_lt_20"] <= 7 and q["points_orth_lt_25"] <= 35
    assert q["ar_max"] <= 133.5 and q["vr_max"] <= 1490.5


def test_check_reads_back_same_quality(rotor):
    rep, out = rotor
    chk = P.check_su2(out)
    a, b = rep["parts"]["all"]["su2_quality"], chk["su2_quality"]
    assert chk["n_nonpositive"] == 0
    for k in ("orth_min_deg", "ar_max", "vr_max"):
        assert abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(a[k])), k


def _two_hex_case(tmp_path):
    """2 hexes along x; cellZone 'spin' = cell 0; internal faceZone 'iface' = the shared face; pointZone 'pz'; a set."""
    pm = tmp_path / "constant" / "polyMesh"; (pm / "sets").mkdir(parents=True)
    pid = lambda i, j, k: i * 4 + j * 2 + k
    pts = [(i, j, k) for i in range(3) for j in range(2) for k in range(2)]
    F = [[pid(1, 0, 0), pid(1, 1, 0), pid(1, 1, 1), pid(1, 0, 1)]]                       # internal, +x (owner 0)
    for i in (0, 1):
        side = [[pid(i, 0, 0), pid(i, 0, 1), pid(i, 1, 1), pid(i, 1, 0)]] if i == 0 else \
               [[pid(2, 0, 0), pid(2, 1, 0), pid(2, 1, 1), pid(2, 0, 1)]]
        F += side + [[pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)],
                     [pid(i, 1, 0), pid(i, 1, 1), pid(i + 1, 1, 1), pid(i + 1, 1, 0)],
                     [pid(i, 0, 0), pid(i, 1, 0), pid(i + 1, 1, 0), pid(i + 1, 0, 0)],
                     [pid(i, 0, 1), pid(i + 1, 0, 1), pid(i + 1, 1, 1), pid(i, 1, 1)]]
    own = [0] + [0] * 5 + [1] * 5
    hdr = lambda cls, obj, note="": ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " + cls + ";\n"
                                     + (f'    note "{note}";\n' if note else "") + f"    object {obj};\n}}\n")
    (pm / "points").write_text(hdr("vectorField", "points") + f"{len(pts)}\n(\n" + "".join(f"({x} {y} {z})\n" for x, y, z in pts) + ")\n")
    (pm / "faces").write_text(hdr("faceList", "faces") + f"{len(F)}\n(\n" + "".join(f"4({' '.join(map(str, f))})\n" for f in F) + ")\n")
    (pm / "owner").write_text(hdr("labelList", "owner", "nPoints:12 nCells:2 nFaces:11 nInternalFaces:1") + "11\n(\n" + "".join(f"{o}\n" for o in own) + ")\n")
    (pm / "neighbour").write_text(hdr("labelList", "neighbour") + "1\n(\n1\n)\n")
    (pm / "boundary").write_text(hdr("polyBoundaryMesh", "boundary") + "2\n(\n"
                                 "    walls\n    {\n        type wall;\n        inGroups 1(wall);\n        nFaces 5;\n        startFace 1;\n    }\n"
                                 "    outer\n    {\n        type patch;\n        nFaces 5;\n        startFace 6;\n    }\n)\n")
    (pm / "cellZones").write_text(hdr("regIOobject", "cellZones") + "1\n(\nspin\n{\n    type cellZone;\n    cellLabels List<label> 1(0);\n}\n)\n")
    (pm / "faceZones").write_text(hdr("regIOobject", "faceZones") + "1\n(\niface\n{\n    type faceZone;\n    faceLabels List<label> 1(0);\n"
                                  "    flipMap List<bool> 1(0);\n}\n)\n")
    (pm / "pointZones").write_text(hdr("regIOobject", "pointZones") + "1\n(\npz\n{\n    type pointZone;\n    pointLabels List<label> 1(0);\n}\n)\n")
    (pm / "sets" / "someSet").write_text("dummy\n")
    return tmp_path


def test_conformal_zone_stays_one_zone(tmp_path):
    """Conformal cellZone boundary (internal faceZone, no baffles): of2su2 does NOT split the mesh
    → ONE SU2 zone; the cellZone is reported with the reason; every other polyMesh item written or listed."""
    case = _two_hex_case(tmp_path)
    out = tmp_path / "two.su2"
    rep = P.convert(case, out, log=lambda *a: None)
    txt = out.read_text()
    assert not txt.startswith("NZONE") and rep["parts"]["all"]["n_nonpositive"] == 0
    info = rep["info"]
    nc = " ".join(info["not_carried_over"])
    assert "cellZone 'spin'" in nc and "conformal in OpenFOAM" in nc and "createBaffles" in nc
    assert "pointZone 'pz'" in nc and "sets/someSet" in nc
    assert {m["openfoam_patch"]: m["type"] for m in info["markers"]} == {"walls": "wall", "outer": "patch"}
    assert info["markers"][0]["inGroups"] == ["wall"]
    assert (tmp_path / "two.su2.info.json").is_file()


def test_baffled_zones_become_su2_zones(tmp_path):
    """Baffled in OpenFOAM (cyclicAMI pair, duplicated points) → one SU2 zone per cellZone + rest, AMI patches as the
    interface markers, nothing added by of2su2."""
    case = _ami_two_hex_case(tmp_path)
    out = tmp_path / "ami.su2"
    rep = P.convert(case, out, log=lambda *a: None)
    txt = out.read_text()
    assert txt.startswith("NZONE= 2") and txt.count("IZONE=") == 2
    assert set(rep["parts_order"]) == {"spin", "rest"}
    it = rep["info"]["interfaces"]
    assert len(it) == 1 and it[0]["kind"].startswith("coupled patches") and set(it[0]["markers"]) == {"AMI1", "AMI2"}
    for z in rep["parts"].values():
        assert z["n_nonpositive"] == 0 and z["faces_once_not_on_marker_or_reverse"] == 0


def test_single_zone_lists_dropped_zones(tmp_path):
    case = _two_hex_case(tmp_path)
    rep = P.convert(case, tmp_path / "one.su2", single_zone=True, log=lambda *a: None)
    nc = " ".join(rep["info"]["not_carried_over"])
    assert "cellZone 'spin'" in nc and "faceZone 'iface'" in nc
    assert not (tmp_path / "one.su2").read_text().startswith("NZONE")


def _hdr(cls, obj, note=""):
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " + cls + ";\n"
            + (f'    note "{note}";\n' if note else "") + f"    object {obj};\n}}\n")


def _ami_two_hex_case(tmp_path, stretch=1.0):
    """2 hexes along x, baffled at x = 1 (AMI1 on cell 0, AMI2 on cell 1, duplicated coincident points, as after
    createBaffles + mergeOrSplitBaffles -split); cellZone 'spin' = cell 0. stretch scales y (non-square interface)."""
    pm = tmp_path / "constant" / "polyMesh"; pm.mkdir(parents=True)
    pa = lambda i, j, k: i * 4 + j * 2 + k                      # cell 0 points, x in {0, 1}
    pb = lambda i, j, k: 8 + (i - 1) * 4 + j * 2 + k            # cell 1 points, x in {1, 2}
    pts = [(i, j * stretch, k) for i in (0, 1) for j in range(2) for k in range(2)] + \
          [(i, j * stretch, k) for i in (1, 2) for j in range(2) for k in range(2)]
    def sides(p, i):
        return [[p(i, 0, 0), p(i + 1, 0, 0), p(i + 1, 0, 1), p(i, 0, 1)], [p(i, 1, 0), p(i, 1, 1), p(i + 1, 1, 1), p(i + 1, 1, 0)],
                [p(i, 0, 0), p(i, 1, 0), p(i + 1, 1, 0), p(i + 1, 0, 0)], [p(i, 0, 1), p(i + 1, 0, 1), p(i + 1, 1, 1), p(i, 1, 1)]]
    walls = [[pa(0, 0, 0), pa(0, 0, 1), pa(0, 1, 1), pa(0, 1, 0)]] + sides(pa, 0) + \
            [[pb(2, 0, 0), pb(2, 1, 0), pb(2, 1, 1), pb(2, 0, 1)]] + sides(pb, 1)
    ami1 = [[pa(1, 0, 0), pa(1, 1, 0), pa(1, 1, 1), pa(1, 0, 1)]]
    ami2 = [[pb(1, 0, 0), pb(1, 0, 1), pb(1, 1, 1), pb(1, 1, 0)]]
    F = walls + ami1 + ami2
    own = [0] * 5 + [1] * 5 + [0, 1]
    (pm / "points").write_text(_hdr("vectorField", "points") + f"{len(pts)}\n(\n" + "".join(f"({x} {y} {z})\n" for x, y, z in pts) + ")\n")
    (pm / "faces").write_text(_hdr("faceList", "faces") + f"{len(F)}\n(\n" + "".join(f"4({' '.join(map(str, f))})\n" for f in F) + ")\n")
    (pm / "owner").write_text(_hdr("labelList", "owner", f"nPoints:16 nCells:2 nFaces:{len(F)} nInternalFaces:0") + f"{len(own)}\n(\n" + "".join(f"{o}\n" for o in own) + ")\n")
    (pm / "neighbour").write_text(_hdr("labelList", "neighbour") + "0\n(\n)\n")
    (pm / "boundary").write_text(_hdr("polyBoundaryMesh", "boundary") + "3\n(\n"
        "    walls\n    {\n        type wall;\n        nFaces 10;\n        startFace 0;\n    }\n"
        "    AMI1\n    {\n        type cyclicAMI;\n        nFaces 1;\n        startFace 10;\n        neighbourPatch AMI2;\n    }\n"
        "    AMI2\n    {\n        type cyclicAMI;\n        nFaces 1;\n        startFace 11;\n        neighbourPatch AMI1;\n    }\n)\n")
    (pm / "cellZones").write_text(_hdr("regIOobject", "cellZones") + "1\n(\nspin\n{\n    type cellZone;\n    cellLabels List<label> 1(0);\n}\n)\n")
    return tmp_path


def _marker_tris(path, zone_marker):
    """{marker: set of coordinate-triangles} from a multizone .su2 file."""
    txt = open(path).read().split("IZONE=")[1:]
    out = {}
    for z in txt:
        L = z.split("\n")
        i = next(k for k, l in enumerate(L) if l.startswith("NPOIN="))
        n = int(L[i].split("=")[1].split()[0]); X = [tuple(map(float, l.split()[:3])) for l in L[i + 1:i + 1 + n]]
        for k, l in enumerate(L):
            if l.startswith("MARKER_TAG=") and l.split("=")[1].strip() in zone_marker:
                m = int(L[k + 1].split("=")[1]); faces = [list(map(int, x.split()))[1:] for x in L[k + 2:k + 2 + m]]
                out[l.split("=")[1].strip()] = (len(faces), {frozenset(X[v] for v in f) for f in faces}, [len(f) for f in faces])
    return out


def test_tri_interface_ignores_conformal_facezone(tmp_path):
    """--tri-interface acts only on interfaces that come from OpenFOAM (coupled patches); a conformal faceZone mesh is
    left as one zone and untouched."""
    case = _two_hex_case(tmp_path)
    rep = P.convert(case, tmp_path / "tri.su2", tri_interface=True, log=lambda *a: None)
    assert rep["tri_interface_faces"] == 0 and rep["tri_interface_cells"] == 0
    assert rep["native"]["hex"] == 2 and "modified" not in rep["info"]


def test_tri_interface_ami_pair_same_diagonal(tmp_path):
    """cyclicAMI pair with duplicated points (baffled mesh): both sides get the SAME two triangles (by coordinates),
    for a square and a stretched interface."""
    for s in (1.0, 2.5):
        d = tmp_path / f"s{s}"
        case = _ami_two_hex_case(d, stretch=s)
        out = d / "ami.su2"
        rep = P.convert(case, out, tri_interface=True, log=lambda *a: None)
        assert rep["tri_interface_faces"] == 2 and rep["tri_interface_cells"] == 2
        for z in rep["parts"].values():
            assert z["n_nonpositive"] == 0 and z["faces_once_not_on_marker_or_reverse"] == 0
        m = _marker_tris(out, {"AMI1", "AMI2"})
        assert m["AMI1"][0] == m["AMI2"][0] == 2 and m["AMI1"][1] == m["AMI2"][1], s
        # without the option the AMI faces stay quads and the cells stay hexes
        rep0 = P.convert(case, d / "q.su2", log=lambda *a: None)
        m0 = _marker_tris(d / "q.su2", {"AMI1", "AMI2"})
        assert m0["AMI1"][2] == [4] and rep0["tri_interface_faces"] == 0
