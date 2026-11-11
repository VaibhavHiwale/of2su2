#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# of2su2 — Copyright (C) 2026 Vaibhav Hiwale. GPL v3 or later (see LICENSE): contains the transition-pattern table of
# MixedOcTree (C) 2013-2017 Claudio Lobos et al., GPL v3 — see _MOT_TABLE.
"""of2su2 — convert an OpenFOAM mesh (constant/polyMesh) into an SU2 mesh (.su2). One file, Python 3 + numpy.
No OpenFOAM installation needed.

  CONVERT   python3 of2su2.py <case dir | polyMesh dir> <out.su2>
                [--opt-apex 40]          improve the worst cells against SU2's quality metrics (slower)
                [--opt-vr RATIO] [--tri-interface] [--nproc N]
                [--single-zone] [--merge A B | --no-merge]
  CHECK     python3 of2su2.py --check <mesh.su2> [--vtk flow.vtk]

What it does (snappyHexMesh meshes included):
  * hex / prism / pyramid / tet cells → native SU2 elements;
  * every other cell (polyhedra, hanging-node cells, tet-wedges) → pyramids + tets around one added centre point;
    polygon faces are triangulated once per face, so the SU2 mesh stays conformal (SU2 has no polyhedra and no hanging
    nodes);
  * OpenFOAM patches → SU2 markers (boundary faces written with SU2's winding);
  * the mesh comes back as it was — only unsupported cells change; of2su2 never changes the topology itself.
    cellZones become SU2 zones ONLY if OpenFOAM already separated them (baffled, e.g. createBaffles +
    mergeOrSplitBaffles -split → cyclicAMI patches AMI1/AMI2): one 'NZONE=' file, one zone per cellZone + 'rest',
    the coupled patches as the interface markers. A conformal cellZone boundary (internal faceZone) stays ONE
    SU2 zone (reported). Names are kept (made SU2-safe if needed). --single-zone always writes one zone.
  * <out.su2>.info.json + summary: patch types / groups, zones, interface marker pairs, renames, limits, and everything
    NOT carried over (pointZones, polyMesh/sets, ...) — nothing is dropped silently.

Every conversion checks itself and writes <out.su2>.json:
  * validity  — no element with volume <= 0, every element face matched (conformal), Σ volume = domain volume;
  * quality   — SU2's own dual-CV metrics (orthogonality, CV face aspect ratio, sub-volume ratio), computed exactly as
                SU2 does (verified point by point against SU2 v8.5.0, max rel. diff 5e-7). SU2 rates orthogonality
                < 20° as poor.
  * where the bad points are (added centre points vs original mesh points).
--check scores any .su2 file the same way (and with --vtk compares against SU2's MESH_QUALITY output).
"""
import argparse
import json
import re
import sys
import time
from itertools import combinations  # noqa: F401  (kept for small-polygon fallbacks)
from pathlib import Path

import numpy as np

__version__ = "1.0.1"

VTK_TET, VTK_HEX, VTK_PRISM, VTK_PYR = 10, 12, 13, 14
NODES = {VTK_TET: 4, VTK_HEX: 8, VTK_PRISM: 6, VTK_PYR: 5}
# outward faces of the VTK linear cells (positive-volume VTK ordering), -1 = triangle
FACES = {
    VTK_TET: [(0, 2, 1, -1), (0, 1, 3, -1), (1, 2, 3, -1), (2, 0, 3, -1)],
    VTK_HEX: [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    VTK_PRISM: [(0, 1, 2, -1), (3, 5, 4, -1), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)],
    VTK_PYR: [(0, 3, 2, 1), (0, 1, 4, -1), (1, 2, 4, -1), (2, 3, 4, -1), (3, 0, 4, -1)],
}


# =================================================================================================== polyMesh reader
def _header(data):
    """(binary?, label dtype, scalar dtype, offset after the FoamFile header)."""
    m = re.search(rb"FoamFile\s*\{(.*?)\}", data[:4000], re.S)
    h = m.group(1) if m else b""
    binary = bool(re.search(rb"format\s+binary", h))
    lab = re.search(rb"label\s*=\s*(\d+)", h)
    sca = re.search(rb"scalar\s*=\s*(\d+)", h)
    ldt = np.dtype("<i8") if lab and lab.group(1) == b"64" else np.dtype("<i4")
    sdt = np.dtype("<f4") if sca and sca.group(1) == b"32" else np.dtype("<f8")
    return binary, ldt, sdt, (m.end() if m else 0)


def _strip_comments(b):
    b = re.sub(rb"/\*.*?\*/", b" ", b, flags=re.S)
    return re.sub(rb"//[^\n]*", b" ", b)


def _list_at(data, pos, binary, dtype, width=1):
    """Read one OpenFOAM list starting at/after pos → (array, end pos). Handles N(…), N{v}, binary N(<bytes>)."""
    m = re.compile(rb"\s*(\d+)\s*([({])").match(data, pos) or re.compile(rb".*?(\d+)\s*([({])", re.S).match(data, pos)
    n, br = int(m.group(1)), m.group(2)
    if br == b"{":                                                   # uniform list N{v}
        end = data.index(b"}", m.end())
        v = np.array(data[m.end():end].replace(b"(", b" ").replace(b")", b" ").split(), dtype=dtype)
        return np.tile(v, n).reshape(n, width) if width > 1 else np.full(n, v[0], dtype), end + 1
    if binary and n:
        nbytes = n * width * dtype.itemsize
        arr = np.frombuffer(data, dtype, n * width, m.end()).copy()
        end = m.end() + nbytes
        end = data.index(b")", end) + 1
        return (arr.reshape(n, width) if width > 1 else arr), end
    depth, i = 1, m.end()
    while depth:                                                     # find the matching ')'
        c = data[i:i + 1]
        depth += (c == b"(") - (c == b")")
        i += 1
    body = data[m.end():i - 1].replace(b"(", b" ").replace(b")", b" ")
    arr = np.array(body.split(), dtype=dtype) if n else np.zeros(0, dtype)
    return (arr.reshape(n, width) if width > 1 else arr), i


def read_points(p):
    data = Path(p).read_bytes()
    binary, _, sdt, pos = _header(data)
    if not binary:
        data = _strip_comments(data)
        pos = _header(data)[3]
    return _list_at(data, pos, binary, sdt if binary else np.dtype(float), 3)[0].astype(float)


def read_labels(p):
    data = Path(p).read_bytes()
    binary, ldt, _, pos = _header(data)
    if not binary:
        data = _strip_comments(data)
        pos = _header(data)[3]
    return _list_at(data, pos, binary, ldt if binary else np.dtype(np.int64))[0].astype(np.int64)


def read_faces(p):
    """faces → (offsets n+1, flat labels). faceCompactList (ascii/binary) or old ascii faceList n(a b c)."""
    data = Path(p).read_bytes()
    binary, ldt, _, pos = _header(data)
    compact = b"faceCompactList" in data[:4000]
    if not binary:
        data = _strip_comments(data)
        pos = _header(data)[3]
    if compact:
        dt = ldt if binary else np.dtype(np.int64)
        off, pos = _list_at(data, pos, binary, dt)
        lab, _ = _list_at(data, pos, binary, dt)
        return off.astype(np.int64), lab.astype(np.int64)
    m = re.compile(rb".*?(\d+)\s*\(", re.S).match(data, pos)
    n = int(m.group(1))
    sizes, labs = [], []
    for fm in re.finditer(rb"(\d+)\s*\(([^)]*)\)", data[m.end():]):
        v = fm.group(2).split()
        sizes.append(int(fm.group(1)))
        labs.extend(v)
        if len(sizes) == n:
            break
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64), np.array(labs, np.int64)


def read_boundary(p):
    """[(name, type, startFace, nFaces)]."""
    text = _strip_comments(Path(p).read_bytes()).decode(errors="ignore")
    body = text[text.index("(", text.index("}", text.index("FoamFile"))):]
    out = []
    for m in re.finditer(r"(\S+)\s*\{([^}]*)\}", body):
        get = lambda k: re.search(rf"\b{k}\s+([^;]+);", m.group(2)).group(1).strip()
        out.append((m.group(1), get("type"), int(get("startFace")), int(get("nFaces"))))
    return out


def read_zones(p, kind):
    """OpenFOAM cellZones / faceZones / pointZones → {name: labels} (ascii or binary; any OpenFOAM word as name —
    the earlier `\\w+` pattern silently skipped names with '.' or '-'). kind: 'cell' | 'face' | 'point'."""
    p = Path(p)
    if not p.is_file():
        return {}
    data = p.read_bytes()
    binary, ldt, _, _ = _header(data)
    zones = {}
    pat = rb"\n\s*([^\s{}();]+)\s*\{\s*type\s+" + kind.encode() + rb"Zone;"
    for m in re.finditer(pat, data):
        lm = re.compile(kind.encode() + rb"Labels\s+List<label>\s*").search(data, m.end())
        if lm is None:
            zones[m.group(1).decode()] = np.zeros(0, np.int64); continue
        if re.match(rb"\s*0\s*;", data[lm.end():lm.end() + 16]):     # empty zone written as "List<label> 0;"
            zones[m.group(1).decode()] = np.zeros(0, np.int64); continue
        arr, _ = _list_at(data, lm.end(), binary, ldt if binary else np.dtype(np.int64))
        zones[m.group(1).decode()] = arr.astype(np.int64)
    return zones


def read_cell_zones(p):
    return read_zones(p, "cell")


def read_patch_info(p):
    """{patch: {type, inGroups, neighbourPatch, transform}} from polyMesh/boundary (what SU2's marker list cannot hold)."""
    text = _strip_comments(Path(p).read_bytes()).decode(errors="ignore")
    body = text[text.index("(", text.index("}", text.index("FoamFile"))):]
    info = {}
    for m in re.finditer(r"(\S+)\s*\{([^}]*)\}", body):
        d = {}
        for k in ("type", "neighbourPatch", "transform"):
            mm = re.search(rf"\b{k}\s+([^;]+);", m.group(2))
            if mm:
                d[k] = mm.group(1).strip()
        g = re.search(r"\binGroups\s+[0-9]*\s*\(([^)]*)\)", m.group(2))
        d["inGroups"] = g.group(1).split() if g else []
        info[m.group(1)] = d
    return info


def read_polymesh(path):
    path = Path(path)
    pm = path if (path / "faces").is_file() else path / "constant" / "polyMesh"
    off, lab = read_faces(pm / "faces")
    return {"dir": str(pm), "points": read_points(pm / "points"), "foff": off, "flab": lab,
            "owner": read_labels(pm / "owner"), "neighbour": read_labels(pm / "neighbour"),
            "patches": read_boundary(pm / "boundary"), "zones": read_zones(pm / "cellZones", "cell"),
            "face_zones": read_zones(pm / "faceZones", "face"), "point_zones": read_zones(pm / "pointZones", "point"),
            "patch_info": read_patch_info(pm / "boundary"),
            "sets": sorted(q.name for q in (pm / "sets").iterdir()) if (pm / "sets").is_dir() else []}


# ======================================================================================================= geometry
def tri_quality(a, b, c):
    """4√3 A / Σl² (1 = equilateral, 0 = degenerate)."""
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    s = ((b - a) ** 2).sum() + ((c - b) ** 2).sum() + ((a - c) ** 2).sum()
    return 4 * np.sqrt(3) * A / s if s > 0 else 0.0


def triangulate_polygon(X, f):
    """Max-min-quality triangulation of polygon f (point labels, OpenFOAM order kept) → list of (a, b, c)."""
    n = len(f)
    P = X[f]
    best = {}
    choice = {}
    for gap in range(2, n):                                          # DP over sub-chains i..j
        for i in range(n - gap):
            j = i + gap
            bq, bk = -1.0, None
            for k in range(i + 1, j):
                q = tri_quality(P[i], P[k], P[j])
                if k - i > 1:
                    q = min(q, best[(i, k)])
                if j - k > 1:
                    q = min(q, best[(k, j)])
                if q > bq:
                    bq, bk = q, k
            best[(i, j)], choice[(i, j)] = bq, bk
    out, stack = [], [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = choice[(i, j)]
        out.append((int(f[i]), int(f[k]), int(f[j])))
        stack += [(i, k), (k, j)]
    return out, best[(0, n - 1)]


def tet_vol(a, b, c, d):
    """Signed volume, positive when (a,b,c) seen from d is counter-clockwise (VTK tet)."""
    return np.einsum("...i,...i", np.cross(b - a, c - a), d - a) / 6.0


def element_volumes(X, conn, types):
    """Volume of every element (outward VTK faces, face-centroid fan → exact for planar faces)."""
    vol = np.zeros(len(types))
    for t, faces in FACES.items():
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        cn = np.stack([conn[i] for i in idx]) if isinstance(conn, list) else conn[idx, :NODES[t]]
        P = X[cn]
        cc = P.mean(1)
        v = np.zeros(len(idx))
        for f in faces:
            k = [i for i in f if i >= 0]
            fc = P[:, k].mean(1)
            for a, b in zip(k, k[1:] + k[:1]):
                v += tet_vol(P[:, a], P[:, b], fc, cc) * -1.0       # outward face, apex inside → minus sign
        vol[idx] = v
    return vol


def subtet_min(X, pieces, x):
    """Smallest sub-tet volume of a cell's pieces (outward (a,b,c[,d]); quads as face-centroid fan) with apex x."""
    v = np.inf
    for p in pieces:
        Q = X[list(p)]
        tris = [Q] if len(p) == 3 else [np.array([Q[k], Q[(k + 1) % 4], Q.mean(0)]) for k in range(4)]
        for T in tris:
            v = min(v, float(np.dot(np.cross(T[1] - T[0], T[2] - T[0]), T[0] - x)) / 6.0)
    return v


def apex_kernel(X, pieces, x0, iters=30):
    """Apex inside the kernel: maximise Σ log(volume of every sub-tet) (analytic centre); start x0.
    pieces: list of outward (a,b,c[,d]) node tuples. Returns (x, min sub-tet volume)."""
    T = []                                                           # quads: face-centroid fan (as the volume check)
    for p in pieces:
        if len(p) == 3:
            T.append(X[list(p)])
        else:
            Q = X[list(p)]
            fc = Q.mean(0)
            T += [np.array([Q[k], Q[(k + 1) % 4], fc]) for k in range(4)]
    T = np.array(T)                                                  # (m,3,3)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])               # outward normal ×2A
    c = np.einsum("ij,ij->i", n, T[:, 0])                            # volume_j(x) = (c_j − n_j·x)/6  (>0 inside)
    scale = np.abs(c - n @ x0).max() + 1e-300
    n, c = n / scale, c / scale
    x = x0.copy()
    s = c - n @ x
    if s.min() <= 0:                                                 # phase 1: push to max-min (subgradient)
        h = np.ptp(T.reshape(-1, 3), axis=0).max()
        best_x, best_s = x.copy(), s.min()
        for it in range(400):
            j = np.argmin(s)
            x = x + (0.5 * h / (it + 1)) * (-n[j] / (np.linalg.norm(n[j]) + 1e-300))
            s = c - n @ x
            if s.min() > best_s:
                best_x, best_s = x.copy(), s.min()
            if s.min() > 0:
                break
        if s.min() <= 0:                                             # no strict kernel: keep the max-min point
            return best_x, float(best_s * scale / 6)
    for _ in range(iters):                                           # phase 2: damped Newton on −Σ log s
        g = (n / s[:, None]).sum(0)
        H = (n[:, :, None] * n[:, None, :] / (s ** 2)[:, None, None]).sum(0)
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6 and (c - n @ (x + t * dx)).min() <= 0:
            t *= 0.5
        x = x + t * dx
        s = c - n @ x
        if np.linalg.norm(t * dx) < 1e-12 * (np.abs(x).max() + 1):
            break
    return x, float(s.min() * scale / 6)


# ================================================================================================== cell analysis
def cell_faces(mesh):
    """CSR cell → faces (face ids) and orientation flag (+1 owner: face points out; −1 neighbour)."""
    own, nei = mesh["owner"], mesh["neighbour"]
    nc = int(max(own.max(), nei.max() if len(nei) else -1)) + 1
    fid = np.concatenate([np.arange(len(own)), np.arange(len(nei))])
    cel = np.concatenate([own, nei])
    sgn = np.concatenate([np.ones(len(own), np.int8), -np.ones(len(nei), np.int8)])
    o = np.argsort(cel, kind="stable")
    cnt = np.bincount(cel, minlength=nc)
    return nc, np.concatenate([[0], np.cumsum(cnt)]), fid[o], sgn[o]


def classify(mesh, nc, coff, cfid, csgn):
    """Native VTK element (nodes) for every tet/pyr/prism/hex cell → (types, nodes(nc,8) −1 padded); other cells type 0."""
    foff, flab = mesh["foff"], mesh["flab"]
    fsize = np.diff(foff)
    nf = np.diff(coff)
    types = np.zeros(nc, np.int64)
    nodes = np.full((nc, 8), -1, np.int64)
    first = coff[:-1]
    # per-cell number of tri / quad / other faces
    cs = fsize[cfid]
    cell_of = np.repeat(np.arange(nc), nf)
    ntri = np.bincount(cell_of, cs == 3, nc).astype(int)
    nquad = np.bincount(cell_of, cs == 4, nc).astype(int)
    cand = {VTK_HEX: (nf == 6) & (nquad == 6), VTK_PRISM: (nf == 5) & (ntri == 2) & (nquad == 3),
            VTK_PYR: (nf == 5) & (ntri == 4) & (nquad == 1), VTK_TET: (nf == 4) & (ntri == 4)}
    for t, m in cand.items():
        cells = np.flatnonzero(m)
        if not len(cells):
            continue
        k = nf[cells[0]]
        F = cfid[first[cells][:, None] + np.arange(k)]               # (n,k) faces
        S = csgn[first[cells][:, None] + np.arange(k)]
        sz = fsize[F]
        maxs = sz.max()
        V = np.full((len(cells), k, maxs), -1, np.int64)             # face nodes, oriented OUTWARD of the cell
        for j in range(k):
            for q in range(maxs):
                ok = sz[:, j] > q
                V[ok, j, q] = flab[foff[F[ok, j]] + q]
            rev = S[:, j] < 0
            for r in np.flatnonzero(rev):
                s_ = sz[r, j]
                V[r, j, :s_] = V[r, j, :s_][::-1]
        # base face: hex → face 0 (quad), prism → first tri, pyr → the quad, tet → face 0
        if t == VTK_HEX or t == VTK_TET:
            bsel = np.zeros(len(cells), int)
        elif t == VTK_PRISM:
            bsel = np.argmax(sz == 3, axis=1)
        else:
            bsel = np.argmax(sz == 4, axis=1)
        nb = 4 if t in (VTK_HEX, VTK_PYR) else 3
        base_out = V[np.arange(len(cells)), bsel, :nb]               # outward
        base_in = base_out[:, ::-1]                                  # inward (normal → rest of the cell)
        allpts = V.reshape(len(cells), -1)
        if t in (VTK_TET, VTK_PYR):                                  # apex = the point not in the base
            isb = (allpts[:, :, None] == base_in[:, None, :]).any(2) | (allpts < 0)
            apex = np.where(~isb, allpts, -1).max(1)
            el = np.column_stack([base_in, apex])
            good = np.ones(len(cells), bool)
        else:                                                        # opposite node of each base node via edges
            a_ = V
            b_ = np.roll(V, -1, axis=2)
            if maxs == 4:                                            # triangles padded with −1: close the loop
                tri = sz == 3
                b_[tri, 2] = V[tri, 0]
            E0, E1 = a_.reshape(len(cells), -1), b_.reshape(len(cells), -1)
            inb1 = (E1[:, :, None] == base_in[:, None, :]).any(2) | (E1 < 0)
            opp = np.full((len(cells), nb), -1, np.int64)
            good = np.ones(len(cells), bool)
            for q in range(nb):
                m_ = (E0 == base_in[:, [q]]) & ~inb1
                cntq = m_.sum(1)
                good &= cntq >= 1
                opp[:, q] = np.where(m_, E1, -1).max(1)
                # both directions of the edge appear (two faces): the other direction
                m2 = (E1 == base_in[:, [q]]) & ~((E0[:, :, None] == base_in[:, None, :]).any(2) | (E0 < 0))
                alt = np.where(m2, E0, -1).max(1)
                good &= (opp[:, q] == alt) | (opp[:, q] < 0) | (alt < 0)
                opp[:, q] = np.where(opp[:, q] < 0, alt, opp[:, q])
            good &= (opp >= 0).all(1)
            if t == VTK_PRISM:                                       # VTK wedge: (0,1,2) outward, 3-5 opposite
                el = np.column_stack([base_out, opp[:, ::-1]])
            else:
                el = np.column_stack([base_in, opp])
        nu = np.array([len(set(r[r >= 0].tolist())) for r in allpts]) if len(cells) < 0 else None  # noqa
        uniq = np.sort(el, 1)
        good &= (np.diff(uniq, axis=1) != 0).all(1)
        cells, el = cells[good], el[good]
        types[cells] = t
        nodes[cells, :el.shape[1]] = el
    return types, nodes


# ============================================================== hanging-node hex cells: MixedOcTree transition patterns
# The 325 mixed-element transition patterns of E. González & C. Lobos ("A set of mixed-element transition patterns for
# adaptive 3D meshing", tech. report 2014/15, doi 10.13140/RG.2.1.3367.4400), as coded in MixedOcTree
# (github.com/clobosmesher/MixedOcTree, src/PatternFactory.cpp, commit a721c67; GPL v3, (C) 2013-2017 Claudio Lobos et al.).
# A hex with hanging nodes (edge mid-nodes 8-19, face centres 20-25 in MixedOcTree's TransitionCube numbering; 26 = cell
# centre, added by some patterns) is split into tets / pyramids / prisms / hexes; 260 of 325 patterns add no point.
# Table: one line per pattern "mask:element,element,..."; mask = sum of 2**id of the hanging points; element = node ids,
# converted to VTK orientation (checked: every element > 0, Σ volume = 1, conformal, on the unit cube).
# Pattern lookup over the 24 cube rotations as in MixedOcTree TransitionTemplate.cpp:170-190.
_MOT_TABLE = """
2048:4 0 1 5 11,5 1 2 6 11,6 2 3 7 11,4 5 6 7 11
526336:0 1 11 4 5 19,1 2 11 5 6 19,2 3 11 6 7 19
133120:17 1 2 11,4 7 11 17,5 1 17 11,4 5 17 11,17 2 6 11,7 17 6 11,4 0 1 5 11,6 2 3 7 11
10240:13 0 1 11,5 4 13 11,4 0 13 11,1 11 2 13,5 13 6 11,6 13 2 11,6 2 3 7 11,7 4 5 6 11
18432:11 14 1 5,11 2 1 14,11 14 5 6,11 3 2 14,11 3 14 7,11 14 6 7,4 0 1 5 11,5 6 7 4 11
272384:13 0 1 11,5 4 13 11,4 0 13 11,3 7 18 11,7 4 18 11,18 4 5 11,2 18 6 11,18 2 3 11,2 13 1 11,6 5 13 11,13 2 6 11,18 5 6 11
534528:13 0 1 11,19 4 5 13,13 1 2 11,5 13 6 19,11 0 4 19 13,6 2 11 19 13,2 3 11 6 7 19
6400:0 11 8 12,11 8 12 6,2 8 11 6,6 11 7 12,8 5 12 6,5 1 2 6 8,6 2 3 7 11,7 4 5 6 12
657408:17 1 2 26,17 2 6 26,1 17 5 26,0 11 1 26,2 11 3 26,1 11 2 26,3 11 19 7 26,11 0 4 19 26,4 0 1 5 26,6 2 3 7 26,19 4 5 17 26,19 17 6 7 26
530432:0 1 12 11,5 4 12 19,5 1 11 19 12,2 3 11 6 7 19,1 2 11 5 6 19
526592:19 4 5 8,6 8 19 5,0 4 19 11 8,5 1 2 6 8,2 11 19 6 8,2 3 11 6 7 19
527360:19 6 7 10,5 10 6 19,7 3 11 19 10,5 1 2 6 10,11 1 5 19 10,0 1 11 4 5 19
17340416:1 12 24 5,6 15 24 2,6 24 19 5,2 11 24 1,0 12 24 11 1,11 24 15 3 2,15 24 19 7 6,12 4 19 24 5,1 2 6 5 24
657920:0 11 9 1 4 19 17 5,11 3 2 9 19 7 6 17
550912:0 11 1 13,4 19 13 5,19 7 14 6,11 3 2 14,11 14 13 19,1 2 14 13 11,5 13 14 6 19,3 11 19 7 14,11 0 4 19 13
38144:5 8 12 6 10 15,2 6 10 1 5 8,6 7 15 5 4 12,8 0 12 10 3 15
538624:11 2 1 13,11 13 12 19,19 6 13 5,12 0 1 13 11,4 12 13 5 19,11 19 6 2 13,3 11 2 7 19 6
17340928:24 6 5 19,24 9 5 6,19 7 15 24 6,4 19 24 12 5,12 24 9 1 5,24 15 2 9 6,0 12 1 11 24 9,11 24 9 3 15 2
17340672:5 6 19 24,6 2 15 24,5 8 24 12,5 8 1 24,2 8 11 24,1 8 2 24,15 24 19 7 6,12 4 19 24 5,15 3 11 24 2,0 12 24 11 8,1 2 6 5 24
300288:1 26 5 8,26 6 5 18,26 15 10 18,12 26 8 5,12 26 5 4,4 26 5 18,4 7 26 18,7 15 26 18,10 18 6 2 26,8 10 2 1 26,1 2 6 5 26,4 7 15 12 26,12 15 10 8 26,0 12 8 3 15 10
17357312:2 14 9 26,7 6 26 19,26 14 9 5,19 5 6 26,6 5 14 26,1 9 5 26,4 19 26 5,0 26 1 12,4 26 12 5,12 26 1 5,3 15 14 2 26,11 3 2 9 26,24 15 3 11 26,19 7 15 24 26,15 7 6 14 26,0 11 9 1 26,4 19 24 12 26,12 24 11 0 26
17364992:0 1 11 12 13 24,2 3 11 14 15 24,14 15 24 6 7 19,12 13 24 4 5 19,1 2 11 13 14 24,13 14 24 5 6 19
6144:1 0 11 12,1 5 12 11,7 11 12 6,11 5 12 6,7 4 5 6 12,6 2 3 7 11,5 1 2 6 11
18390784:5 6 20 9,6 5 24 19,5 6 24 20,12 4 19 24 5,24 19 7 15 6,9 1 8 20 5,9 20 10 2 6,12 24 20 8 5,24 15 10 20 6,11 20 24 0 8 12,15 3 10 24 11 20
20561664:16 6 19 26,4 19 26 16,1 9 13 26,5 6 16 26,6 13 9 26,13 6 5 26,7 26 19 6,3 15 10 26,10 15 6 26,15 7 6 26,10 2 26 6,9 2 6 26,0 11 20 8 26,12 0 8 21 26,12 24 11 0 26,4 19 24 12 26,4 12 21 16 26,8 20 9 1 26,13 21 8 1 26,5 16 21 13 26,11 3 10 20 26,24 15 3 11 26,19 7 15 24 26,20 10 2 9 26
28179712:15 23 24 3 10 11,7 18 19 15 23 24,0 8 11 12 21 24,12 21 24 4 16 19,11 8 10 24 21 23,24 21 23 19 16 18,21 23 14 13 16 18 6 5,8 10 2 1 21 23 14 13
63962624:12 13 16 26,16 13 5 26,4 12 16 26,11 10 9 26,10 2 9 26,3 10 11 26,12 0 1 13 26,0 11 9 1 26,13 1 9 22 26,5 13 22 17 26,16 5 17 25 26,19 4 16 25 26,4 19 24 12 26,0 12 24 11 26,22 9 2 14 26,10 23 14 2 26,17 22 14 6 26,6 14 23 18 26,18 25 17 6 26,23 10 3 15 26,15 3 11 24 26,18 23 15 7 26,15 24 19 7 26,19 25 18 7 26
33554176:21 8 1 13 26,21 13 5 16 26,12 0 8 21 26,4 12 21 16 26,0 11 20 8 26,8 20 9 1 26,20 10 2 9 26,11 3 10 20 26,12 4 19 24 26,24 19 7 15 26,15 3 11 24 26,0 12 24 11 26,14 2 10 23 26,6 14 23 18 26,23 10 3 15 26,18 23 15 7 26,13 1 9 22 26,22 9 2 14 26,5 13 22 17 26,17 22 14 6 26,19 16 17 18 26,19 4 16 26,5 17 16 26,6 18 17 26,7 19 18 26
3358464:0 11 12 8 20 21,8 20 21 1 9 13,16 4 12 21 7,11 20 21 12 7,10 20 11 3 7,5 16 21 13 17,20 9 13 21 17,2 6 10 9 17 20,7 16 17 21,7 21 17 20,10 6 17 20 7
3751680:8 20 21 1 9 13,0 11 12 8 20 21,5 16 21 13 6,20 9 13 21 6,20 10 2 9 6,16 4 12 21 19,11 20 21 12 19,11 19 20 3 7 10,16 6 19 21,21 6 19 20,7 10 20 19 6
67108608:0 11 20 8 12 24 26 21,8 20 9 1 21 26 22 13,12 24 26 21 4 19 25 16,21 26 22 13 16 25 17 5,11 3 10 20 24 15 23 26,24 15 23 26 19 7 18 25,20 10 2 9 26 23 14 22,26 23 14 22 25 18 6 17
20578048:1 13 21 8 26,9 1 8 20 26,13 1 9 26,2 9 20 10 14,26 9 14 20,9 14 13 26,26 10 20 14,3 10 20 11 26,15 3 11 24 26,15 10 3 26,10 15 14 26,12 0 8 21 26,0 11 20 8 26,0 12 24 11 26,6 14 15 7 26,7 15 24 19 26,19 6 7 26,4 19 24 12 26,19 4 16 26,16 4 12 21 26,16 6 19 26,5 16 21 13 26,16 5 6 26,5 13 14 6 26
1064704:9 1 8 20 13,0 11 20 8 12,10 2 9 20 6,11 3 10 20 7,7 12 13 6 20,20 12 13 8,7 20 6 10,13 6 20 9,7 12 20 11,13 5 6 12 4 7
2192128:12 0 8 21 3,1 13 21 8 9,3 9 8 21,9 14 21 3,14 9 21 13,7 3 12 21,3 14 21 7,16 4 12 21 7,21 14 6 16 7,6 5 16 14 13 21,2 14 9 3
2210048:1 13 21 8 2,12 0 8 21 11,11 2 8 21,15 11 21 2,11 15 21 12,2 6 13 21,15 2 21 6,5 16 21 13 6,15 21 16 7 6,15 12 21 7 4 16,15 3 11 2
9984:10 9 13 6,10 9 8 13,9 1 8 13,10 2 9 6,7 3 10 4 0 8,8 4 7 10 13,7 6 10 13,7 4 5 6 13
7424:11 10 12 7,11 10 8 12,0 11 8 12,3 10 11 7,5 1 8 6 2 10,5 8 10 6 12,7 6 10 12,5 6 7 4 12
71680:0 11 1 12,6 2 3 7 11,16 4 12 7,12 7 11 16,11 1 12 16,1 16 11 5,16 6 7 11,5 11 6 16,5 1 2 6 11
74240:9 1 0 13,3 7 6 2 9,5 16 13 6,6 13 9 16,0 9 13 16,16 0 9 4,7 16 6 9,9 4 7 16,0 4 7 3 9
6520576:13 21 22 1 8 9,8 11 21 9 10 22,11 3 10 7,12 0 8 21 11,2 14 22 9 10,10 22 21 11 7,11 21 12 7,22 10 14 7,16 4 12 21 7,16 21 22 17 7,6 17 22 14 7,5 16 17 13 21 22
55573248:0 12 24 11 8,3 11 24 15 26,19 24 25 4 12 16,7 15 18 19 24 25,26 11 24 8,2 9 11 3 26,8 11 9 26,14 2 3 15 26,9 2 14 22 26,26 13 22 8,9 26 22 8,12 26 24 8,13 1 9 22 8,12 16 25 24 26,15 24 25 18 26,18 14 15 26,22 14 18 25 26,13 22 25 16 26,8 13 16 12 26,5 13 16 17 22 25,17 22 25 6 14 18
4420352:12 0 8 3,16 4 12 7,9 2 14 22 3,14 3 7 22,8 3 9 22,14 6 17 22 7,8 22 12 3,3 12 7 22,13 1 9 22 8,12 22 16 7,17 7 16 22,17 5 13 22 16,8 13 16 12 22
31457024:1 13 8 26,9 1 8 20 26,13 1 9 22 26,2 14 22 9 26,10 2 9 20 26,2 10 23 14 26,3 15 23 10 26,3 10 20 11 26,15 3 11 24 26,0 12 24 11 26,0 11 20 8 26,12 0 8 26,12 4 19 24 16,26 16 12 24,16 26 19 24,15 7 18 23 26,7 15 24 19 26,7 19 18 26,6 14 23 18 26,6 17 22 14 26,6 18 17 26,26 17 22 16,13 26 22 16,5 13 22 17 16,12 8 13 16 26,19 16 17 18 26
56622848:19 24 25 4 12 16,7 15 18 19 24 25,22 17 25 13 5 16,14 6 18 22 17 25,8 0 12 20 11 24,20 11 24 10 3 15,1 8 13 9 20 22,9 20 22 2 10 14,8 13 16 12 20 22 25 24,20 22 25 24 10 14 18 15
17602816:12 4 19 24 5,7 15 24 19 18,11 0 12 24 8,15 3 11 24 2,11 2 8 24,19 5 18 24,5 12 8 24,18 2 15 24,8 1 5 24,18 5 6 24,2 1 8 24,2 18 6 24,5 1 2 6 24
17358336:2 14 9 10,15 3 11 24 10,9 11 10 24,24 14 15 10,24 9 14 10,14 15 24 6 7 19,4 19 24 12 5,11 9 24 0 1 12,9 24 14 5,1 12 24 9 5,19 6 14 24 5
21691136:11 0 12 24 8,22 9 11 24 8,13 1 9 22 8,24 12 13 22 8,12 24 22 13 4 19 17 5,2 9 11 3 14 22 24 15,14 22 24 15 6 17 19 7
26016768:12 13 24 4 5 19,0 1 11 12 13 24,7 18 19 15 23 24,15 23 24 3 10 11,13 23 24 5 18 19,1 10 11 13 23 24,13 14 23 5 6 18,1 2 10 13 14 23
14155520:0 11 20 8 12,15 7 18 23 26,11 15 3 26,11 3 10 20 26,3 15 23 10 26,15 11 12 26,12 4 7 15 26,11 26 20 12,26 8 20 12,2 14 22 9 26,10 2 9 20 26,2 10 23 14 26,18 6 14 23 26,6 17 22 14 26,6 18 17 26,7 4 16 18 26,18 16 17 26,1 8 20 9 26,13 1 9 22 26,1 13 8 26,17 5 13 22 16,8 13 16 12 26,16 4 12 26,13 26 22 16,26 17 22 16
623616:19 4 16 0,19 7 15 6,3 15 10 0,0 19 15 16,6 19 16 15,0 15 10 16,15 6 10 16,16 0 1 10,16 1 5 10,16 5 6 10,5 1 2 6 10
80128:0 11 8 12,16 4 12 7,12 7 11 16,3 11 7 16,6 2 3 7 16,2 11 3 16,11 2 8 13,16 11 13 2,8 13 16 12 11,1 13 8 2,13 2 6 16,5 16 13 6
10407680:10 20 23 3 11 15,2 9 14 10 20 23,7 18 23 15 19,18 6 14 23 17,19 17 18 23,20 11 15 23 19,9 20 23 14 17,17 19 20 23,0 4 8 11 19 20,4 5 8 19 17 20,9 17 20 1 5 8
983040:5 17 16 1,19 4 16 0,7 19 18 3,6 18 17 2,0 19 3 16,19 16 17 18 3,18 2 3 17,0 16 3 17,0 3 2 1 17,16 0 1 17
26147840:13 1 2 14 26,1 10 2 26,2 10 23 14 26,10 1 11 26,6 14 23 18 26,11 24 23 10 26,24 19 18 23 26,6 17 14 26,17 6 18 26,17 13 14 26,17 5 13 26,19 17 18 26,11 1 0 26,1 13 12 0 26,13 5 4 12 26,5 17 19 4 26,12 4 19 24 26,0 12 24 11 26,15 24 23 7 19 18,3 11 10 15 24 23
13563392:0 12 15 3 26,10 0 3 26,10 3 15 23 26,0 10 9 26,15 7 18 23 26,22 9 10 23 26,17 22 23 18 26,19 7 15 26,7 19 18 26,12 19 15 26,4 19 12 26,19 17 18 26,0 9 1 26,12 0 1 13 26,4 12 13 5 26,19 4 5 17 26,5 13 22 17 26,13 1 9 22 26,6 17 18 14 22 23,14 22 23 2 9 10
4613888:0 3 10 8 22,8 10 9 22,10 3 7 18 22,10 18 14 22,2 14 22 9 10,0 4 7 3 22,4 18 7 22,18 4 17 22,4 0 8 22,13 4 8 22,17 5 13 22 4,13 1 9 22 8,6 17 22 14 18
411136:4 7 18 0 3 10,4 18 17 0 10 9,0 1 9 4 5 17,9 17 18 10 14,6 17 14 18,2 14 9 10
34570496:7 19 25 18 15,8 0 3 15,19 0 8 25 15,2 8 3 15,25 2 15 8,18 2 15 25,6 18 25 17 2,1 17 25 8 2,1 5 17 8 16 25,8 16 25 0 4 19
34554112:17 6 18 25 14,1 8 2 14,1 17 25 8 14,8 3 2 14,3 25 14 8,3 18 14 25,18 7 19 25 3,19 0 8 25 3,8 16 25 0 4 19,1 5 17 8 16 25
26213376:0 1 11 12 13 24,1 10 11 13 23 24,1 2 10 13 14 23,15 23 24 3 10 11,7 18 19 15 23 24,4 19 24 12 16,18 6 14 23 17,17 18 23 16 19 24,24 16 17 23 13,16 12 13 24,17 13 14 23,17 5 13 16
1875712:11 3 10 20 15,7 19 18 15,0 11 20 8 12,15 11 12 19 20,19 12 5 20,4 19 12 5,5 12 8 20,18 10 15 20,20 18 19 15,19 5 18 20,10 18 20 2 6 9,18 6 9 20 5,9 1 8 20 5
156672:0 4 7 3 10,0 10 1 13,0 13 4 10,4 13 14 7 10,13 1 2 14 10,13 4 7 14 17,17 5 13 4,6 17 14 7
303104:0 3 2 1 13,15 4 7 18,0 15 3 13,15 0 4 13,15 2 3 13,4 13 5 15,18 4 5 15,5 15 13 18,18 2 15 13,2 18 6 13,5 13 6 18
606208:19 4 16 0,0 3 2 1 14,14 3 7 19,19 6 7 14,0 19 3 14,19 0 16 14,16 0 1 14,1 14 5 16,16 6 19 14,16 5 6 14
856320:19 4 16 12,18 3 7 19,12 0 8 3,19 3 12 26,3 19 18 26,19 12 16 26,19 16 18 26,18 2 3 26,2 18 6 26,26 8 3 12,2 8 3 26,8 16 12 26,1 8 2 26,8 1 5 16 26,16 5 6 18 26,5 1 2 6 26
562176:5 1 2 6 10,10 1 0 12,12 1 5 10,12 5 6 15 10,0 12 15 3 10,5 12 15 6 19,4 19 12 5,19 7 15 6
282624:2 1 0 3 12,5 14 6 18,14 1 2 12,1 14 5 12,3 14 2 12,12 5 4 14,5 18 4 14,14 4 12 18,3 18 14 12,18 3 7 12,12 4 7 18
1470208:2 9 20 10 14,17 6 18 14,9 1 8 20 13,9 14 17 13 20,13 17 4 20,17 5 13 4,13 4 8 20,10 18 14 20,18 20 17 14,4 17 18 20,3 7 11 10 18 20,7 18 20 11 4,0 11 20 8 4
967680:10 0 3 15,1 10 2 14,7 18 15 19,18 0 15 19,10 15 18 14 0,0 10 1 14,0 14 1 18,19 0 1 17 18,17 1 14 18,6 18 17 14,1 5 17 0 4 19
8775168:3 15 23 10 26,10 23 14 2 26,15 7 18 23 26,23 18 6 14 26,2 9 10 26,2 14 9 26,0 3 10 26,0 15 3 26,0 4 15 26,4 7 15 26,0 10 9 26,4 16 18 7 26,0 16 4 26,0 9 1 13,0 9 13 26,0 13 16 26,13 9 14 26,5 16 13 26,5 6 18 16 26,5 13 14 6 26
17350144:3 11 24 15 26,11 0 12 24 26,15 24 19 7 26,24 12 4 19 26,3 10 11 26,3 15 10 26,9 11 10 26,0 11 9 1 26,6 15 7 26,6 10 15 26,6 7 19 26,2 9 10 6,6 9 10 26,5 19 4 26,5 6 19 26,0 1 13 12 26,12 13 5 4 26,1 9 13 26,9 6 13 26,5 13 6 26
17537280:6 7 15 17 19 24,0 12 24 11 8,3 11 24 15 2,4 19 24 12 16,6 15 24 17 2,16 17 19 24,8 16 12 24,8 11 2 24,24 8 17 2,16 8 17 24,1 8 2 17,1 5 16 8 17
17734656:11 24 15 3 10,7 15 24 19 18,10 15 18 24,10 24 18 17,2 10 18 6 17,17 18 19 24,0 12 24 11 1,1 11 10 24,1 24 10 17,4 5 12 19 17 24,12 5 17 24 1,1 10 2 17
35626752:3 10 20 11 15,0 11 20 8 12,7 19 25 18 15,4 16 25 19 12,12 19 15 11 25,11 20 25 15,11 25 20 12,10 18 25 20 15,8 20 25 16 12,1 8 20 9 5 16 25 17,2 9 20 10 6 17 25 18
819456:0 3 8 15,7 18 15 19,0 19 15 8,0 4 19 8,15 19 5 8,4 5 19 8,5 18 19 15,2 8 3 15,1 8 2 15,2 15 18 1,1 5 8 15,15 5 18 1,1 2 6 5 18
9887488:11 3 15 20 10 23,20 10 23 9 2 14,0 11 20 8 12,11 15 23 20 12,4 7 18 12 15 23,23 12 4 18 17,6 14 23 18 17,4 5 17 12,20 23 17 12,9 20 23 14 17,8 20 17 5 12,1 8 5 9 20 17
34580480:7 19 25 18 15,3 11 15 2,11 19 15 25,0 4 16 11 19 25,0 11 1 13,0 16 25 11 13,1 11 2 13,2 15 18 25,2 25 11 15,6 18 25 17 2,2 11 25 13,2 13 25 17,5 17 25 16 13
51362816:15 18 7 24 25 19,4 16 12 19 25 24,0 1 11 12 13 24,3 11 24 15 10,15 24 25 18 10,1 13 24 11 10,13 24 10 25,12 16 25 24 13,10 18 25 2 6 17,1 10 2 13,2 10 25 17 13,5 17 25 16 13
1617664:11 20 19 0 8 4,3 10 20 11 15,6 7 19 15,11 20 19 15,6 10 15 20,2 9 20 10 6,6 19 20 15,4 19 20 8 13,4 13 5 19,1 8 20 9 13,6 13 9 20,6 19 13 20,6 13 19 5
44288:3 15 10 11,0 11 8 4,1 8 10 2 13,8 11 10 13,4 8 13 11,10 11 15 13,4 15 11 13,4 7 15 13,2 10 6 13,10 15 6 13,15 7 6 13,4 5 6 7 13
18546432:24 20 11 15 10 3,0 12 8 11 24 20,10 20 24 15 26,20 8 12 24 26,15 24 19 7 26,24 12 4 19 26,10 15 14 26,14 15 7 6 26,6 7 19 17 26,4 5 17 19 26,14 6 17 26,10 14 20 26,2 9 20 10 14,9 26 20 14,8 26 20 13,8 13 12 26,9 20 26 13,1 8 20 9 13,12 13 5 4 26,5 13 17 26,13 9 14 17 26
44544:0 11 9 1 13,0 4 11 13,3 15 10 11,2 9 10 6,9 11 10 15,6 10 15 9,11 13 15 9,11 15 13 4,6 13 9 15,6 5 13 15,4 13 5 15,4 5 6 7 15
28311296:15 24 23 7 19 18,3 11 10 15 24 23,24 12 21 19 4 16,11 0 8 24 12 21,18 23 24 19 26,23 10 11 24 26,19 24 21 16 26,24 11 8 21 26,8 11 10 9 26,9 10 23 26,8 9 21 26,1 13 21 8 9,2 10 23 14 9,9 13 21 26,9 23 14 26,17 23 18 26,17 14 23 26,6 14 23 18 17,16 21 17 26,13 17 21 26,5 16 21 13 17,16 17 18 19 26,13 9 14 17 26
837632:0 1 11 4 5 19,2 3 15 14 11,7 19 18 15,14 15 18 19,11 19 15 14,1 11 2 14,1 5 19 11 14,5 18 19 14,5 6 18 14
17496064:3 11 24 15 26,0 12 24 11 26,15 24 19 7 26,12 4 19 24 26,2 11 3 26,1 11 2 26,2 3 15 14 26,14 15 7 6 26,6 7 19 17 26,1 0 11 26,6 17 14 26,5 17 19 4 26,4 12 13 5 26,0 1 13 12 26,5 13 17 26,13 14 17 26,1 2 14 13 26
27262720:0 12 24 11 26,12 4 19 24 26,11 24 15 3 26,24 19 7 15 26,2 10 23 14 26,10 3 15 23 26,14 23 18 6 26,23 15 7 18 26,1 8 20 9 26,9 20 10 2 26,10 20 11 3 26,8 0 11 20 26,4 16 19 26,7 19 18 26,6 18 17 26,4 12 16 26,0 8 12 26,6 17 14 26,2 14 9 26,1 13 8 26,1 9 13 26,13 17 16 26,16 17 18 19 26,13 9 14 17 26,8 13 16 12 26,13 16 17 5
604160:0 1 13 12 11,2 3 11 6 7 19,1 11 2 13,4 19 12 16,11 12 19 13,12 13 16 19,2 11 19 6 13,6 19 16 13,5 6 16 13
602880:0 8 12 3,3 12 19 8,3 19 7 8,2 3 7 6 9,8 3 9 7,4 19 12 16,8 7 9 13,8 13 19 7,9 13 7 6,1 9 13 8,8 13 16 12 19,6 7 19 13,6 19 16 13,5 6 16 13
2225920:0 12 15 3 26,12 4 7 15 26,0 3 10 8 26,8 10 9 26,1 8 9 26,3 15 10 26,14 10 15 26,14 15 7 6 26,2 14 9 10,9 10 14 26,1 13 21 8 26,13 5 16 21 26,8 21 12 0 26,4 12 21 16 26,1 9 13 26,4 16 7 26,6 7 16 26,9 14 13 26,5 6 16 26,13 14 6 5 26
2995456:0 3 10 8 26,0 12 15 3 26,1 8 10 2 26,12 19 15 26,4 19 12 26,15 19 7 18,4 16 19 26,15 19 18 26,16 18 19 26,10 3 15 26,10 15 18 26,2 10 18 6 26,5 6 18 16 26,5 13 6 26,2 6 13 26,1 2 13 26,0 8 21 12 26,12 21 16 4 26,1 13 21 8 26,21 13 5 16 26
1269504:3 7 10 26,3 11 7 26,3 10 20 11 26,7 11 12 26,4 7 12 16,12 16 7 26,0 12 11 26,0 8 12 26,0 11 20 8 26,8 16 12 26,1 8 20 9 26,2 9 20 10 26,7 14 10 26,2 10 14 26,2 14 9 26,6 14 7 26,6 7 17 26,7 16 17 26,6 17 14 26,5 17 16 26,9 14 17 26,1 9 17 5 26,1 5 16 8 26
19644160:12 21 24 4 16 19,0 8 11 12 21 24,6 7 15 17 19 24,3 11 24 15 10,11 10 24 8 9 21,1 13 21 8 9,24 17 6 15 10,21 24 10 9 17,13 9 17 21,2 6 17 9 10,5 16 21 13 17,16 19 24 21 17
2586368:0 3 12 8 10 21,10 21 18 3 12 7,16 18 21 4 7 12,10 18 14 21,14 18 17 21,16 17 18 21,6 18 17 14,2 14 9 10,8 10 9 21,9 10 14 21,5 16 21 13 17,1 13 21 8 9,13 9 14 17 21
67328:0 4 7 3 26,0 3 10 8 26,0 8 16 4 26,3 7 10 26,6 10 7 26,4 16 7 26,6 7 16 26,8 10 9 26,2 9 10 26,2 10 6 26,2 6 9 26,1 8 9 26,1 5 16 8 26,1 9 5 26,5 9 6 26,5 6 16 26
75008:0 4 7 3 26,0 3 10 8 26,1 8 10 2 26,0 8 16 4 26,4 16 7 26,10 3 7 26,6 10 7 26,6 7 16 26,2 10 6 26,1 13 8 26,8 13 16 26,5 16 13 26,5 6 16 26,5 13 6 26,1 2 13 26,2 6 13 26
18734848:11 20 24 0 8 12,3 10 15 11 20 24,10 20 24 15 26,20 8 12 24 26,2 9 20 10 26,1 8 20 9 26,7 15 24 19 26,4 19 24 12 26,7 18 15 26,7 19 18 26,16 18 19 26,4 16 19 26,4 12 16 26,8 16 12 26,14 10 15 18 26,1 5 16 8 26,6 14 18 26,5 6 18 16 26,5 14 6 26,2 10 14 26,2 14 9 26,5 9 14 26,1 9 5 26
826624:7 18 15 19,4 5 19 12,0 11 8 12,3 10 11 15,12 19 15 11 26,5 19 12 26,10 11 15 26,11 8 12 26,8 11 10 26,8 5 12 26,10 15 18 26,15 19 18 26,5 18 19 26,2 10 18 6 26,1 8 10 2 26,5 6 18 26,1 5 8 26,1 2 6 5 26
21953280:3 11 24 15 2 9 22 14,6 17 22 14 18,17 19 24 22 18,7 15 24 19 18,14 22 24 15 18,1 9 22 13 8,9 11 24 22 8,0 12 24 11 8,13 22 24 12 8,5 13 22 17 4 12 24 19
1658624:11 19 7 3 26,0 4 19 11 26,0 8 16 4 26,1 5 16 8 26,1 8 20 9 26,9 20 10 2 26,10 20 11 3 26,20 8 0 11 26,3 7 10 26,7 14 10 26,2 10 14 26,2 14 9 26,6 14 7 26,6 7 19 26,4 16 19 26,6 19 16 26,5 6 16 26,5 14 6 26,5 9 14 26,1 9 5 26
51370752:19 24 25 4 12 16,7 15 18 19 24 25,0 12 24 11 26,11 24 15 3 26,2 3 15 14 26,2 9 11 3 26,0 11 8 26,8 11 9 26,1 8 9 26,14 15 18 26,6 14 18 26,6 17 14 26,9 14 17 26,9 2 14 26,6 18 25 17 26,24 25 18 15 26,12 16 25 24 26,0 8 12 26,8 16 12 26,5 17 25 16 26,1 5 16 8 26,1 9 17 5 26
929792:6 18 17 2,7 19 18 3,2 17 19 3 18,3 19 12 2 17 13,0 3 12 1 2 13,4 12 19 5 13 17
48384:10 3 15 11,0 12 11 8,11 15 10 26,8 12 11 26,12 15 11 26,8 11 10 26,1 8 10 2 26,12 4 7 15 26,6 10 15 26,7 6 15 26,2 10 6 26,8 13 12 26,1 13 8 26,1 2 13 26,2 6 13 26,5 13 6 26,12 13 5 4 26,4 5 6 7 26
1388288:0 11 4 8 20 16,4 11 7 16 20 18,11 3 7 20 10 18,20 10 18 9 2 6,8 13 16 20,1 8 20 9 13,13 18 16 20,9 20 18 6 13,5 6 18 16 13
858368:11 19 7 3 26,3 7 18 26,7 19 18 26,2 11 3 26,2 3 18 26,2 18 6 26,4 16 19 12,0 11 8 12,8 11 19 16 12,8 16 19 11 26,2 8 11 26,16 18 19 26,1 8 2 26,1 5 16 8 26,5 6 18 16 26,1 2 6 5 26
8897792:16 18 23 4 7 15,0 15 4 8 23 16,0 3 15 8 10 23,8 10 23 1 2 14,16 17 18 23,8 17 16 23,6 14 23 18 17,1 5 16 8 17,1 8 23 14 17
42981120:0 3 10 8 26,0 12 15 3 26,0 8 12 26,12 19 15 26,3 15 23 10 26,2 10 23 14 26,2 9 10 26,8 10 9 26,2 14 9 26,1 8 9 26,15 7 19 23 18 25,23 18 25 14 6 17,23 15 19 25 26,14 23 25 17 26,9 14 17 26,4 19 12 26,4 12 16 26,8 16 12 26,4 16 25 19 26,5 17 25 16 26,1 5 16 8 26,1 9 17 5 26
17604352:3 11 24 15 10,0 12 24 11 8,7 15 24 19 18,4 19 24 12 5,8 11 10 9 24,8 5 12 24,5 18 19 24,10 15 18 24,9 10 18 24,2 10 18 6 9,8 9 5 24,5 9 18 24,5 6 18 9,1 8 9 5
333824:3 7 18 11,2 3 18 11,7 11 12 18,0 12 11 1,16 18 7 4 12,12 16 18 11,1 16 12 11,1 2 18 16 11,2 6 18 1 5 16
233472:3 2 15 0 1 12,12 4 7 15 16,6 7 17 15,7 16 17 15,2 6 17 15,12 17 16 15,1 12 15 2 17,1 16 12 17,5 17 16 1
9427968:3 15 23 10 11,2 10 23 14 1,1 11 10 23,0 12 11 1,4 19 12 16,15 7 18 23 19,11 12 19 15 23,12 16 19 23,16 17 18 19 23,1 12 11 23,1 16 12 23,1 17 16 23,1 17 5 16,1 14 17 23,14 23 18 6 17
9362432:15 7 18 23 19,6 14 23 18 17,17 18 19 23,3 15 23 10 11,2 10 23 14 1,1 11 10 23,11 12 19 15 23,12 17 19 23,0 11 1 12,1 12 11 23,1 17 12 23,1 14 17 23,1 17 5 12,4 5 17 19 12
26148608:7 18 19 15 23 24,15 23 24 3 10 11,23 24 19 18 26,10 11 24 23 26,8 11 10 9 26,12 4 19 24 26,0 12 24 11 26,0 11 8 26,0 8 12 26,6 14 23 18 26,17 6 18 26,6 17 14 26,2 10 23 14 26,9 10 2 26,9 2 14 26,17 18 19 26,8 9 13 26,8 13 9 1,8 13 12 26,12 13 5 4 26,4 5 17 19 26,5 13 17 26,13 9 14 17 26
108800:2 3 15 11,0 8 16 4 11,4 15 11 16,4 16 7 15,6 7 16 15,11 15 2 16,2 15 6 16,2 8 11 16,8 13 16 2,1 13 8 2,2 6 13 16,5 13 6 16
9961216:23 20 10 14 9 2,3 15 11 10 23 20,9 20 23 14 26,20 11 15 23 26,0 11 20 8 12,11 12 15 26,20 12 11 26,23 15 7 18 26,12 4 7 15 26,16 18 7 4 26,4 12 16 26,20 8 12 26,14 23 18 6 17,17 23 18 26,17 14 23 26,16 17 18 26,1 8 20 9 13,13 20 9 26,13 8 20 26,13 17 16 26,5 13 17 16,12 8 13 16 26,13 9 14 17 26
1879808:11 3 10 20 15,18 7 19 15,10 15 18 20,11 19 15 20,18 15 19 20,11 20 19 0 8 4,2 9 6 10 20 18,4 19 20 8 13,1 8 20 9 13,13 18 19 20,9 20 18 6 13,5 19 4 13,5 18 19 13,5 6 18 13
860672:9 3 2 18,2 18 6 9,0 3 9 13,0 9 1 13,3 19 7 18,0 16 4 19,0 3 18 16 19,0 16 18 3 13,3 9 13 18,13 9 6 18,5 6 18 16 13
17537792:9 11 24 2 3 15,9 24 17 2 15 6,6 7 15 17 19 24,4 19 24 12 16,16 17 19 24,8 16 12 24,0 12 24 11 8,8 11 9 24,8 9 17 16 24,1 9 8 5 17 16
157696:0 11 1 13,2 11 3 14,1 2 14 13 11,3 7 14 11,0 4 11 13,4 13 14 7 11,4 7 14 13 17,4 5 17 13,6 7 17 14
42240:0 3 10 8 15,8 15 10 13,0 8 4 15,8 13 4 15,4 7 15 13,1 8 10 2 13,6 15 7 13,6 10 15 13,2 10 6 13,4 5 6 7 13
942592:2 3 14 9,3 7 18 19,0 3 9 19,3 18 14 19,9 3 14 19,17 18 19 14,17 6 18 14,0 9 1 13,0 19 9 13,0 4 19 13,5 17 19 4 13,13 9 14 17 19
745984:2 3 14 9,3 7 14 19,6 7 19 17 14,0 3 9 19,3 14 9 19,0 9 1 13,0 9 13 19,0 16 4 19,0 13 16 19,16 17 19 13,5 17 16 13,9 14 17 13 19
43520:2 9 11 3 15,0 11 9 1 13,13 9 15 11,0 13 4 11,4 15 11 13,9 15 2 6,13 9 6 15,4 13 5 15,5 13 6 15,4 5 6 7 15
17669376:0 12 24 11 8,3 11 24 15 10,4 19 24 12 16,24 19 7 15 18,16 18 19 24,10 15 18 24,8 11 10 24,8 16 12 24,8 10 18 16 24,8 16 18 10 1 5 6 2
877056:0 4 19 16,3 19 7 18,0 3 18 16 19,0 9 1 13,2 9 3 14,0 13 14 3 9,3 14 18 0 13 16,5 13 16 6 14 18
17472768:4 5 12 19 17 24,19 17 24 7 6 15,0 12 24 11 8,11 24 15 3 10,8 11 10 24,6 15 24 17 10,5 17 24 12 8,8 10 17 24,2 6 17 10,1 17 5 8,1 8 10 2 17
295680:0 3 8 15,3 9 8 15,2 9 3 15,0 8 4 15,2 15 18 9,2 6 9 18,8 9 18 15,4 18 7 15,4 18 15 8,4 8 5 18,8 9 5 18,5 6 18 9,1 8 9 5
661760:3 11 19 7 26,11 12 19 26,2 11 3 26,2 3 7 6 26,6 7 19 17 26,2 6 17 26,2 8 11 26,1 8 2 26,1 2 17 26,0 11 8 12,8 12 11 26,19 12 4 26,5 17 19 4 26,5 4 12 26,5 12 8 26,1 5 8 26,1 17 5 26
303616:2 9 3 15,0 3 9 15,2 15 18 9,2 18 6 9,4 18 7 15,0 9 1 13,0 4 15 13,13 4 15 18,13 9 18 15,4 5 18 13,5 6 18 13,9 6 13 18,13 9 15 0
625920:10 3 15 11,10 11 15 26,8 11 10 26,6 10 15 26,11 19 15 26,6 15 7 19,6 15 19 26,6 2 10 26,0 8 11 4 16 19,8 16 19 11 26,6 19 16 26,1 8 10 2 26,5 6 16 26,8 1 5 16 26,1 2 6 5 26
101632:10 3 15 11,10 11 15 26,4 15 11 26,6 10 15 26,8 11 10 26,0 11 8 26,0 4 11 26,0 8 16 4 26,1 8 10 2 26,8 1 5 16 26,2 10 6 26,6 15 7 26,4 7 15 26,4 16 7 26,6 7 16 26,5 6 16 26,1 2 6 5 26
1021952:2 11 3 15,0 11 1 12,15 7 18 19,12 4 19 16,11 12 19 15 18,2 15 18 11,11 12 18 17,16 17 18 19 12,2 18 17 11,2 6 17 18,1 11 2 17,1 12 11 17,1 16 12 17,1 16 17 5
887040:15 7 18 19,2 11 3 15,2 11 15 26,11 19 15 26,2 15 18 26,15 19 18 26,2 8 11 26,2 18 6 26,1 8 2 26,0 8 11 4 16 19,8 16 19 11 26,16 18 19 26,5 6 18 16 26,1 5 16 8 26,1 2 6 5 26
17627648:0 12 24 11 26,12 4 19 24 26,11 24 15 3 26,24 19 7 15 26,0 11 9 1 26,11 3 2 9 26,2 3 15 14 26,0 1 13 12 26,12 13 5 4 26,1 9 13 26,13 9 14 26,9 2 14 26,15 7 18 26,14 15 18 26,6 14 18 26,4 5 19 26,5 18 19 26,7 19 18 26,5 6 18 26,13 14 6 5 26
545280:6 7 19 14,3 7 14 19,2 9 3 14,0 3 9 12,9 14 19 12 3,9 12 19 14 5,5 14 6 19,4 5 19 12,0 9 1 12,1 5 12 9
17431040:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,0 11 9 1 26,2 9 11 3 26,0 1 13 12 26,2 3 15 14 26,14 15 7 6 26,1 9 13 26,13 9 14 26,2 14 9 26,4 12 16 26,13 16 12 26,13 5 16 26,6 7 19 26,6 19 16 26,5 6 16 26,4 16 19 26,13 14 6 5 26
35618560:3 10 20 11 7 18 25 19,10 20 25 18 14,2 9 20 10 14,25 17 6 18 14,14 25 17 20,0 11 20 8 12,11 19 25 20 12,4 16 25 19 12,25 12 16 20,25 16 13 20,17 25 13 20,5 17 25 16 13,1 8 20 9 13,8 13 16 12 20,13 9 14 17 20
17562624:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,3 10 11 26,3 15 10 26,10 15 14 26,2 10 14 26,0 11 1 26,1 11 10 26,1 10 2 26,0 1 13 12 26,1 2 14 13 26,14 15 7 6 26,6 7 19 17 26,6 17 14 26,13 14 17 26,12 16 4 26,4 16 19 26,16 17 19 26,12 13 16 26,13 17 16 26,5 13 17 16
983552:3 18 19 7,2 3 18 9,0 19 3 9,9 18 19 3,4 16 19 0,9 19 16 0,0 9 1 16,16 17 18 19 9,1 9 17 5 16,2 6 17 9 18
154624:3 7 10 11,1 2 14 10,0 11 1 12,7 14 10 11,11 7 14 17,6 7 17 14,1 11 10 14,1 14 17 11,7 11 12 17,1 17 12 11,4 17 7 12,4 5 17 12,1 5 12 17
17496832:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,2 3 15 14 26,14 15 7 6 26,2 9 11 3 26,4 5 17 19 26,6 7 19 17 26,0 11 8 26,0 8 12 26,8 11 9 26,12 8 13 26,8 9 13 26,1 8 9 13,12 13 5 4 26,5 13 17 26,9 2 14 26,14 6 17 26,13 9 14 17 26
1605376:11 19 20 3 7 10,10 20 19 7 14,6 7 19 14,2 9 20 10 14,0 11 20 8 12,11 12 19 20,12 8 13 20,1 8 20 9 13,19 12 13 20,12 13 5 4 19,13 14 19 20,13 9 14 20,13 14 6 5 19
854272:0 8 11 4 16 19,11 19 7 3 26,3 7 18 26,18 7 19 26,2 11 3 26,2 3 18 26,2 18 6 26,8 16 19 11 26,2 8 11 26,16 18 19 26,1 8 2 26,8 1 5 16 26,5 6 18 16 26,1 2 6 5 26
17677312:24 19 7 15 18,11 24 15 3 10,12 4 19 24 16,0 1 11 12 13 24,1 13 24 11 10,16 18 19 24,12 13 16 24,10 15 18 24,10 24 18 13,13 18 16 24,1 10 2 13,2 10 18 6 13,5 6 18 16 13
827904:2 9 3 15,0 3 9 15,0 9 1 13,0 9 13 15,2 15 18 9,6 2 18 9,15 7 18 19,0 19 15 13,13 18 19 15,13 9 18 15,4 19 0 13,4 5 19 13,5 18 19 13,5 13 6 18,13 9 6 18
17684992:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,0 11 9 1 26,2 9 11 3 26,2 3 15 14 26,7 18 15 26,14 15 18 26,6 14 18 26,0 1 12 26,12 16 4 26,1 16 12 26,4 16 19 26,16 18 19 26,18 7 19 26,1 5 16 26,1 9 5 26,9 2 14 26,5 6 18 16 26,5 14 6 26,5 9 14 26
18743040:11 20 24 0 8 12,3 10 15 11 20 24,8 12 24 20 26,20 24 15 10 26,1 8 20 9 26,2 9 20 10 26,12 4 19 24 26,24 19 7 15 26,4 12 16 26,4 16 19 26,16 18 19 26,7 19 18 26,15 7 18 26,12 8 13 16 26,14 10 15 18 26,1 13 8 26,1 9 13 26,9 14 13 26,9 2 14 26,2 10 14 26,13 14 18 16 26,5 13 16 6 14 18
17676800:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,1 0 11 9 26,9 11 3 2 26,0 1 13 12 26,2 3 15 26,15 7 18 26,2 15 18 26,13 5 16 26,12 13 16 26,12 16 4 26,4 16 19 26,16 18 19 26,7 19 18 26,2 18 6 26,2 6 9 26,1 9 13 26,9 6 13 26,5 13 6 26,5 6 18 16 26
216064:10 3 7 11,1 10 2 14,10 7 14 17,6 7 17 14,4 16 7 11,4 11 0 16,0 11 1 16,11 10 17 16 7,11 16 17 10 1,1 14 17 10,1 17 5 16
1945344:1 8 20 9 26,9 20 10 2 26,20 11 3 10 26,8 0 11 20 26,3 11 15 26,10 3 15 26,11 19 15 26,0 4 19 11 26,15 7 18 19,15 19 18 26,10 15 18 26,2 10 18 6 26,4 16 19 26,0 8 16 4 26,16 18 19 26,1 13 8 26,8 13 16 26,13 5 16 26,5 6 18 16 26,1 9 13 26,9 2 6 26,13 9 6 26,5 13 6 26
337408:0 4 7 3 26,0 3 10 26,10 3 7 18 26,0 16 4 26,16 18 7 4 26,0 10 9 26,0 9 1 13,9 13 0 26,0 13 16 26,9 10 2 26,13 5 16 26,2 10 18 6 26,9 2 6 26,9 6 13 26,5 13 6 26,5 6 18 16 26
271616:0 4 7 3 26,0 3 10 8 26,1 8 10 2 26,2 10 18 6 26,10 3 7 18 26,0 8 4 26,4 8 13 26,4 13 5 26,1 13 8 26,4 18 7 26,4 5 18 26,5 6 18 26,1 2 13 26,2 6 13 26,5 13 6 26
2035456:1 8 20 9 26,9 20 10 2 26,20 11 3 10 26,8 0 11 20 26,3 11 19 7 26,0 4 19 11 26,2 10 18 6 26,10 3 7 18 26,1 5 16 8 26,0 8 16 4 26,4 16 19 26,7 19 18 26,5 17 16 26,6 18 17 26,16 17 18 19 26,1 9 17 5 26,9 2 6 17 26
112640:0 1 13 12 11,2 3 15 11,1 11 2 13,2 11 15 13,11 12 15 13,2 15 6 13,12 4 7 15 16,7 6 15 16,12 13 16 15,16 13 6 15,16 6 13 5
330752:3 10 11 26,1 11 10 26,3 11 7 26,4 7 11 26,1 10 2 26,2 10 18 6 26,10 3 7 18 26,0 4 11 26,0 11 1 26,0 16 4 26,0 1 16 26,1 5 16 26,16 18 7 4 26,5 6 18 16 26,1 2 6 5 26
869376:3 7 18 10 19,0 19 3 10,1 10 2 14,0 16 4 19,0 16 19 10,1 16 0 10,16 18 19 10,10 18 14 16,1 10 14 16,1 5 16 14,5 6 18 16 14
3126528:0 12 15 3 26,0 3 10 8 26,10 3 15 26,1 8 10 2 26,7 19 18 15,10 15 18 26,12 19 15 26,15 19 18 26,4 19 12 26,0 8 21 12 26,1 13 21 8 26,13 5 16 21 26,21 16 4 12 26,4 16 19 26,2 10 18 6 26,1 2 13 26,17 6 18 26,2 6 17 26,2 17 13 26,5 13 17 26,5 17 16 26,16 17 18 19 26
10485504:20 10 23 9 2 14,11 3 15 20 10 23,20 23 14 9 26,11 15 23 20 26,12 19 15 11 26,4 12 16 19,12 16 19 26,15 19 7 26,0 12 11 26,1 8 20 9 26,8 0 11 20 26,0 8 12 26,1 13 8 26,7 18 23 15 26,6 14 23 18 26,7 19 18 26,6 18 17 26,14 6 17 26,1 9 13 26,8 13 16 12 26,16 17 18 19 26,13 9 14 17 26,13 17 16 26,13 16 17 5
815872:0 3 8 12,2 9 3 14,1 8 9 13,3 9 8 13,12 8 13 3,13 9 14 3,3 12 19 13,3 7 18 19,3 19 18 13,3 18 14 13,12 13 5 4 19,5 18 19 13,13 14 6 5 18
623104:2 9 3 15,0 3 9 15,6 2 15 9,6 15 7 19,0 9 6 15,0 19 15 6,0 4 19 16,6 19 16 0,0 9 16 6,0 9 1 16,5 6 16 9,1 9 5 16
632320:0 10 15 3,2 6 9 10,6 7 19 15,0 16 4 19,0 9 1 13,6 10 15 0,0 10 9 6,0 19 15 6,9 6 13 0,6 19 16 0,0 13 16 6,5 13 6 16
17431296:0 12 24 11 26,11 24 15 3 26,24 19 7 15 26,12 4 19 24 26,0 11 8 26,8 11 9 26,8 9 13 26,1 8 9 13,0 8 12 26,2 9 11 3 26,2 3 15 14 26,14 15 7 6 26,13 9 14 26,2 14 9 26,6 7 19 26,4 12 16 26,4 16 19 26,6 19 16 26,8 13 16 12 26,5 16 13 26,5 6 16 26,13 14 6 5 26
17628160:0 1 11 12 13 24,12 13 24 4 5 19,7 15 24 19 18,3 11 24 15 10,1 13 24 11 10,10 15 18 14 24,24 13 14 10,24 14 13 18,1 2 14 13 10,5 19 24 13 18,13 14 6 5 18
442624:0 3 8 15,2 3 15 14 8,1 8 2 14,0 8 4 15,4 18 7 15,14 15 18 17,4 17 18 15,14 17 18 6,8 15 14 17,8 15 17 4,4 5 17 8,1 14 17 8,1 17 5 8
1400576:3 7 11 10 18 20,0 11 20 8 12,11 7 18 20 12,4 16 18 7 12,12 16 18 20,8 16 12 20,10 18 14 20,9 20 10 2 14,14 18 16 20,5 6 18 16 14,9 5 16 20 14,8 20 16 1 9 5
1867520:3 11 19 7 26,11 12 19 26,0 11 20 8 12,3 10 20 11 26,3 7 18 10 26,7 19 18 26,11 20 12 26,8 13 12 20,12 20 13 26,8 20 9 1 13,4 19 12 26,12 13 5 4 26,4 5 19 26,5 18 19 26,10 18 14 26,14 18 6 26,5 6 18 26,20 10 14 26,2 9 20 10 14,20 13 14 9,20 14 13 26,13 14 6 5 26
267008:0 3 8 12,3 12 7 18,4 18 7 12,4 5 18 12,8 3 18 5 12,2 9 3 18,9 2 6 18,8 5 18 3 9,5 6 18 9,1 9 5 8
1670912:3 10 7 11 20 19,0 11 20 8 12,11 12 19 20,4 16 19 12,12 16 19 20,8 13 16 12 20,1 8 20 9 13,10 20 19 7 14,14 19 16 20,6 7 19 14,6 19 16 14,2 9 20 10 14,9 14 13 20,13 14 16 20,13 14 6 5 16
2084608:3 11 15 26,10 3 15 26,11 19 15 26,0 4 19 11 26,1 8 20 9 26,9 20 10 2 26,20 11 3 10 26,0 11 20 8 26,10 15 18 14 26,18 7 19 15,15 19 18 26,2 10 14 26,2 14 9 26,6 18 17 14,14 18 17 26,9 14 17 26,0 8 16 4 26,1 5 16 8 26,4 16 19 26,16 17 18 19 26,5 17 16 26,1 9 17 5 26
17628928:11 24 15 3 10,0 12 24 11 8,8 11 10 9 24,9 2 14 10,7 15 24 19 18,10 15 18 14 24,9 10 14 24,8 9 13 24,1 9 13 8,13 9 14 24,12 8 13 24,13 14 18 24,12 13 24 4 5 19,5 19 24 13 18,13 14 6 5 18
138496:7 11 10 3,8 0 11 12,6 10 7 17,11 7 10 17,7 11 12 17,4 7 12 17,4 5 17 12,8 11 10 17,8 11 17 12,5 12 8 17,2 6 17 10,1 17 5 8,1 8 10 2 17
615936:2 9 10 14,0 9 1 13,3 7 10 19,4 19 0 16,0 19 3 10,7 14 10 19,6 14 7 19,9 10 19 16 0,9 16 19 10 14,6 19 16 14,0 13 16 9,13 9 14 16,5 13 14 6 16
17677056:15 3 2 24 11 9,0 12 24 11 8,9 11 24 8,8 24 9 13,8 9 1 13,4 19 24 12 16,7 15 24 19 18,16 18 19 24,8 13 16 12 24,9 24 15 2 18,6 2 18 9,13 9 18 24,16 13 18 24,9 6 13 18,5 6 18 16 13
254976:0 3 10 15,7 6 14 15 17,0 16 4 15,4 16 7 15,7 16 17 15,0 10 1 13,13 10 15 16 0,14 10 15 17,13 16 15 10 17,5 17 16 13,13 14 17 10,1 2 14 13 10
121856:0 11 1 12,3 10 11 15,1 10 2 14,1 11 10 15,12 15 11 1,14 10 15 1,12 4 7 15 16,1 16 12 15,14 15 16 1,1 5 16 14,14 15 7 6 16,5 6 16 14
17824768:24 13 12 11 1 0,4 19 24 12 16,24 19 7 15 18,11 24 15 3 10,1 13 24 11 10,12 13 16 24,14 10 15 18 24,24 13 14 10,1 2 14 13 10,24 18 14 17,13 14 17 24,16 13 17 24,16 17 18 19 24,6 18 17 14,16 5 17 13
804608:3 7 18 10 19,0 3 10 8 19,0 8 4 19,4 5 19 8,8 19 10 5,5 18 19 10,10 18 14 5,8 10 9 5,1 8 9 5,9 10 2 14,9 10 14 5,5 6 18 14
34576128:4 16 25 19 12,7 19 25 18 15,12 15 25 19,12 8 0 15 10 3,12 15 10 8 26,12 15 26 25,8 16 12 26,12 25 26 16,10 15 18 26,18 25 26 15,8 10 9 26,1 8 9 26,9 10 2 26,5 17 25 16 26,17 6 18 25 26,8 1 5 16 26,1 9 17 5 26,9 2 6 17 26,2 10 18 6 26
742400:1 10 2 14,0 3 10 12,0 10 1 12,4 16 19 12,3 19 7 10,12 19 3 10,7 14 10 19,6 7 19 17 14,12 10 14 16 19,12 16 14 10 1,16 17 19 14,1 14 17 16,1 17 5 16
970240:3 10 11 15,7 19 18 15,2 9 10 14,11 19 15 10,10 15 18 14 19,19 17 18 14,19 10 17 14,17 9 14 10,6 18 17 14,11 9 17 19 10,4 0 1 5 19 11 9 17
445184:14 2 9 15 3 11,7 4 18 15,15 18 14 17,4 17 18 15,6 18 17 14,11 15 14 9 17,8 11 9 17,4 15 11 17,0 11 8 4,4 11 8 17,5 17 4 8,1 9 17 5 8
17669888:0 12 24 11 8,11 24 15 3 10,12 4 19 24 16,24 19 7 15 18,8 16 12 24,19 16 18 24,10 15 18 24,8 11 10 9 24,8 9 16 24,16 9 18 24,9 10 18 24,8 1 5 16 9,6 18 16 5 9,10 18 6 2 9
579328:12 8 0 15 10 3,12 15 10 8 19,15 7 6 14 19,15 14 10 19,5 12 8 19,5 19 4 12,8 19 10 5,19 14 10 5,5 6 19 14,2 9 10 14,8 10 9 5,1 8 9 5,5 9 14 10
660480:0 3 10 12,3 19 7 10,3 12 19 10,17 6 7 19 10,10 6 2 17,19 10 12 17,5 17 19 4 12,0 10 1 12,12 10 1 17,1 10 2 17,12 1 5 17
575232:15 7 6 14 19,0 3 10 8 15,0 19 15 8,0 8 4 19,9 10 2 14,10 15 14 8,8 10 9 14,14 19 8 15,4 5 19 8,8 19 14 5,5 6 19 14,8 14 9 5,1 8 9 5
397824:0 3 9 12,0 9 1 12,7 3 12 18,12 4 7 18,3 18 2 9,12 9 18 3,12 18 9 17,17 18 4 12,5 17 4 12,1 9 17 5 12,9 2 6 17 18
1000960:3 7 18 10 19,19 4 16 0,0 3 10 19,19 10 0 18,9 10 18 17 0,19 16 17 18 0,0 17 9 16,1 16 0 9,1 9 17 5 16,18 10 9 17 14,17 6 18 14,9 10 2 14
1035264:7 19 18 15,3 10 11 15,0 4 19 11 16,15 18 14 10 11,11 18 19 15,14 18 17 11,6 18 17 14,19 16 17 18 11,0 11 1 16,1 10 2 14,1 11 10 14,1 14 17 11,1 17 16 11,5 17 16 1
644096:0 12 15 3 10,1 10 2 14,0 10 1 12,15 12 14 10,1 14 12 10,15 7 6 14 19,19 4 16 12,12 15 14 19,19 14 12 16,16 6 19 14,1 16 12 14,1 5 16 14,6 16 5 14
1025536:0 3 10 15,18 7 19 15,19 4 16 0,18 15 19 0,18 10 15 0,10 18 17 9 0,19 16 17 18 0,0 9 1 13,0 9 13 17,0 13 16 17,16 5 17 13,9 2 10 17 6 18
2604800:11 12 15 26,3 10 11 15,15 10 11 26,10 15 18 26,7 18 15 26,4 7 15 12 26,8 11 10 9 26,0 12 11 26,0 11 8 26,12 0 8 21 26,4 12 21 16 26,16 18 7 4 26,18 16 17 26,9 10 18 17 26,1 8 9 26,1 9 13 26,9 17 13 26,5 13 17 26,5 17 16 26,1 13 21 8 26,13 5 16 21 26,9 2 10 17 6 18
1761024:11 3 10 20 15,15 14 10 20,20 10 2 9 14,0 11 20 8 12,19 15 11 12 20,17 6 14 19 7 15,15 19 17 14 20,9 14 17 20,12 17 19 20,5 17 19 4 12,8 20 17 5 12,1 8 5 9 20 17
1026048:4 19 11 0 16,7 15 19 18,0 13 16 11,0 11 1 13,2 11 3 15,1 11 2 13,11 19 15 16,2 15 18 17,19 16 17 18 15,6 18 17 2,15 17 13 11 16,17 15 11 13 2,13 17 5 16
17538560:11 24 15 3 10,24 19 17 15 7 6,17 6 15 24 10,9 11 10 24,10 17 9 24,9 2 6 17 10,4 19 24 12 16,17 19 16 24,12 1 0 24 9 11,12 24 9 1 16,9 24 17 16,1 9 17 5 16
568576:4 19 11 0 8,3 10 11 15,7 19 6 15,8 11 10 13,19 11 8 13,11 15 10 13,11 19 15 13,8 13 4 19,5 19 4 13,1 8 10 2 13,15 6 10 13,10 6 2 13,15 6 13 19,19 6 13 5
708096:3 10 11 15,2 9 10 14,11 9 14 15 10,17 6 14 19 7 15,17 14 9 19 15 11,0 11 9 1 4 19 17 5
17799936:0 12 24 11 8,12 4 19 24 16,24 19 7 15 18,24 9 11 15 2 3,8 11 9 24,8 16 12 24,9 24 15 2 18,24 18 9 17,9 2 6 17 18,19 16 17 18 24,8 9 17 16 24,1 9 8 5 17 16
852736:0 3 8 19,19 18 7 3,4 0 8 16 19,9 3 2 18,2 18 6 9,9 18 19 8 3,8 19 18 9 16,16 5 6 18 9,16 8 1 5 9
3012352:0 12 15 3 26,0 3 10 8 26,12 15 26 19,3 15 10 26,19 18 7 15,15 18 14 10 26,15 19 18 26,18 6 14 26,4 19 12 26,16 19 4 26,16 18 19 26,12 21 16 4 26,0 8 21 12 26,8 10 9 26,9 10 2 14,9 10 14 26,1 8 9 26,8 1 13 21 26,21 13 5 16 26,1 9 13 26,13 9 14 26,5 6 18 16 26,13 14 6 5 26
3127040:0 12 15 3 26,0 3 10 8 26,3 15 10 26,0 8 21 12 26,12 19 15 26,19 18 7 15,10 15 18 26,19 26 18 15,4 19 12 26,12 21 16 4 26,16 19 4 26,8 10 9 26,1 8 9 26,8 1 13 21 26,21 13 5 16 26,1 9 13 26,9 10 18 17 26,9 17 13 26,9 2 10 17 6 18,19 16 17 18 26,5 17 16 26,13 17 5 26
305664:0 11 9 1 13,11 3 2 9 15,11 9 13 15,0 13 4 11,11 13 4 15,4 18 7 15,15 18 2 9,18 6 2 9,15 9 13 18,18 6 9 13,13 18 4 15,5 18 4 13,5 6 18 13
936448:19 7 3 11 18,9 11 3 2 14,18 14 3 11,11 14 9 18,11 9 17 19 18,9 14 17 18,17 6 18 14,0 11 9 1 4 19 17 5
637952:0 1 13 12 11,11 3 10 15,19 6 7 15,16 19 4 12,11 10 1 13,1 10 2 13,11 10 13 15,15 10 13 6,10 2 13 6,19 15 11 12 13,13 6 19 15,12 13 16 19,16 6 19 13,13 5 16 6
723968:19 7 3 11 10,4 19 11 0 16,19 17 6 7 10,0 11 1 16,11 10 17 16 19,10 11 16 17 1,5 17 16 1,10 6 2 17,1 10 2 17
1020928:12 15 3 0 10,0 10 1 12,4 16 19 12,19 18 7 15,12 19 15 16,10 15 18 17,19 16 17 18 15,18 6 2 10 17,15 17 12 16,12 15 10 17,12 10 1 17,12 17 1 16,5 17 16 1,1 10 2 17
361984:0 3 10 15,4 15 0 16,4 16 18 7 15,15 10 0 16,0 10 9 16,0 9 1 16,10 15 18 16,16 10 9 18,18 6 2 10 9,6 18 16 5 9,16 1 5 9
843008:0 11 8 12,3 10 11 15,18 7 19 15,8 11 10 15,12 19 15 11 8,15 18 14 10 8,15 18 8 19,1 8 10 2 14,12 19 8 5,5 19 4 12,5 18 19 8,18 8 5 14,1 5 8 14,5 6 18 14
698624:2 11 3 15,17 6 7 19 15,2 15 6 17,2 8 11 13,1 8 2 13,0 4 19 11 8,11 8 19 13,4 8 13 19,11 13 17 15 2,13 11 15 17 19,5 17 19 4 13
684800:0 3 8 12,7 14 3 19,19 17 6 7 14,2 9 3 14,12 8 13 3,8 3 9 13,1 8 9 13,19 17 14 3,13 17 19 12 3,9 14 17 13 3,17 13 5 19 12 4
943616:7 18 10 3 19,0 3 10 19,0 9 1 13,2 9 10 14,0 10 9 13,0 10 13 19,0 13 4 19,10 17 19 18,19 17 10 13,10 18 14 17,17 6 18 14,5 17 19 4 13,9 14 17 13 10
3129088:18 7 19 15,3 10 11 15,0 8 21 12 11,12 19 15 11 21,12 21 16 4 19,19 16 17 18 21,11 15 10 21,10 15 18 21,8 11 10 9 21,9 10 18 17 21,8 1 13 21 9,21 13 5 16 17,9 17 13 21,9 2 10 17 6 18,15 19 18 21
3143424:0 12 15 3 26,0 3 10 8 26,15 10 3 26,0 8 21 12 26,8 10 9 26,8 9 1 26,8 1 13 21 26,1 9 13 26,9 10 2 14,9 10 14 26,12 19 15 26,19 12 4 26,12 21 16 4 26,21 13 5 16 26,19 18 7 15,19 18 15 26,16 19 4 26,15 18 14 10 26,18 17 6 14,14 18 17 26,19 16 17 18 26,13 9 14 17 26,13 17 5 26,5 17 16 26
544000:4 19 11 0 8,19 7 3 11 10,8 11 10 19,19 8 4 5,7 10 19 14,7 6 14 19,10 14 8 19,1 8 10 2 14,19 14 8 5,8 14 1 5,5 6 19 14
697344:0 3 10 15,7 19 17 6 15,0 10 1 13,1 10 2 13,0 15 10 19,10 15 6 17,15 17 10 19,10 6 2 17,0 10 13 19,0 4 19 13,19 17 10 13,10 17 2 13,5 17 19 4 13
112128:4 7 15 12 16,0 12 15 3 10,0 10 9 12,0 1 13 12 9,6 7 16 15,15 6 12 16,15 6 10 12,9 10 2 6,12 10 9 6,12 9 13 6,12 13 16 6,16 13 5 6
17678080:0 12 24 11 8,11 24 15 3 10,12 4 19 24 16,24 19 7 15 18,16 18 19 24,10 15 18 24,8 11 10 9 24,24 10 9 18,10 18 6 2 9,13 24 8 9,1 8 9 13,24 9 13 18,13 9 6 18,13 18 16 24,12 8 13 16 24,6 18 16 5 13
176384:0 12 15 3 8,2 8 3 15,8 15 2 13,12 8 13 15,1 8 2 13,17 6 7 15,7 15 12 4 17,2 15 6 17,15 2 13 17,12 15 13 17,12 13 5 4 17
317952:0 11 9 1 12,14 2 9 15 3 11,11 15 14 9 12,7 15 12 4 18,12 15 14 18,12 5 4 18,18 14 12 5,9 14 5 12,1 9 5 12,14 18 6 5
181504:0 3 10 8 15,1 8 10 2 14,14 10 15 8,7 6 14 15 17,4 17 7 15,0 8 4 15,15 17 14 8,1 14 17 8,8 17 4 15,8 17 1 5,5 4 8 17
794368:0 11 8 12,12 11 8 19,12 19 8 5,5 19 4 12,10 3 11 18 7 19,8 11 19 9 10 18,19 18 9 8 5,1 8 9 5,10 18 6 2 9,5 6 18 9
681472:7 3 19 10,3 10 0 19,0 10 9 19,7 10 19 14,9 10 2 14,9 10 14 19,17 6 7 19 14,0 9 13 19,0 9 1 13,4 19 0 13,13 9 14 17 19,5 17 19 4 13
612608:11 3 2 14,2 8 11 14,1 8 2 14,19 7 3 11 14,7 19 6 14,0 11 8 12,16 19 4 12,8 11 19 16 12,19 11 8 16 14,19 16 6 14,16 5 6 14,16 8 1 5 14
164096:0 3 8 15,3 2 8 15,4 15 0 8,17 6 7 15,7 4 17 15,2 15 6 17,8 2 17 15,8 17 4 15,4 5 17 8,8 2 1 17,8 17 1 5
162304:1 13 9 0 12 11,11 3 2 9 14,3 14 11 7,11 14 17 12 7,17 13 12 14 9 11,7 17 6 14,7 12 4 17,12 13 5 4 17
830976:0 11 9 1 13,9 11 10 13,19 11 0 4 13,4 19 13 5,10 11 3 15,19 18 7 15,11 10 18 19 15,19 18 10 11 13,5 18 19 13,13 10 9 18,18 6 2 10 9,6 13 9 18,6 18 5 13
111360:15 3 0 12 8,9 8 3 15,9 3 2 15,8 15 9 13,1 8 9 13,15 6 2 9,15 9 13 6,7 15 12 4 16,12 8 13 16 15,16 6 7 15,13 6 16 15,16 6 13 5
699392:0 11 1 13,1 11 10 13,1 10 2 13,19 11 0 4 13,11 3 10 15,17 6 7 19 15,11 19 15 13,11 15 10 13,13 17 19 15,6 10 15 17,15 10 13 17,2 10 6 17,2 17 13 10,5 17 19 4 13
906752:12 15 3 0 10,9 10 2 14,0 9 1 12,0 10 9 12,16 19 4 12,19 18 7 15,12 15 18 16 19,10 15 12 14 18 16,12 16 14 10 9,1 16 12 9,5 9 14 16,1 5 16 9,5 6 18 16 14
345344:5 6 18 16 14,16 8 1 5 14,8 10 2 1 14,18 10 8 16 14,0 3 10 8 4 7 18 16
967424:4 19 0 8,3 8 0 15,0 19 15 8,19 18 7 15,3 15 14 2 9,8 3 9 15,19 15 8 9,15 14 17 19 18,15 19 17 14 9,18 17 6 14,19 8 17 9,5 17 19 4 8,1 9 17 5 8
385536:15 14 2 3 9,0 3 9 15,0 15 9 13,0 9 1 13,13 14 15 9,4 15 0 16,0 13 16 15,4 16 18 7 15,13 14 18 16 15,16 5 13 18 6 14
454400:0 11 8 4,4 18 7 15,9 10 2 14,11 3 10 15,1 8 9 13,8 11 10 9 13,15 18 14 10 17,18 17 6 14,4 8 13 11,4 17 18 15,15 17 13 11 4,17 15 11 13 10,5 17 4 13,13 9 14 17 10
1769216:0 11 20 8 12,1 8 20 9 13,9 20 10 2 14,20 11 3 10 15,13 12 8 20,15 14 10 20,12 19 15 11 20,19 17 14 15 20,12 13 17 19 20,13 9 14 17 20,13 5 17 12 4 19,17 6 14 19 7 15
754432:0 3 8 15,8 3 9 15,9 3 2 15,19 15 0 8,4 0 8 16 19,7 19 17 6 15,2 6 17 9 15,16 19 8 15,19 16 17 15,8 9 17 16 15,8 1 9 16 5 17
723712:11 0 8 19 4 16,11 8 9 19 16 17,8 1 9 16 5 17,9 11 3 2 17 19 7 6
552448:0 11 9 1 13,9 10 2 14,7 3 11 19 10,19 11 0 4 13,11 10 9 19,13 11 9 19,19 7 10 14,19 7 14 6,9 14 19 10,14 13 9 19,5 19 4 13,13 14 6 5 19
169216:17 6 7 15,6 10 15 17,6 2 10 17,7 15 12 4 17,17 4 5 12,8 0 12 10 3 15,12 15 10 8 17,5 12 8 17,1 8 10 2 17,5 8 1 17
759296:0 9 1 12,0 10 9 12,15 3 0 12 10,7 19 17 6 15,19 15 12 17,16 19 4 12,17 19 16 12,6 10 15 17,10 17 12 15,12 10 9 17,2 6 17 9 10,12 17 9 16,1 16 12 9,9 17 5 1 16
124672:0 3 10 8 15,0 8 16 4 15,4 16 7 15,8 15 10 16,14 10 15 16,6 14 15 7 16,8 10 14 13 16,14 6 5 13 16,1 8 9 13,9 10 2 14,8 13 14 10 9
957696:0 11 8 12,11 3 10 15,18 7 19 15,10 15 18 17,17 18 19 15,11 15 10 17,10 18 6 2 17,8 11 10 17,19 15 11 12 17,11 17 8 12,1 8 10 2 17,5 17 19 4 12,5 12 8 17,8 1 5 17
661248:0 3 10 8 12,3 19 7 10,3 12 19 10,7 19 17 6 10,12 17 19 10,8 17 12 10,5 17 19 4 12,5 12 8 17,8 10 9 17,1 9 17 5 8,9 2 6 17 10
988672:0 9 1 12,0 10 9 12,0 3 10 12,12 19 3 10,7 18 10 3 19,16 19 4 12,16 9 10 19 12,12 9 1 16,10 18 19 9 17 16,9 2 10 17 6 18,1 9 17 5 16
774912:0 12 15 3 8,3 9 8 15,3 15 14 2 9,15 7 19 14 6 17,12 15 8 19,8 15 9 19,15 19 17 14 9,16 19 4 12,8 16 12 19,8 9 17 16 19,1 9 8 5 17 16
259072:15 3 0 12 10,1 13 12 0 10,15 12 13 14 10,2 14 13 1 10,15 7 6 14 17,7 15 12 4 16,17 7 16 15,16 15 12 17,12 15 14 13 17,12 13 16 17,16 5 17 13
941568:19 7 18 11 3 10,19 18 17 11 10 9,9 10 2 14,17 6 18 14,17 18 10 9 14,5 17 19 4 12,1 9 17 5 12,0 11 9 1 12,11 19 17 9 12
896512:10 11 3 15,18 7 19 15,10 18 19 11 15,4 19 11 0 16,0 11 9 1 13,0 13 16 11,11 10 9 13,16 11 13 19,11 19 18 10 13,16 18 19 13,13 10 9 18,2 10 18 6 9,13 9 6 18,16 5 6 18 13
834816:0 11 8 12,10 11 3 15,18 7 19 15,10 8 11 13,1 8 10 2 13,12 11 8 13,11 15 10 13,18 10 15 13,10 18 6 2 13,12 19 15 11 13,13 18 19 15,19 18 13 5,6 18 5 13,13 5 4 12 19
431360:8 0 12 10 3 15,7 15 12 4 18,18 10 15 17,12 15 17 18,4 17 18 12,12 15 10 8 17,10 18 6 2 17,1 8 10 2 17,5 17 4 12,12 8 5 17,8 1 5 17
739072:3 7 10 19,10 7 14 19,7 19 17 6 14,9 10 2 14,17 19 10 14,0 3 10 8 19,0 8 16 4 19,19 10 8 16,17 19 16 10,9 17 16 8 10,17 9 14 10,1 9 8 5 17 16
515584:0 11 9 1 12,10 11 3 15,9 10 2 14,11 15 10 9,12 15 11 9,15 18 14 10 9,12 4 16 15 7 18,12 16 18 15 9,1 16 12 9,18 16 17 9,1 9 17 5 16,17 9 14 18,18 17 6 14
416000:1 8 2 14,8 11 2 14,11 3 2 14,7 12 4 18,7 11 12 18,7 3 11 18,3 14 11 18,11 8 0 12,8 14 18 12 11,14 8 12 18 17,18 17 6 14,17 18 4 12,12 8 5 17,8 1 5 17,1 14 17 8,5 17 4 12
660992:0 9 1 12,0 10 9 12,0 3 10 12,12 19 3 10,19 7 3 10,7 19 17 6 10,19 10 12 17,5 17 19 4 12,10 17 9 12,1 9 17 5 12,2 6 17 9 10
941824:0 11 8 12,10 3 11 18 7 19,19 18 10 11 12,8 11 10 9 12,10 18 17 9 12,19 17 18 12,5 17 19 4 12,17 8 9 12,5 12 8 17,9 17 5 1 8,9 10 2 14,17 6 18 14,18 10 9 17 14
539904:0 11 8 12,10 8 11 13,13 12 8 11,3 11 19 7 10,7 19 6 10,12 11 13 19,13 10 19 11,1 8 10 2 13,10 6 13 19,2 10 6 13,6 19 5 13,13 5 4 12 19
431616:0 9 1 12,0 10 9 12,15 3 0 12 10,7 15 12 4 18,12 15 10 18,9 10 18 17 12,17 18 4 12,5 17 4 12,1 9 17 5 12,9 2 10 17 6 18
874240:0 3 10 8 12,3 7 18 10 19,12 19 3 10,9 10 2 14,16 19 4 12,19 10 12 18,16 18 19 12,10 18 16 8 12,18 10 8 16 9,10 18 14 9,16 18 9 14,16 5 6 18 14,5 9 14 16,1 5 16 8 9
898304:8 0 12 10 3 15,18 7 19 15,16 19 4 12,15 12 19 18,16 18 19 12,12 15 10 8 18,8 10 2 1 13,18 10 8 13,18 6 2 10 13,16 5 6 18 13,12 8 13 16 18
493312:0 3 10 8 15,0 8 16 4 15,4 16 18 7 15,10 18 16 8 15,8 1 9 16 5 17,8 9 10 16 17 18,9 2 10 17 6 18
989696:0 11 9 1 12,19 7 3 11 18,11 3 2 9 18,18 11 19 9,19 11 12 9,16 19 4 12,19 12 16 9,12 9 1 16,9 17 5 1 16,2 6 17 9 18,19 16 17 18 9
749312:11 0 8 19 4 16,7 3 11 19 10,7 14 10 19,9 10 2 14,17 6 7 19 14,9 8 16 10 11 19,19 10 9 16 14,17 19 16 14,16 9 8 13,8 9 1 13,9 14 17 13 16,17 16 5 13
979712:15 3 0 12 8,3 15 14 2 9,3 9 8 15,19 18 7 15,17 6 18 14,19 15 14 17 18,15 12 19 8,14 9 15 8,15 19 17 14 8,9 14 17 13 8,12 13 17 19 8,1 8 9 13,19 4 12 17 5 13
859648:10 3 11 18 7 19,18 10 11 19 9,10 18 6 2 9,0 11 9 1 12,19 11 12 9,19 12 16 9,16 19 4 12,16 18 19 9,1 16 12 9,1 5 16 9,6 18 16 5 9
399872:0 11 9 1 12,7 12 4 18,7 11 12 18,7 3 11 18,11 3 2 9 18,9 18 12 11,18 9 12 17,4 5 17 12,9 17 5 1 12,4 17 18 12,2 6 17 9 18
1020416:15 3 0 12 9,0 9 1 12,3 2 9 15,15 12 19 9,15 18 2 9,18 7 19 15,9 18 19 15,16 19 4 12,12 19 9 16,1 16 12 9,19 16 17 18 9,2 6 17 9 18,1 9 17 5 16
584960:19 11 0 4 8,3 10 11 15,15 7 6 14 19,11 15 10 19,15 14 10 19,10 8 11 19,8 10 14 13 19,10 2 14 8 1 13,4 8 13 19,4 13 5 19,14 6 5 13 19
961792:0 4 19 11 8,10 11 3 15,18 7 19 15,10 18 19 11 15,18 10 11 19 17,10 18 6 2 17,10 8 11 13,11 10 13 17,11 13 8 19,11 17 13 19,4 8 13 19,8 10 2 1 13,10 2 13 17,5 17 19 4 13
880128:10 3 11 18 7 19,19 11 0 4 16,0 11 9 1 13,16 0 13 11,19 18 10 11 16,9 11 10 13,11 10 13 16,13 14 18 16 10,9 10 2 14,10 14 9 13,16 5 13 18 6 14
858880:11 8 0 12,16 19 4 12,19 7 3 11 18,11 3 2 9 18,18 6 2 9,18 11 19 9,8 11 19 16 12,19 11 8 16 9,16 18 19 9,6 18 16 5 9,1 5 16 8 9
438784:3 2 9 15,15 3 0 12 9,0 1 13 12 9,15 18 2 9,7 15 12 4 18,15 9 12 18,12 18 9 17,18 4 17 12,12 9 13 17,2 6 17 9 18,13 5 4 12 17
745728:19 7 3 14,17 6 7 19 14,3 2 8 14,19 3 0 8,8 14 19 3,14 8 19 17,16 4 0 8 19,17 19 16 8,17 14 8 13,16 8 13 17,16 5 17 13,1 2 14 13 8
355072:7 18 3 11,18 7 4 16 11,8 16 4 0 11,9 11 3 2 14,3 18 14 11,11 9 8 13,16 8 13 11,14 13 9 11,8 9 1 13,13 14 18 16 11,16 5 13 18 6 14
673024:19 7 3 10,0 3 10 8 19,4 19 0 8,19 7 10 14,7 19 17 6 14,19 10 8 14,1 8 10 2 14,19 17 14 8,5 17 19 4 8,14 17 1 8,17 5 1 8
22019840:24 11 0 12 8,19 24 12 4 16,7 15 24 19 18,15 3 11 24 10,8 11 10 9 24,10 15 18 14 24,24 10 22 14,24 10 9 22,2 14 22 9 10,24 14 22 18,24 18 22 17,22 14 6 17 18,19 16 17 18 24,24 9 8 22,24 22 8 13,9 22 13 1 8,13 24 22 16,24 17 22 16,22 17 5 13 16,12 8 13 16 24
184832:0 9 1 12,15 3 0 12 9,3 15 14 2 9,5 17 4 12,7 15 12 4 17,7 6 14 15 17,17 9 14 15,12 15 9 17,9 17 5 1 12
369920:0 3 10 8 15,0 8 16 4 15,4 16 18 7 15,8 10 18 16 15,16 18 10 8 13,1 8 10 2 13,2 10 18 6 13,6 18 16 5 13
1001216:7 18 10 3 19,16 4 0 8 19,0 3 10 8 19,19 8 16 10,19 16 17 18 10,9 17 16 8 10,8 1 9 16 5 17,17 18 10 14,10 14 9 17,9 10 2 14,17 6 18 14
241152:0 11 9 1 13,10 11 3 15,4 11 0 16,4 15 11 16,4 7 15 16,16 17 7 15,17 6 7 15,0 13 16 11,10 15 6 17,10 17 16 11 15,17 10 11 16 9,16 11 13 9,2 6 17 9 10,16 9 13 17,16 5 17 13
448256:8 0 12 10 3 15,7 15 12 4 18,12 15 17 18,17 18 4 12,5 17 4 12,15 18 14 10 17,12 15 10 8 17,17 6 18 14,9 10 2 14,8 10 9 17,10 14 9 17,12 8 5 17,9 17 5 1 8
980736:8 0 12 10 3 15,15 10 8 12 13,8 10 9 13,1 8 9 13,9 10 2 14,18 7 19 15,10 15 18 14 17,17 6 18 14,19 17 18 15,15 10 13 17,12 13 17 19 15,13 5 17 12 4 19,9 14 17 13 10
749056:0 11 9 1 13,9 10 2 14,7 3 11 19 10,19 11 0 4 16,0 13 16 11,7 14 10 19,7 19 17 6 14,11 10 14 19,9 11 10 14,19 14 11 17,9 14 17 13 11,17 19 16 11,16 13 17 11,16 5 17 13
1045504:15 3 0 12 10,0 1 13 12 10,13 15 12 10,15 18 14 10 13,2 14 13 1 10,18 7 19 15,18 15 19 13,15 12 19 13,16 19 4 12,19 12 16 13,18 14 13 17,17 6 18 14,16 5 17 13,19 16 17 18 13
179200:0 1 13 12 11,11 10 1 13,10 2 1 13,10 11 3 15,10 11 15 13,12 15 11 13,17 6 7 15,7 15 12 4 17,10 15 6 17,15 10 13 17,12 15 13 17,13 5 4 12 17,2 10 6 17,2 17 13 10
739840:4 19 11 0 16,2 9 11 3 14,3 11 19 7 14,17 6 7 19 14,9 17 19 11 14,19 17 9 11 16,0 11 9 1 16,1 9 17 5 16
261888:10 11 3 15,0 11 8 12,8 11 10 9 15,9 10 2 14,15 14 10 9,12 11 8 15,8 9 1 13,8 9 13 15,7 15 12 4 16,15 7 6 14 17,17 7 16 15,12 8 13 16 15,13 9 14 17 15,13 17 16 15,17 13 16 5
687872:3 11 19 7 10,11 8 0 12,7 14 10 19,7 19 17 6 14,9 10 2 14,19 10 11 12,8 11 10 9 12,13 12 8 9,8 9 1 13,14 19 12 9 10,12 13 9 19 17 14,19 4 12 17 5 13
659200:4 19 11 0 8,3 11 19 7 10,8 11 10 9 19,17 6 7 19 10,19 10 9 17,19 17 9 8,2 6 17 9 10,9 17 5 1 8,5 17 19 4 8
944640:0 11 9 1 13,11 3 2 9 14,4 19 11 0 13,19 7 3 11 18,3 18 14 11,19 17 18 11,11 18 14 17,19 17 11 13,13 9 14 17 11,17 6 18 14,5 17 19 4 13
1035008:11 3 15 9 2 14,11 15 14 9 26,11 19 15 26,11 9 8 26,11 0 8 19 4 16,19 11 8 16 26,14 15 18 26,18 7 19 15,18 15 19 26,17 6 18 14,14 18 17 26,17 9 14 26,19 16 17 18 26,9 17 16 8 26,8 1 9 16 5 17
994560:11 0 8 19 4 16,10 3 11 18 7 19,10 8 11 13,18 10 11 19 13,19 11 8 16 13,18 10 13 17,19 16 17 18 13,16 5 17 13,1 8 10 2 13,10 18 6 2 17,17 13 2 10
847616:19 11 0 4 8,8 11 10 9 13,19 11 8 13,8 9 1 13,8 13 4 19,19 4 5 13,10 11 3 15,18 7 19 15,11 15 10 13,11 19 15 13,13 18 19 15,15 18 14 10 13,5 18 19 13,9 10 2 14,14 13 9 10,14 6 5 13 18
1038080:15 3 10 12 0 8,12 15 10 8 26,15 12 19 26,18 7 19 15,16 19 4 12,12 19 26 16,8 16 12 26,10 15 18 14 26,15 18 26 19,8 10 9 26,9 10 2 14,9 10 14 26,26 18 14 17,17 6 18 14,17 9 14 26,19 16 17 18 26,9 17 16 8 26,8 1 9 16 5 17
1032960:3 8 0 15,3 9 8 15,3 15 14 2 9,15 0 19 8,8 16 4 0 19,18 7 19 15,15 14 9 18,18 14 9 17,17 6 18 14,9 18 19 8 15,19 16 8 18 17 9,8 1 9 16 5 17
520704:3 15 14 2 9,15 3 0 12 9,0 1 13 12 9,13 15 12 9,14 15 18 9,15 9 13 18,9 14 17 13 18,17 6 18 14,18 7 15 16 4 12,18 15 12 16 13,18 16 17 13,16 5 17 13
685312:0 3 10 8 12,19 7 3 10,12 19 3 10,7 14 10 19,17 6 7 19 14,12 19 10 14,10 14 13 8 12,10 2 14 8 1 13,19 12 13 17 14,19 4 12 17 5 13
464128:0 3 10 8 12,7 18 10 3 12,4 16 18 7 12,8 10 18 16 12,8 16 18 10 17,1 8 10 2 17,1 5 16 8 17,10 18 6 2 17
481536:11 3 2 14,8 11 2 14,1 8 2 14,7 18 3 11,18 14 3 11,7 11 12 18,18 7 4 16 12,11 8 0 12,18 12 8 14 11,12 18 14 8 17,12 18 17 16,8 16 12 17,8 14 1 17,1 5 16 8 17,17 6 18 14
17825536:0 12 24 11 8,11 24 15 3 10,24 19 7 15 18,12 4 19 24 16,8 11 10 9 24,10 15 18 14 24,8 9 1 13,8 9 13 24,9 10 2 14,9 10 14 24,17 6 18 14,14 18 17 24,13 17 16 24,17 13 16 5,12 8 13 16 24,9 14 17 13 24,19 16 17 18 24
871168:11 0 8 19 4 16,19 7 3 11 18,11 3 2 9 14,3 18 14 11,11 18 14 19,11 14 9 19,19 11 8 16 9,19 14 9 16,16 18 19 14,1 5 16 8 9,16 14 9 5,6 18 16 5 14
587264:15 3 0 12 10,0 10 9 12,1 13 12 0 9,12 10 9 15,9 10 2 14,15 14 10 9,15 12 13 14 9,13 5 4 12 19,14 6 5 13 19,15 7 6 14 19,12 15 14 13 19
171520:0 11 9 1 12,10 11 3 15,9 11 10 15,12 15 11 9,17 6 7 15,17 6 15 10,15 10 9 17,2 6 17 9 10,7 15 12 4 17,12 15 9 17,17 4 5 12,1 9 17 5 12
724224:11 0 8 19 4 16,19 7 3 11 10,19 11 8 16 10,17 6 7 19 10,17 19 16 10,16 10 8 17,2 10 6 17,1 8 10 2 17,5 16 8 1 17
430848:0 12 15 3 8,3 9 8 15,3 2 9 15,7 15 12 4 18,15 2 9 18,12 15 17 18,15 18 9 17,15 9 8 17,15 8 12 17,2 6 17 9 18,17 18 4 12,5 17 4 12,12 8 5 17,9 17 5 1 8
912128:11 0 8 19 4 16,11 3 15 9 2 14,11 9 8 13,8 9 1 13,11 15 14 9 13,11 19 15 13,19 11 8 16 13,18 7 19 15,14 15 18 19,15 14 13 19,13 14 18 16 19,14 6 18 13 5 16
639744:3 8 0 15,3 9 8 15,3 15 14 2 9,15 7 6 14 19,15 8 0 19,16 4 0 8 19,15 9 8 19,15 14 9 19,19 9 8 16,19 14 9 16,16 19 14 6,5 16 14 6,5 16 9 14,5 16 8 1 9
876032:0 11 9 1 12,10 3 11 18 7 19,16 19 4 12,19 11 12 16,19 18 10 11 16,9 11 10 16,12 11 9 16,12 9 1 16,10 18 14 16,9 10 2 14,9 10 14 16,16 5 6 18 14,16 14 9 5,16 9 1 5
521472:18 7 15 16 4 12,15 3 10 12 0 8,10 2 14 8 1 13,15 18 14 10 12 16 13 8,16 18 14 13 17,16 5 17 13,17 6 18 14
879616:10 3 11 18 7 19,19 11 0 4 16,0 11 1 13,0 11 13 16,11 19 18 10 16,11 10 1 13,11 10 13 16,13 14 18 16 10,1 2 14 13 10,14 6 18 13 5 16
250624:15 3 0 12 8,3 9 8 15,3 15 14 2 9,4 7 15 12 16,15 17 16 7,15 7 6 14 17,8 16 12 15,17 9 14 15,9 17 16 8 15,8 1 9 16 5 17
769280:10 11 3 15,11 8 0 12,7 19 17 6 15,10 15 6 17,2 10 6 17,15 10 11 17,10 8 11 17,17 10 8 13,8 10 2 1 13,13 10 2 17,19 15 11 12 17,8 17 12 11,12 19 17 16,16 19 4 12,16 5 17 13,16 12 8 13 17
1045760:15 3 10 12 0 8,12 15 10 8 19,18 7 19 15,10 15 18 14 19,10 14 13 8 19,16 19 4 12,16 12 8 13 19,10 2 14 8 1 13,19 13 16 14,19 16 17 18 14,17 6 18 14,16 14 13 17,16 5 17 13
357632:7 3 12 26,12 3 0 26,3 7 18 10 26,0 3 10 8 26,8 12 0 26,10 18 14 26,8 10 14 13 26,10 2 14 8 1 13,7 12 4 26,4 16 18 7 26,12 16 4 26,16 12 8 13 26,13 14 18 16 26,14 6 18 13 5 16
381696:15 3 0 12 8,3 9 8 15,3 15 14 2 9,18 7 15 16 4 12,14 15 18 9,15 9 8 18,16 18 15 12 8,8 18 9 16,16 18 9 14,8 1 5 16 9,16 14 9 5,6 18 16 5 14
617984:0 11 9 1 13,19 7 3 11 10,7 14 10 19,7 6 14 19,11 0 4 19 16,13 16 0 11,16 11 13 19,9 11 10 19,11 9 13 19,9 10 2 14,9 10 14 19,13 9 14 19,13 19 14 16,19 16 6 14,14 6 5 13 16
652544:15 3 10 12 0 8,12 15 10 8 19,16 19 4 12,15 7 6 14 19,15 14 10 19,8 10 14 13 19,10 2 14 8 1 13,16 12 8 13 19,19 14 13 16,19 16 6 14,14 6 5 13 16
551936:0 11 1 13,11 10 1 13,4 19 11 0 13,19 7 3 11 10,11 10 13 19,7 19 6 14,7 14 10 19,13 10 14 19,2 14 13 1 10,5 19 4 13,14 6 5 13 19
584192:0 11 9 1 13,4 19 11 0 13,19 4 5 13,11 3 15 9 2 14,15 7 6 14 19,11 15 14 9 19,11 9 13 19,13 9 14 19,14 6 5 13 19
846592:4 19 11 0 8,11 3 15 9 2 14,18 7 19 15,14 15 18 19,11 15 14 9 19,11 9 8 19,19 9 8 13,8 9 1 13,4 8 13 19,13 9 14 19,19 4 5 13,19 14 13 18,5 18 19 13,14 6 5 13 18
456192:15 3 0 12 10,0 10 9 12,1 13 12 0 9,9 10 2 14,7 15 12 4 18,10 15 18 14 12,12 10 9 14,12 18 14 17,17 18 4 12,17 6 18 14,4 12 13 5 17,13 9 14 17 12
259584:15 3 0 12 10,9 10 2 14,0 10 9 12,12 0 1 13 9,12 10 9 15,15 14 10 9,15 7 6 14 17,7 15 12 4 16,16 17 7 15,12 9 13 15,9 14 17 13 15,12 13 16 15,13 17 16 15,16 5 17 13
917248:11 8 0 12,10 11 3 15,18 7 19 15,16 19 4 12,9 10 2 14,15 10 11 26,8 11 10 9 26,10 15 18 14 26,26 10 9 14,19 15 11 12 26,8 9 1 13,8 9 13 26,19 26 18 15,8 26 12 11,12 19 26 16,16 18 19 26,13 9 14 26,12 8 13 16 26,14 18 16 13 26,14 6 18 13 5 16
523520:11 8 0 12,10 11 3 15,8 10 15 12 11,18 7 15 16 4 12,15 18 14 10 12 16 13 8,16 5 17 13,17 6 18 14,18 14 13 16 17,10 2 14 8 1 13
391680:12 0 11 13 1 9,11 3 15 9 2 14,14 9 13 15 11 12,18 7 15 16 4 12,14 18 15 13 16 12,16 5 13 18 6 14
964352:15 3 10 12 0 8,12 15 10 8 19,8 10 9 19,18 7 19 15,18 10 15 19,10 18 17 9 19,9 2 10 17 6 18,19 9 8 17,8 9 1 13,8 9 13 17,12 13 17 19 8,13 5 17 12 4 19
390400:18 7 15 16 4 12,14 6 18 13 5 16,10 2 14 8 1 13,15 3 10 12 0 8,10 15 18 14 8 12 16 13
10876160:23 14 2 10 21 13 1 8,15 23 10 3 12 21 8 0,18 6 14 23 16 5 13 21,7 18 23 15 4 16 21 12
1048320:11 8 0 12,10 11 3 15,9 10 2 14,18 7 19 15,16 19 4 12,8 11 10 9 15,12 19 15 11 8,10 15 18 14 9,9 18 19 8 15,18 9 8 19 13,8 9 1 13,12 8 13 16 19,13 9 14 17 18,17 6 18 14,19 16 17 18 13,16 5 17 13
"""
_MOT_E2I = (0, 18, 20, 2, 6, 24, 26, 8, 9, 19, 11, 1, 3, 21, 23, 5, 15, 25, 17, 7, 10, 12, 22, 14, 4, 16, 13)
_MOT_VTK = {4: VTK_TET, 5: VTK_PYR, 6: VTK_PRISM, 8: VTK_HEX}


def _mot_setup():
    pats = {}
    for line in _MOT_TABLE.strip().splitlines():
        mask, els = line.split(":")
        pats[int(mask)] = [tuple(int(v) for v in e.split()) for e in els.split(",")]
    grid = np.array([[i % 3, i // 9, (i // 3) % 3] for i in _MOT_E2I])     # internal id i = 9*y + 3*z + x
    gid = {tuple(g): k for k, g in enumerate(grid)}
    rots = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    M = np.zeros((3, 3), int)
                    for i, sgn in zip(range(3), (sx, sy, sz)):
                        M[i, perm[i]] = sgn
                    if round(np.linalg.det(M)) == 1:
                        rots.append(np.array([gid[tuple(M @ (g - 1) + 1)] for g in grid]))
    return pats, gid, rots


MOT_PATTERNS, _MOT_GID, _MOT_ROT = _mot_setup()


def hex_with_hanging_nodes(mesh, c, coff, cfid, angle=35.0):
    """If cell c is a hexahedron whose faces carry hanging nodes (snappy/hexRef8 2:1 transition), return
    {cell point: MixedOcTree id 0-25}, else None. The cell's faces are grouped into 6 big faces (neighbouring faces whose
    normals differ < angle); points on 3 big faces = corners, on 2 = edge mid-nodes, on 1 = face centres."""
    foff, flab, X, own = mesh["foff"], mesh["flab"], mesh["points"], mesh["owner"]
    fids = list(cfid[coff[c]:coff[c + 1]])
    if len(fids) < 6:
        return None
    nrm = []
    for f in fids:
        v = X[flab[foff[f]:foff[f + 1]]]; cc = v.mean(0)
        n = np.cross(v - cc, np.roll(v, -1, 0) - cc).sum(0)
        nrm.append((n if own[f] == c else -n) / (np.linalg.norm(n) + 1e-300))
    par = list(range(len(fids)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    edges = {}
    for i, f in enumerate(fids):
        v = flab[foff[f]:foff[f + 1]].tolist()
        for a, b in zip(v, v[1:] + v[:1]):
            edges.setdefault(frozenset((a, b)), []).append(i)
    cmin = np.cos(np.radians(angle))
    for fs in edges.values():
        if len(fs) == 2 and np.dot(nrm[fs[0]], nrm[fs[1]]) > cmin:
            par[find(fs[0])] = find(fs[1])
    groups = {}
    for i in range(len(fids)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) != 6 and len(fids) == 6:
        # warped thin cells (snapped layers): the normal test fails, but a 6-face cell with hanging nodes only on edges
        # has each face as a big face — topology decides (corner = on 3 faces, edge node = on 2).
        groups = {i: [i] for i in range(6)}
    if len(groups) != 6:
        return None
    memb = {}
    for gi, g in enumerate(groups.values()):
        for i in g:
            for p in flab[foff[fids[i]]:foff[fids[i] + 1]].tolist():
                memb.setdefault(p, set()).add(gi)
    corners = [p for p, s in memb.items() if len(s) == 3]
    if len(corners) != 8 or any(len(s) > 3 for s in memb.values()):
        return None
    adj = {p: [q for q in corners if q != p and len(memb[p] & memb[q]) == 2] for p in corners}
    if any(len(v) != 3 for v in adj.values()):
        return None
    o = corners[0]; a, b, d = adj[o]
    if np.dot(np.cross(X[a] - X[o], X[b] - X[o]), X[d] - X[o]) < 0:   # right-handed frame (x=a, y=b, z=d)
        a, b = b, a
    coord = {o: (0, 0, 0), a: (2, 0, 0), b: (0, 2, 0), d: (0, 0, 2)}
    for q1, q2 in ((a, b), (a, d), (b, d)):
        com = [r for r in adj[q1] if r in adj[q2] and r != o]
        if len(com) != 1 or com[0] in coord:
            return None
        coord[com[0]] = tuple(np.add(coord[q1], coord[q2]))
    rest = [p for p in corners if p not in coord]
    if len(rest) != 1:
        return None
    coord[rest[0]] = (2, 2, 2)
    for p in corners:
        if any(sum(x != y for x, y in zip(coord[p], coord[q])) != 1 for q in adj[p]):
            return None
    lab = {}
    for p, s in memb.items():
        if len(s) == 3:
            g = coord[p]
        elif len(s) == 2:
            ends = [q for q in corners if s <= memb[q]]
            if len(ends) != 2:
                return None
            g = tuple(np.add(coord[ends[0]], coord[ends[1]]) // 2)
        else:
            gi = next(iter(s)); fc = [q for q in corners if gi in memb[q]]
            if len(fc) != 4:
                return None
            g = tuple(np.array([coord[q] for q in fc]).sum(0) // 4)
        if g not in _MOT_GID:
            return None
        lab[p] = _MOT_GID[g]
    return lab if len(set(lab.values())) == len(lab) else None


# ============================================================================================ parallel per-cell workers
# The per-cell / per-face steps are independent → run in forked worker processes that read the big arrays from _S
# (copy-on-write, nothing pickled but the small per-chunk results). Without fork (Windows) → serial, same results.
_S = {}
CEN = -2                                                             # pattern centre point (MixedOcTree id 26)
APX = -3                                                             # centre point of a centre-point split


def _pmap(func, items, nproc):
    """[func(chunk) for chunk in chunks(items)] in parallel (fork) or serially; results in input order."""
    items = list(items)
    if not items:
        return []
    import os
    import multiprocessing as mp
    n = max(1, min(nproc or os.cpu_count() or 1, len(items) // 200 or 1))
    m = n * 8                                                        # ~8 chunks per worker for load balance
    chunks = [ch for ch in (items[k * len(items) // m:(k + 1) * len(items) // m] for k in range(m)) if ch]
    if n == 1 or "fork" not in mp.get_all_start_methods():
        return [func(ch) for ch in chunks]
    with mp.get_context("fork").Pool(n) as pool:
        return pool.map(func, chunks, chunksize=1)


def _cell_pieces(c, fpieces):
    """Outward boundary pieces of cell c (polygon faces → their fixed/DP pieces) + number of cancelled pairs."""
    coff, cfid, csgn, foff, flab = _S["coff"], _S["cfid"], _S["csgn"], _S["foff"], _S["flab"]
    pieces = []
    for j in range(coff[c], coff[c + 1]):
        f, s = int(cfid[j]), csgn[j]
        if f in fpieces:
            for tri in fpieces[f]:
                pieces.append(tri if s > 0 else tri[::-1])
        else:
            v = flab[foff[f]:foff[f + 1]].tolist()
            pieces.append(tuple(v if s > 0 else v[::-1]))
    # pieces that occur twice with opposite orientation (zero-thickness part of a sliver cell, e.g. two pentagons
    # sharing 4 points) cancel: the cells on both sides then meet directly on that piece
    keyc = {}
    for i, p in enumerate(pieces):
        keyc.setdefault(tuple(sorted(p)), []).append(i)
    drop = {i for v in keyc.values() if len(v) == 2 for i in v}
    return [p for i, p in enumerate(pieces) if i not in drop], len(drop) // 2


def _w_triangulate(faces):
    X, foff, flab = _S["X"], _S["foff"], _S["flab"]
    out = []
    for f in faces:
        tris, q = triangulate_polygon(X, flab[foff[f]:foff[f + 1]])
        out.append((int(f), tris, q))
    return out


def _w_pprism(cells):
    coff, cfid, fsize, foff, flab = _S["coff"], _S["cfid"], _S["fsize"], _S["foff"], _S["flab"]
    out = []
    for c in cells:
        fids = cfid[coff[c]:coff[c + 1]]
        sz = fsize[fids]
        n = len(fids) - 2
        if n < 5 or (sz == n).sum() != 2 or (sz == 4).sum() != n:
            continue
        caps = fids[sz == n]
        sA = set(flab[foff[caps[0]]:foff[caps[0] + 1]].tolist()); sB = set(flab[foff[caps[1]]:foff[caps[1] + 1]].tolist())
        if sA & sB:
            continue
        mp, ok = {}, True
        for f in fids[sz == 4]:
            q = flab[foff[f]:foff[f + 1]].tolist()
            for k in range(4):
                u, w = q[k], q[(k + 1) % 4]
                if u in sB and w in sA:
                    u, w = w, u
                if u in sA and w in sB:
                    ok &= mp.setdefault(u, w) == w
        if ok and len(mp) == n and set(mp.values()) == sB:
            out.append((int(c), int(caps[0]), int(caps[1]), mp))
    return out


def _w_pattern(cells):
    """Per cell: MixedOcTree pattern candidate → (c, why|None, els, xc, req) — why None = accepted locally (shared-face
    conflicts are checked serially by the caller)."""
    X, coff, cfid, fsize, foff, flab, mesh = (_S[k] for k in ("X", "coff", "cfid", "fsize", "foff", "flab", "mesh"))
    fpieces = _S["fpieces"]
    out = []
    for c in cells:
        lab = hex_with_hanging_nodes(mesh, c, coff, cfid)
        if lab is None:
            out.append((c, "not_hex", None, None, None)); continue
        hang = [k for k in lab.values() if k >= 8]
        hit = None
        for R in _MOT_ROT:
            m = 0
            for k in hang:
                m |= 1 << int(R[k])
            if m in MOT_PATTERNS:
                hit = (R, m)
                break
        if hit is None:
            out.append((c, "no_pattern", None, None, None)); continue
        R, m = hit
        pt = {int(R[k]): p for p, k in lab.items()}                  # pattern id → cell point
        els = [(_MOT_VTK[len(e)], [CEN if k == 26 else pt[k] for k in e]) for e in MOT_PATTERNS[m]]
        li = {p: i for i, p in enumerate(list(lab) + [CEN])}
        xc = X[list(lab)].mean(0)
        Xl = np.vstack([X[list(lab)], xc])
        T_ = np.array([t for t, _ in els]); N_ = np.full((len(els), 8), -1, np.int64)
        for i, (t, e) in enumerate(els):
            N_[i, :len(e)] = [li[p] for p in e]
        pmin = element_min_subvol(Xl, N_, T_).min()
        if pmin <= 0:
            out.append((c, "invalid", None, None, None)); continue
        if su2_flip_distorted(Xl, N_, T_)[0].any():                  # SU2 would flip an element (diagonal tests)
            out.append((c, "su2_flip", None, None, None)); continue
        # quality gate: the pattern's smallest SU2 sub-volume must not be smaller than the centre-point split's (in
        # warped snapped cells the hanging nodes are far from mid-edge and some pattern elements become slivers —
        # on the rotor test mesh the max SU2 sub-volume ratio rises from 1,490 to 125,564 without this gate).
        pcs, _ = _cell_pieces(c, fpieces)
        cp = np.unique(np.concatenate([list(q) for q in pcs]))
        Xc = np.vstack([X[cp], X[cp].mean(0)]); lc = {p: i for i, p in enumerate(cp)}; ac = len(cp)
        Tc = np.array([VTK_TET if len(q) == 3 else VTK_PYR for q in pcs]); Nc = np.full((len(pcs), 8), -1, np.int64)
        for i, q in enumerate(pcs):
            v = [lc[p] for p in q][::-1] + [ac]
            Nc[i, :len(v)] = v
        if pmin < element_min_subvol(Xc, Nc, Tc).min():
            out.append((c, "worse_than_centre", None, None, None)); continue
        cnt = {}
        for t, e in els:
            for fl in FACES[t]:
                fv = tuple(e[i] for i in fl if i >= 0)
                cnt.setdefault(frozenset(fv), []).append(fv)
        bfac = [v[0] for v in cnt.values() if len(v) == 1]
        req, why, used = {}, None, 0
        for f in cfid[coff[c]:coff[c + 1]]:
            f = int(f)
            sv = set(flab[foff[f]:foff[f + 1]].tolist())
            on = [fv for fv in bfac if CEN not in fv and set(fv) <= sv]
            used += len(on)
            if fsize[f] <= 4:
                if len(on) != 1 or set(on[0]) != sv:
                    why = "quad_split"
                    break
            else:
                req[f] = frozenset(frozenset(fv) for fv in on)
        if why is None and used != len(bfac):
            why = "invalid"
        out.append((c, why, els if why is None else None, xc, req if why is None else None))
    return out


def _w_no_kernel(cells):
    """Cells (centre-point split) that have no valid interior point with the current face pieces."""
    X, fpieces = _S["X"], _S["fpieces"]
    out = []
    for c in cells:
        pcs, _ = _cell_pieces(c, fpieces)
        x0 = X[np.unique(np.concatenate([list(q) for q in pcs]))].mean(0)
        if subtet_min(X, pcs, x0) > 0 or apex_kernel(X, pcs, x0)[1] > 0:
            continue
        out.append(c)
    return out


def _w_decompose(cells):
    """Per cell: polygonal prism → triangular prisms, else centre-point split → (c, [(vtk, nodes)], apex xyz|None, info);
    the centre point is the placeholder APX."""
    X, fpieces, pprism, prism_ok, apex = (_S[k] for k in ("X", "fpieces", "pprism", "prism_ok", "apex"))
    out = []
    for c in cells:
        info = {}
        if c in prism_ok:
            fa, fb, mp = pprism[c]
            els = []
            for t in fpieces[fa]:
                a_, b_, c_ = t
                if np.dot(np.cross(X[b_] - X[a_], X[c_] - X[a_]), X[mp[a_]] - X[a_]) > 0:   # (0,1,2) must face away
                    b_, c_ = c_, b_
                els.append((a_, b_, c_, mp[a_], mp[b_], mp[c_]))
            E = np.array(els, np.int64)
            Ep = np.pad(E, ((0, 0), (0, 2)), constant_values=-1); Tp = np.full(len(E), VTK_PRISM)
            if element_min_subvol(X, Ep, Tp).min() > 0 and not su2_flip_distorted(X, Ep, Tp)[0].any():
                out.append((c, [(VTK_PRISM, e) for e in els], None, {"pp_split": 1}))
                continue
            info["pp_invalid"] = 1                                   # warped column: centre point below
        pieces, nd = _cell_pieces(c, fpieces)
        info["cancel"] = nd
        x0 = X[np.unique(np.concatenate([list(p) for p in pieces]))].mean(0)
        if apex == "kernel":
            x, mv = apex_kernel(X, pieces, x0)
        else:
            # centroid (best SU2 quality); where it lies outside the cell's kernel (some sub-tet <= 0 — warped 7-/8-gon
            # faces of snappy transition cells) fall back to the kernel apex for THIS cell only
            x, mv = x0, subtet_min(X, pieces, x0)
            if mv <= 0:
                x, mv = apex_kernel(X, pieces, x0)
                info["fallback"] = 1
        els = [(VTK_TET, (p[2], p[1], p[0], APX)) if len(p) == 3 else (VTK_PYR, (p[3], p[2], p[1], p[0], APX))
               for p in pieces]                                      # outward piece → inward base + apex

        def su2_flips(xx):
            Xa = np.vstack([X, xx[None]]); ia = len(X)
            N = np.full((len(els), 8), -1, np.int64)
            for i, (tt, e) in enumerate(els):
                N[i, :len(e)] = [ia if q == APX else q for q in e]
            return int(su2_flip_distorted(Xa, N, np.array([tt for tt, _ in els]))[0].sum())
        if mv > 0 and su2_flips(x):                                  # SU2 would flip a piece → try the kernel point
            xk, mk = apex_kernel(X, pieces, x0)
            if mk > 0 and not su2_flips(xk):
                x, mv = xk, mk
            info["su2_flip_cells"] = int(su2_flips(x) > 0)
        info["mv"] = mv
        info["neg"] = int(mv <= 0)
        out.append((c, els, x, info))
    return out


# ======================================================================================================= converter
def interface_faces(mesh):
    """Faces that become SU2 zone-interface markers: the faces of coupled patches (patch_info has a neighbourPatch,
    e.g. cyclicAMI from createBaffles). of2su2 never splits a conformal mesh itself (zone interfaces must come from
    OpenFOAM), so internal faces are never interface faces."""
    out = []
    pinfo = mesh.get("patch_info", {})
    for name, ptype, st, n in mesh["patches"]:
        if pinfo.get(name, {}).get("neighbourPatch"):
            out.append(np.arange(st, st + n))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def conformal_zone_faces(mesh):
    """Number of INTERNAL faces whose two cells are in different cellZones (cells in no zone count as 'rest').
    > 0 means the cellZone boundary is conformal in OpenFOAM (not baffled) → of2su2 keeps ONE SU2 zone."""
    own, nei = mesh["owner"], mesh["neighbour"]
    if not any(len(v) for v in mesh["zones"].values()) or not len(nei):
        return 0
    zone_of = np.full(int(max(own.max(), nei.max())) + 1, -1, np.int64)   # last cell may be only a neighbour
    for k, cells in enumerate(mesh["zones"].values()):
        zone_of[cells] = k
    return int((zone_of[own[:len(nei)]] != zone_of[nei]).sum())


def triangulate_canonical(X, f):
    """Triangulate polygon/quad f so that a geometrically identical face (e.g. the AMI twin with duplicated points and
    reversed orientation) gets the SAME triangles: the vertex order is made canonical from the COORDINATES (start at the
    lexicographically smallest point, walk towards its smaller neighbour) — Dompierre et al. 1999's consistent-diagonal
    idea with coordinates instead of labels — then the max-min-quality DP (triangulate_polygon) runs on that order.
    Triangles are returned in f's own orientation."""
    f = [int(v) for v in f]; n = len(f)
    key = [tuple(np.round(X[v], 9)) for v in f]
    i0 = min(range(n), key=lambda i: key[i])
    fwd = key[(i0 + 1) % n] < key[(i0 - 1) % n]
    order = [f[(i0 + k) % n] if fwd else f[(i0 - k) % n] for k in range(n)]
    tris, q = triangulate_polygon(X, np.array(order))
    if not fwd:
        tris = [t[::-1] for t in tris]
    return tris, q


def convert_mesh(mesh, apex="centroid", patterns=True, nproc=None, tri_interface=False, log=print):
    t0 = time.time()
    X = mesh["points"]
    foff, flab = mesh["foff"], mesh["flab"]
    fsize = np.diff(foff)
    nc, coff, cfid, csgn = cell_faces(mesh)
    types, nodes = classify(mesh, nc, coff, cfid, csgn)
    log(f"cells {nc:,}: " + ", ".join(f"{n} {int((types == t).sum()):,}" for t, n in
                                      ((VTK_HEX, "hex"), (VTK_PRISM, "prism"), (VTK_PYR, "pyr"), (VTK_TET, "tet"))) +
        f", other {int((types == 0).sum()):,}  ({time.time() - t0:.1f} s)")
    # ---- parallel helpers: workers are forked → they read the big arrays from _S without copying (Linux/macOS fork)
    _S.clear()
    _S.update(X=X, foff=foff, flab=flab, fsize=fsize, coff=coff, cfid=cfid, csgn=csgn, mesh=mesh, apex=apex)
    tl = time.time()

    def stage(msg):
        nonlocal tl
        log(f"  {msg}  ({time.time() - tl:.1f} s)")
        tl = time.time()

    # polygon faces → triangles (once per face)
    poly_faces = np.flatnonzero(fsize > 4)
    fpieces, minq = {}, 1.0
    for part in _pmap(_w_triangulate, poly_faces, nproc):
        for f, tris, q in part:
            fpieces[f] = tris
            minq = min(minq, q)
    stage(f"polygon faces triangulated: {len(poly_faces):,}")
    iface, forced = np.zeros(0, np.int64), set()
    if tri_interface:
        # --tri-interface: SU2's WEIGHTED_AVERAGE (CSlidingMesh) needs triangular interface elements (it gives NaN
        # on quad interfaces). Every interface quad/polygon → canonical triangles (identical on both sides); each cell
        # touching such a face is split by the centre point (SMESH generic split, SMESH_MeshEditor.cxx:2330-2428):
        # triangles → tets, its other quads → pyramids, so the neighbours keep their quads.
        iface = interface_faces(mesh)
        iface = iface[fsize[iface] >= 4]
        own_, nei_ = mesh["owner"], mesh["neighbour"]
        for f in iface:
            fpieces[int(f)], q = triangulate_canonical(X, flab[foff[f]:foff[f + 1]])
            minq = min(minq, q)
            forced.add(int(own_[f]))
            if f < len(nei_):
                forced.add(int(nei_[f]))
        fl = np.array(sorted(forced), np.int64)
        types[fl] = 0
        stage(f"--tri-interface: {len(iface):,} interface faces triangulated, {len(forced):,} adjacent cells → centre split")
    other = np.flatnonzero(types == 0)
    # polygonal prisms (n-gon caps + n side quads, n >= 5: snappy layer cells over split faces) → n-2 triangular
    # prisms, NO added point (extrusion split, as SMESH SplitVolumes splits hexa into prisms). A centre point in these
    # thin cells is almost always poor (SU2 orthogonality < 25°).
    # Caps shared along a layer column get the SAME triangulation (mapped through the side quads) → conformal.
    pprism = {}                                                      # cell → (cap A face, cap B face, A→B vertex map)
    for part in _pmap(_w_pprism, other, nproc):
        for c, fa, fb, mp in part:
            pprism[c] = (fa, fb, mp)
    cap_cells = {}
    for c, (fa, fb, mp) in pprism.items():
        cap_cells.setdefault(fa, []).append(c); cap_cells.setdefault(fb, []).append(c)
    fixed, done, pp_conflict = {}, set(), set()
    for f in iface:
        fixed[int(f)] = frozenset(frozenset(t) for t in fpieces[int(f)])
    for c0 in pprism:                                                # walk each layer column once
        stack = [c0]
        while stack:
            c = stack.pop()
            if c in done:
                continue
            done.add(c)
            fa, fb, mp = pprism[c]
            if fb in fixed and fa not in fixed:
                fa, fb, mp = fb, fa, {v: k for k, v in mp.items()}
            T = fixed.setdefault(fa, frozenset(frozenset(t) for t in fpieces[fa]))
            TM = frozenset(frozenset(mp[v] for v in t) for t in T)
            if fixed.setdefault(fb, TM) != TM:
                pp_conflict.add(c)
                continue
            stack += [c2 for c2 in cap_cells[fa] + cap_cells[fb] if c2 not in done]
    prism_ok = {c for c in pprism if c not in pp_conflict and c not in forced}
    stage(f"polygonal prisms: {len(pprism):,} ({len(pp_conflict)} column conflicts)")

    # hexes with hanging nodes (snappy 2:1 transitions) → MixedOcTree transition pattern (see _MOT_TABLE). The pattern's
    # facets on the cell's faces must keep quads as quads (untouched neighbours) and agree with every triangulation
    # already fixed on a shared polygon face; otherwise (or if any element is not strictly valid) → centre point.
    # Per-cell work in parallel; the shared-face conflict check stays serial, in cell order (same result as serial).
    face_req = {}                                                    # polygon face → pattern cells requiring its pieces
    pat_cells, pat_why = {}, {"no_pattern": 0, "invalid": 0, "su2_flip": 0, "worse_than_centre": 0, "quad_split": 0,
                             "face_conflict": 0}
    _S["fpieces"] = fpieces
    cand = [int(c) for c in other if int(c) not in prism_ok and int(c) not in forced] if patterns else []
    for part in _pmap(_w_pattern, cand, nproc):
        for c, why, els, xc, req in part:
            if why is None:
                for f, Tf in req.items():
                    if fixed.get(f, Tf) != Tf:
                        why = "face_conflict"
                        break
            if why:
                if why != "not_hex":
                    pat_why[why] += 1
                continue
            for f in req:
                if f not in fixed:
                    face_req.setdefault(f, set()).add(c)
                elif f in face_req:
                    face_req[f].add(c)
            fixed.update(req)
            pat_cells[c] = (els, xc)
    stage(f"MixedOcTree patterns: {len(pat_cells):,} cells (fallback {pat_why})")
    fp_dp = dict(fpieces)                                            # DP triangulations, to restore dropped faces

    def apply_fixed():
        fpieces.clear(); fpieces.update(fp_dp)
        for f, T in fixed.items():                                   # pieces ordered as the face → its orientation
            pos = {v: i for i, v in enumerate(flab[foff[f]:foff[f + 1]].tolist())}
            fpieces[f] = sorted(tuple(sorted(t, key=pos.get)) for t in T)   # sorted: order independent of set/process
    apply_fixed()
    # a pattern's triangulation of a shared (warped) polygon face can leave the centre-point cell on the other side
    # without any valid interior point → drop every pattern cell that imposed a triangulation on such a cell's faces,
    # restore the DP triangulation, repeat until stable.
    # Only cells touching a pattern-fixed face can be affected → only those are checked.
    own, nei = mesh["owner"], mesh["neighbour"]
    nint = len(nei)
    n_pat_dropped = 0
    while pat_cells:
        touch = set()
        for f in face_req:
            touch.add(int(own[f]))
            if f < nint:
                touch.add(int(nei[f]))
        chk = sorted(c for c in touch if c not in pat_cells and c not in prism_ok and types[c] == 0)
        _S["fpieces"] = fpieces
        bad = [c for part in _pmap(_w_no_kernel, chk, nproc) for c in part]
        blame = set()
        for c in bad:
            for f in cfid[coff[c]:coff[c + 1]]:
                blame |= face_req.get(int(f), set())
        blame &= set(pat_cells)
        if not blame:
            break
        for c in blame:
            del pat_cells[c]
            n_pat_dropped += 1
        for f in list(face_req):
            face_req[f] -= blame
            if not face_req[f]:
                del face_req[f]; fixed.pop(f, None)
        apply_fixed()
    stage(f"neighbour check: {n_pat_dropped} pattern cells dropped")

    # decompose: pattern cells (serial, cheap) + prism / centre-point cells (parallel). Added points are numbered in
    # cell order exactly as the serial version did.
    _S["fpieces"] = fpieces; _S["prism_ok"] = prism_ok; _S["pprism"] = pprism
    rest = [int(c) for c in other if int(c) not in pat_cells]
    newX, extra = [], {VTK_TET: [], VTK_PYR: [], VTK_PRISM: [], VTK_HEX: []}
    src = {VTK_TET: [], VTK_PYR: [], VTK_PRISM: [], VTK_HEX: []}    # source cell of every decomposition element
    nX = len(X)
    cnt = {"min_sub": np.inf, "neg": 0, "cancel": 0, "fallback": 0, "pp_split": 0, "pp_invalid": 0, "su2_flip_cells": 0}
    by_cell = {}
    infos = {}
    for part in _pmap(_w_decompose, rest, nproc):
        for c, els, x, info in part:
            by_cell[c] = (els, x); infos[c] = info
    # repair: a centre-point cell with no valid interior point (thin warped snappy cells, e.g. a pentagon + 2 quads +
    # triangle cell 0.26 mm thick) → try every fan triangulation of each of its FREE polygon faces (not
    # fixed by a pattern / prism column) and keep the one maximising min(sub-tet volume) of this cell AND the cell on the
    # other side (SMESH SplitVolumes idea: pick the face split variant that works for both neighbours,
    # SMESH_MeshEditor.cxx:2300-2320). Then re-split both cells.
    n_repaired = 0
    centre_cells = set(by_cell) - prism_ok
    for c in [c for c in by_cell if infos[c].get("neg")]:
        for j in range(coff[c], coff[c + 1]):
            f = int(cfid[j])
            if f not in fpieces or f in fixed:
                continue
            c2 = int(nei[f] if own[f] == c else own[f]) if f < nint else None
            if c2 is not None and c2 not in centre_cells:
                continue
            v = flab[foff[f]:foff[f + 1]].tolist(); nv = len(v)
            best, best_val = None, -np.inf
            for k in range(nv):
                fp = dict(fpieces)
                fp[f] = [(v[k], v[(k + i) % nv], v[(k + i + 1) % nv]) for i in range(1, nv - 1)]
                val = np.inf
                for cc in (c, c2):
                    if cc is None:
                        continue
                    pcs, _ = _cell_pieces(cc, fp)
                    x0 = X[np.unique(np.concatenate([list(q) for q in pcs]))].mean(0)
                    m0 = subtet_min(X, pcs, x0)
                    val = min(val, m0 if m0 > 0 else apex_kernel(X, pcs, x0)[1])
                if val > best_val:
                    best, best_val = fp[f], val
            if best_val > 0:
                fpieces[f] = best
                _S["fpieces"] = fpieces
                for cc in (c, c2):
                    if cc is not None:
                        (_, els, x, info), = _w_decompose([cc])
                        by_cell[cc] = (els, x); infos[cc] = info
                n_repaired += 1
                break
    for c, info in infos.items():
        for k in ("neg", "cancel", "fallback", "pp_split", "pp_invalid", "su2_flip_cells"):
            cnt[k] += info.get(k, 0)
        if "mv" in info:
            cnt["min_sub"] = min(cnt["min_sub"], info["mv"])
    APX = -3                                                         # apex placeholder from the workers
    for c in other:
        c = int(c)
        if c in pat_cells:
            els, xc = pat_cells[c]
            a = None
            if any(CEN in e for _, e in els):
                a = nX + len(newX); newX.append(xc)
            for t, e in els:
                extra[t].append(tuple(a if p == CEN else p for p in e)); src[t].append(c)
            continue
        els, x = by_cell[c]
        a = None
        if x is not None:
            a = nX + len(newX); newX.append(x)
        for t, e in els:
            extra[t].append(tuple(a if p == APX else p for p in e)); src[t].append(c)
    del by_cell
    min_sub, n_neg_cells, n_cancel, n_fallback = cnt["min_sub"], cnt["neg"], cnt["cancel"], cnt["fallback"]
    n_pp_split, n_pp_invalid = cnt["pp_split"], cnt["pp_invalid"]
    stage(f"split into elements: {sum(len(v) for v in extra.values()):,}")
    X2 = np.vstack([X, np.array(newX).reshape(-1, 3)])
    log(f"polygon faces {len(poly_faces):,} (min tri quality {minq:.3f}); decomposed cells {len(other):,} → "
        f"{len(extra[VTK_TET]):,} tets + {len(extra[VTK_PYR]):,} pyramids + {len(extra[VTK_PRISM]):,} prisms, "
        f"+ {len(extra[VTK_HEX]):,} hexes, {len(newX):,} added points; MixedOcTree patterns {len(pat_cells):,} cells "
        f"(fallback: {pat_why}, dropped for a neighbour {n_pat_dropped}); polygonal prisms split without apex {n_pp_split} (column conflict "
        f"{len(pp_conflict)}, invalid {n_pp_invalid}); "
        f"cancelled sliver piece pairs {n_cancel}; centroid outside kernel → kernel apex {n_fallback}; "
        f"cells with no strict-kernel apex {n_neg_cells} (repaired by re-triangulating a shared face {n_repaired})  "
        f"({time.time() - t0:.1f} s)")
    # element arrays (native first, then decomposition); keep source cell of every element
    el_type, el_nodes, el_cell = [], [], []
    for t in (VTK_TET, VTK_HEX, VTK_PRISM, VTK_PYR):
        cells = np.flatnonzero(types == t)
        el_type.append(np.full(len(cells), t)); el_nodes.append(nodes[cells]); el_cell.append(cells)
    for t in (VTK_TET, VTK_PYR, VTK_PRISM, VTK_HEX):
        e = np.array(extra[t], np.int64).reshape(-1, NODES[t])
        pad = np.full((len(e), 8), -1, np.int64); pad[:, :NODES[t]] = e
        el_type.append(np.full(len(e), t)); el_nodes.append(pad); el_cell.append(np.array(src[t], np.int64))
    el_type, el_nodes, el_cell = np.concatenate(el_type), np.vstack(el_nodes), np.concatenate(el_cell)
    # boundary faces (polygons → their triangles), outward = OpenFOAM order
    patches = mesh["patches"]
    bfaces, bpatch, bcell = [], [], []
    own_ = mesh["owner"]
    for k, (name, ptype, st, n) in enumerate(patches):
        for f in range(st, st + n):
            if f in fpieces:
                for tri in fpieces[f]:
                    bfaces.append(list(tri) + [-1] * (4 - len(tri))); bpatch.append(k); bcell.append(own_[f])
            else:
                v = flab[foff[f]:foff[f + 1]].tolist()
                bfaces.append(v + [-1] * (4 - len(v))); bpatch.append(k); bcell.append(own_[f])
    bfaces, bpatch = np.array(bfaces, np.int64).reshape(-1, 4), np.array(bpatch, np.int64)
    return {"X": X2, "types": el_type, "nodes": el_nodes, "cell": el_cell, "bfaces": bfaces, "bpatch": bpatch,
            "bcell": np.array(bcell, np.int64), "fpieces": fpieces,
            "names": [p[0] for p in patches], "ncells": nc,
            "stats": {"native": {n: int((types == t).sum()) for t, n in ((VTK_HEX, "hex"), (VTK_PRISM, "prism"),
                                                                         (VTK_PYR, "pyramid"), (VTK_TET, "tet"))},
                      "decomposed_cells": int(len(other)), "polygon_faces": int(len(poly_faces)),
                      "tri_interface_faces": int(len(iface)), "tri_interface_cells": int(len(forced)),
                      "polygon_min_tri_quality": float(minq), "apex": apex,
                      "cancelled_sliver_pairs": int(n_cancel),
                      "apex_cells_outside_kernel": int(n_neg_cells), "centroid_to_kernel_fallback": int(n_fallback), "kernel_repaired_by_face_retriangulation": int(n_repaired),
                      "centre_cells_su2_flip": int(cnt["su2_flip_cells"]),
                      "polygonal_prisms_split": int(n_pp_split), "polygonal_prism_column_conflicts": int(len(pp_conflict)),
                      "polygonal_prisms_invalid_split": int(n_pp_invalid),
                      "mixedoctree_pattern_cells": int(len(pat_cells)), "mixedoctree_fallback": pat_why, "mixedoctree_dropped_for_neighbour": int(n_pat_dropped), "min_subtet_volume": float(min_sub)}}


# ====================================================================================================== validation
def _face_keys(F):
    s = np.sort(F, axis=1) + 1
    return s[:, 0] * (1 << 42) + s[:, 1] * (1 << 21) + s[:, 2], s[:, 3]  # point ids < 2^21? → fall back below


def element_faces(nodes, types):
    out, own = [], []
    for t, faces in FACES.items():
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        cn = nodes[idx]
        loc = np.array(faces)
        f = np.where(loc[None] >= 0, cn[:, np.maximum(loc, 0)], -1)
        out.append(f.reshape(-1, 4)); own.append(np.repeat(idx, len(faces)))
    return np.concatenate(out), np.concatenate(own)


def check(res, sel=None, bsel=None, extra_markers=None):
    """Validation of (a subset of) the converted mesh: volumes, conformity, closure."""
    X, types, nodes = res["X"], res["types"], res["nodes"]
    if sel is not None:
        types, nodes = types[sel], nodes[sel]
    bfaces = res["bfaces"] if bsel is None else res["bfaces"][bsel]
    vol = element_volumes(X, nodes, types)
    F, _ = element_faces(nodes, types)
    key = np.sort(F, axis=1)
    u, cnt = np.unique(key, axis=0, return_counts=True)
    bkey = np.unique(np.sort(bfaces, axis=1), axis=0)
    once = u[cnt == 1]
    # faces used once must be exactly the marker faces
    allk = np.vstack([once, bkey])
    _, c2 = np.unique(allk, axis=0, return_counts=True)
    unmatched = int((c2 == 1).sum())
    # domain volume from the marker faces (divergence theorem, outward)
    vb = 0.0
    for f in bfaces:
        k = f[f >= 0]
        P = X[k]
        fc = P.mean(0)
        for a, b in zip(range(len(k)), list(range(1, len(k))) + [0]):
            vb += np.dot(np.cross(P[a] - fc, P[b] - fc), fc) / 6.0
    return {"elements": int(len(types)), "min_volume": float(vol.min()), "n_nonpositive": int((vol <= 0).sum()),
            "sum_volume": float(vol.sum()), "boundary_volume": float(vb),
            "faces_used_more_than_twice": int((cnt > 2).sum()), "faces_once_not_on_marker_or_reverse": unmatched,
            "boundary_faces": int(len(bfaces))}


# ================================================================================================= dual-mesh quality
def _edge_tables():
    """Per VTK type: edges (a,b) and the outward face holding a→b and the one holding b→a."""
    out = {}
    for t, faces in FACES.items():
        rows = []
        seen = set()
        for fi_, f in enumerate(faces):
            k = [i for i in f if i >= 0]
            for a, b in zip(k, k[1:] + k[:1]):
                if (a, b) in seen or (b, a) in seen:
                    continue
                seen.add((a, b))
                back = [fj for fj, g in enumerate(faces)
                        if (b, a) in list(zip([i for i in g if i >= 0], [i for i in g if i >= 0][1:] + [i for i in g if i >= 0][:1]))]
                rows.append((a, b, fi_, back[0]))
        out[t] = np.array(rows)
    return out


EDGES = _edge_tables()


def _bnd_dual_normals(X, bfaces):
    """Median-dual boundary normal of each (face, corner): quad (corner, next-edge mid, face CG, prev-edge mid)."""
    out_p, out_n = [], []
    for nv in (3, 4):
        F = bfaces[(bfaces[:, 3] < 0) if nv == 3 else (bfaces[:, 3] >= 0), :nv]
        if not len(F):
            continue
        P = X[F]; fc = P.mean(1)
        for j in range(nv):
            p, nx, pv = P[:, j], P[:, (j + 1) % nv], P[:, j - 1]
            m1, m2 = 0.5 * (p + nx), 0.5 * (p + pv)
            out_p.append(F[:, j]); out_n.append(0.5 * (np.cross(m1 - p, fc - p) + np.cross(fc - p, m2 - p)))
    return out_p, out_n


def su2_quality(X, nodes, types, bfaces=None, bpatch=None):
    """SU2-exact dual-CV quality, per point (CPhysicalGeometry::ComputeMeshQualityStatistics, SU2 master 6c0e5c5;
    checked point by point against SU2 v8.5.0 MESH_QUALITY output, max rel. diff 5e-7):
      orth = area-weighted mean over the point's edges of 90° − angle(edge, summed dual-face normal)   [deg]
      ar   = max/min dual-face area, boundary vertex areas included separately per marker
      vr   = max/min |tet(node, edge mid, face CG, element CG)|
    Points used by no element get nan."""
    npnt = len(X)
    keys, S = [], []
    svmin = np.full(npnt, np.inf); svmax = np.zeros(npnt)
    for t, E in EDGES.items():
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        P = X[nodes[idx, :NODES[t]]]
        ec = P.mean(1)
        fcs = np.stack([P[:, [i for i in f if i >= 0]].mean(1) for f in FACES[t]], 1)
        for a, b, fab, fba in E:
            mid = 0.5 * (P[:, a] + P[:, b])
            s = -0.5 * (np.cross(fcs[:, fab] - mid, ec - mid) + np.cross(ec - mid, fcs[:, fba] - mid))
            na, nb = nodes[idx, a], nodes[idx, b]
            keys.append(np.column_stack([np.minimum(na, nb), np.maximum(na, nb)]))
            S.append(s * np.where(na < nb, 1.0, -1.0)[:, None])
        for fi, f in enumerate(FACES[t]):
            k = [i for i in f if i >= 0]
            for a, b in zip(k, k[1:] + k[:1]):
                mid = 0.5 * (P[:, a] + P[:, b])
                for nd in (a, b):
                    v = np.abs(np.einsum("ij,ij->i", np.cross(mid - P[:, nd], fcs[:, fi] - P[:, nd]), ec - P[:, nd])) / 6
                    np.minimum.at(svmin, nodes[idx, nd], v); np.maximum.at(svmax, nodes[idx, nd], v)
    keys, S = np.vstack(keys), np.vstack(S)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    Ssum = np.zeros((len(u), 3)); np.add.at(Ssum, inv.ravel(), S)
    e = X[u[:, 1]] - X[u[:, 0]]
    area = np.linalg.norm(Ssum, axis=1)
    cosang = np.clip(np.einsum("ij,ij->i", e, Ssum) / (np.linalg.norm(e, axis=1) * area + 1e-300), -1, 1)
    orth_e = 90.0 - np.degrees(np.arccos(cosang))
    num = np.zeros(npnt); den = np.zeros(npnt)
    amin = np.full(npnt, np.inf); amax = np.zeros(npnt)
    for c in (0, 1):
        np.add.at(num, u[:, c], area * orth_e); np.add.at(den, u[:, c], area)
        np.minimum.at(amin, u[:, c], area); np.maximum.at(amax, u[:, c], area)
    if bfaces is not None and len(bfaces):
        for m in np.unique(bpatch):
            pts, nrm = _bnd_dual_normals(X, bfaces[bpatch == m])
            bn = np.zeros((npnt, 3))
            for p_, n_ in zip(pts, nrm):
                np.add.at(bn, p_, n_)
            ba = np.linalg.norm(bn, axis=1); on = ba > 0
            amin[on] = np.minimum(amin[on], ba[on]); amax[on] = np.maximum(amax[on], ba[on])
    with np.errstate(invalid="ignore", divide="ignore"):
        orth = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        ar = np.where(den > 0, amax / amin, np.nan)
        vr = np.where(den > 0, svmax / svmin, np.nan)
    return {"orth": orth, "ar": ar, "vr": vr, "edges": u, "orth_edge": orth_e}


# ==================================================================== SU2's own orientation checks (what SU2 will flip)
# SU2 v8.5.0 CPhysicalGeometry::Check_IntElem_Orientation (:3984-4100): an element is FLIPPED when all its test tets are
# negative ("(b-a)x(c-a)·(d-a) < 0"), and reported "distorted" when only some are. The tests split quad faces along fixed
# diagonals, so a valid element on a warped quad can fail them → SU2 flips a correct element (seen on pyramids
# after --opt-vr). Check_BoundElem_Orientation (:4147-4250): a boundary quad is flipped when >= 3 of 4 tets with an interior
# reference point (first node of the adjacent element not on the face) fail.
SU2_TESTS = {VTK_TET: [(0, 1, 2, 3)], VTK_PYR: [(0, 1, 2, 4), (2, 3, 0, 4)], VTK_PRISM: [(0, 2, 1, 3), (3, 4, 5, 2)],
             VTK_HEX: [(0, 1, 2, 5), (0, 2, 3, 7), (4, 6, 5, 1), (4, 7, 6, 3)]}


def _su2_relabels():
    """Orientation-preserving node relabellings of each element type (same element, other start node)."""
    hexc = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    hid = {tuple(v): i for i, v in enumerate(hexc)}
    hexp = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sg in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            M = np.zeros((3, 3), int)
            for i in range(3):
                M[i, perm[i]] = sg[i]
            if round(np.linalg.det(M)) == 1:
                hexp.append(tuple(hid[tuple((M @ (2 * v - 1) + 1) // 2)] for v in hexc))
    return {VTK_TET: [(0, 1, 2, 3)],
            VTK_PYR: [(0, 1, 2, 3, 4), (1, 2, 3, 0, 4), (2, 3, 0, 1, 4), (3, 0, 1, 2, 4)],
            VTK_PRISM: [(0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4),
                        (3, 5, 4, 0, 2, 1), (5, 4, 3, 2, 1, 0), (4, 3, 5, 1, 0, 2)],
            VTK_HEX: hexp}


SU2_RELABELS = _su2_relabels()


def su2_neg_tests(X, conn, t):
    """Number of SU2 orientation test tets that are negative, for elements conn (n, k) of one VTK type t."""
    neg = np.zeros(len(conn), int)
    for a, b, c, d in SU2_TESTS[t]:
        A = X[conn[:, a]]
        neg += np.einsum("ij,ij->i", np.cross(X[conn[:, b]] - A, X[conn[:, c]] - A), X[conn[:, d]] - A) < 0
    return neg


def su2_best_relabel(X, nodes, types):
    """Per element: the relabelling with the fewest negative SU2 test tets → (best perm index, its negative count)."""
    best_i = np.zeros(len(types), int); best_n = np.zeros(len(types), int)
    for t in NODES:
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        cn = nodes[idx, :NODES[t]]
        counts = np.stack([su2_neg_tests(X, cn[:, list(p)], t) for p in SU2_RELABELS[t]], 1)
        best_i[idx] = counts.argmin(1); best_n[idx] = counts.min(1)
    return best_i, best_n


def su2_flip_distorted(X, nodes, types, relabel=True):
    """(flip, distorted) boolean arrays as SU2 would report them (after the best relabelling if relabel)."""
    if relabel:
        _, n = su2_best_relabel(X, nodes, types)
    else:
        n = np.zeros(len(types), int)
        for t in NODES:
            idx = np.flatnonzero(types == t)
            if len(idx):
                n[idx] = su2_neg_tests(X, nodes[idx, :NODES[t]], t)
    ntest = np.array([len(SU2_TESTS.get(int(t), [0])) for t in types]) if len(types) < 64 else \
        np.select([types == t for t in NODES], [len(SU2_TESTS[t]) for t in NODES], 1)
    return n == ntest, (n > 0) & (n < ntest)


def su2_relabel(X, nodes, types):
    """Apply the best relabelling in place (geometry unchanged); returns counts before/after (SU2 tests above)."""
    f0, d0 = su2_flip_distorted(X, nodes, types, relabel=False)
    bi, _ = su2_best_relabel(X, nodes, types)
    for t in NODES:
        for k, p in enumerate(SU2_RELABELS[t]):
            if k == 0:
                continue
            idx = np.flatnonzero((types == t) & (bi == k))
            if len(idx):
                nodes[idx, :NODES[t]] = nodes[idx][:, list(p)]
    f1, d1 = su2_flip_distorted(X, nodes, types, relabel=False)
    return {"flip_before": int(f0.sum()), "distorted_before": int(d0.sum()), "relabelled": int((bi > 0).sum()),
            "flip": int(f1.sum()), "distorted": int(d1.sum()),
            "flip_by_type": {n: int(f1[types == t].sum()) for t, n in ((VTK_TET, "tet"), (VTK_PYR, "pyramid"),
                                                                       (VTK_PRISM, "prism"), (VTK_HEX, "hex"))},
            "distorted_by_type": {n: int(d1[types == t].sum()) for t, n in ((VTK_TET, "tet"), (VTK_PYR, "pyramid"),
                                                                            (VTK_PRISM, "prism"), (VTK_HEX, "hex"))}}


def su2_boundary_flips(X, nodes, types, bfaces):
    """SU2's boundary-quad test (Check_BoundElem_Orientation): for each boundary quad, find its element, take the first
    element node not on the face as reference point, count failing tets; >= 3 → SU2 flips, == 2 → 'distorted'.
    bfaces in SU2 winding (as written: reversed OpenFOAM order)."""
    q = np.flatnonzero(bfaces[:, 3] >= 0)
    if not len(q):
        return {"quads": 0, "flip": 0, "distorted": 0}
    Q = bfaces[q][:, ::-1]                                           # SU2 winding (writer reverses)
    onb = np.zeros(len(X), bool); onb[Q.ravel()] = True
    cand = np.flatnonzero((onb[np.maximum(nodes, 0)] & (nodes >= 0)).any(1))
    F, fo = element_faces(nodes[cand], types[cand])
    m = F[:, 3] >= 0
    key = lambda A: np.sort(A, 1)
    Fk = key(F[m]); Fe = cand[fo[m]]
    o = np.lexsort(Fk.T[::-1]); Fk, Fe = Fk[o], Fe[o]
    Qk = key(Q); oq = np.lexsort(Qk.T[::-1]); Qs = Qk[oq]
    j = np.searchsorted(Fk[:, 0], Qs[:, 0])
    el = np.full(len(Q), -1)
    for r in range(len(Qs)):                                         # few candidates per first-node → short scan
        k = j[r]
        while k < len(Fk) and Fk[k, 0] == Qs[r, 0]:
            if (Fk[k] == Qs[r]).all():
                el[oq[r]] = Fe[k]; break
            k += 1
    ok = el >= 0
    E = nodes[el[ok]]; Qo = Q[ok]
    ref = np.array([next(v for v in e if v >= 0 and v not in set(f)) for e, f in zip(E.tolist(), Qo.tolist())])
    fail = np.zeros(len(Qo), int)
    for a, b, c in ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)):
        A = X[Qo[:, a]]
        fail += np.einsum("ij,ij->i", np.cross(X[Qo[:, b]] - A, X[Qo[:, c]] - A), X[ref] - A) < 0
    bad = fail >= 3
    return {"quads": int(len(q)), "unmatched": int((~ok).sum()), "flip": int(bad.sum()),
            "distorted": int((fail == 2).sum()), "_faces": Qo[bad], "_elems": el[ok][bad]}


def su2_fix_boundary_flips(X, nodes, types, bfaces):
    """SU2 picks its reference point for a boundary quad as the FIRST node of the adjacent element not on the face.
    Where SU2 would wrongly flip a (correctly oriented) quad, relabel that element so its first off-face node is one for
    which SU2's test accepts the correct winding — only among relabellings that keep the element's own SU2 test count at
    its best. Geometry unchanged. Returns (fixed, remaining). (SU2 v8.5.0 Check_BoundElem_Orientation :4147-4250)"""
    r = su2_boundary_flips(X, nodes, types, bfaces)
    fixed = 0
    for f, e in zip(r.get("_faces", []), r.get("_elems", [])):
        t = int(types[e]); k = NODES[t]; cur = nodes[e, :k].copy(); fs = set(f.tolist())
        base_neg = su2_neg_tests(X, cur[None], t)[0]
        best = None
        for p in SU2_RELABELS[t]:
            cand = cur[list(p)]
            if su2_neg_tests(X, cand[None], t)[0] > base_neg:
                continue
            ref = next(v for v in cand if v not in fs)
            fail = 0
            for a, b, c in ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)):
                A = X[f[a]]
                fail += np.dot(np.cross(X[f[b]] - A, X[f[c]] - A), X[ref] - A) < 0
            if fail <= 2 and (best is None or fail < best[0]):
                best = (fail, cand)
        if best is not None:
            nodes[e, :k] = best[1]; fixed += 1
    r2 = su2_boundary_flips(X, nodes, types, bfaces)
    return fixed, r2


def element_min_subvol(X, conn, types):
    """Per element the smallest sub-tet (outward face edge, face CG, element CG) volume — > 0 for every sub-tet means the
    element is valid in the strict sense (stricter than element_volumes > 0)."""
    out = np.full(len(types), np.inf)
    for t, faces in FACES.items():
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        P = X[conn[idx, :NODES[t]]]
        cc = P.mean(1)
        for f in faces:
            k = [i for i in f if i >= 0]
            fc = P[:, k].mean(1)
            for a, b in zip(k, k[1:] + k[:1]):
                out[idx] = np.minimum(out[idx], -tet_vol(P[:, a], P[:, b], fc, cc))
    return out


def optimise_apex(res, n_orig, thr=40.0, sweeps=2, target="orth", q=None, log=print):
    """Move only the added points (index >= n_orig: centre points of centre-point / pattern splits) to improve SU2's
    dual-CV quality — local derivative-free (compass) search in the style of Mesquite / GETMe local smoothing, topology
    unchanged (conformity untouched). Watched points V = the moved point + its cell's vertices; quality exact on the local
    patch (all elements touching V). Every sub-tet of the cell's elements must stay > 0 (element_min_subvol).
      target "orth": candidates = added points with SU2 orthogonality < thr; maximise min orth over V (tie: mean); max AR
                     and max sub-volume ratio over V may not exceed their start values (orth-only optimisation pushed
                     points onto faces: AR 173 → 1,250, VR 357 → 1.2e7 on the test mesh).
      target "vr":   candidates = added points of cells touching a point with SU2 sub-volume ratio > thr; minimise max
                     sub-volume ratio over V; min orth over V may not drop and max AR may not rise (the worst ratios
                     typically sit at corner points next to centre-point cells).
    q: precomputed su2_quality of the whole mesh (else computed)."""
    X, types, nodes = res["X"], res["types"], res["nodes"]
    ne = len(types)
    # point → elements (CSR)
    rows = np.repeat(np.arange(ne), 8).reshape(ne, 8)
    m = nodes >= 0
    pe_p, pe_e = nodes[m], rows[m]
    o = np.argsort(pe_p, kind="stable")
    pe_p, pe_e = pe_p[o], pe_e[o]
    ptr = np.searchsorted(pe_p, np.arange(len(X) + 1))

    def els_of(pts):
        return np.unique(np.concatenate([pe_e[ptr[p]:ptr[p + 1]] for p in pts]))

    if q is None:
        q = su2_quality(X, nodes, types, res.get("bfaces"), res.get("bpatch"))
    if target == "orth":
        cand = [a for a in range(n_orig, len(X)) if ptr[a + 1] > ptr[a] and q["orth"][a] < thr]
    else:
        bad = np.flatnonzero(np.nan_to_num(q["vr"]) > thr)
        ce_all = els_of(bad) if len(bad) else np.array([], np.int64)
        nb = nodes[ce_all]
        cand = sorted({int(a) for a in nb[nb >= n_orig]})
    moved, gain = 0, []
    for sweep in range(sweeps):
        for a in cand:
            ce = pe_e[ptr[a]:ptr[a + 1]]                              # the cell's elements (point is private to it)
            V = np.unique(nodes[ce][nodes[ce] >= 0])
            patch = els_of(V)
            pl = np.unique(nodes[patch][nodes[patch] >= 0])
            Ln = np.where(nodes[patch] >= 0, np.searchsorted(pl, nodes[patch]), -1)   # pl sorted → local ids
            Lt = types[patch]
            Lce = np.flatnonzero(np.isin(patch, ce))
            Xl = X[pl].copy(); ia = int(np.searchsorted(pl, a)); Vl = np.searchsorted(pl, V)

            lim = {}

            def score(x):
                Xl[ia] = x
                if element_min_subvol(Xl, Ln[Lce], Lt[Lce]).min() <= 0:
                    return (-1e18, -1e18)
                fl, di = su2_flip_distorted(Xl, Ln[Lce], Lt[Lce])          # SU2 orientation view (best relabelling)
                if fl.any() or ("dist" in lim and di.sum() > lim["dist"]):
                    return (-1e18, -1e18)
                lim.setdefault("dist", int(di.sum()))
                ql = su2_quality(Xl, Ln, Lt)
                arv, vrv = float(ql["ar"][Vl].max()), float(ql["vr"][Vl].max())
                ov = ql["orth"][Vl]
                if target == "orth":
                    if "ar" in lim and (arv > lim["ar"] or vrv > lim["vr"]):    # never trade AR / volume ratio for orthogonality
                        return (-1e18, -1e18)
                    lim.setdefault("ar", arv); lim.setdefault("vr", vrv)
                    return (float(ov.min()), float(ov.mean()))
                if "ar" in lim and (arv > lim["ar"] or ov.min() < lim["orth"]):   # never trade orthogonality / AR for volume ratio
                    return (-1e18, -1e18)
                lim.setdefault("ar", arv); lim.setdefault("orth", float(ov.min()))
                return (-vrv, float(ov.min()))

            x = X[a].copy(); best = score(x); start = best
            Y = Xl[Vl] - Xl[Vl].mean(0)
            _, sv, ax = np.linalg.svd(Y, full_matrices=False)
            steps = 0.25 * np.maximum(sv / np.sqrt(len(Y)) * 2, 1e-12)  # ~ quarter of the extent along each axis
            for _ in range(10):
                improved = True
                while improved:
                    improved = False
                    for k in range(3):
                        for sgn in (1.0, -1.0):
                            xt = x + sgn * steps[k] * ax[k]
                            st = score(xt)
                            if st > best:
                                x, best, improved = xt, st, True
                steps *= 0.5
            Xl[ia] = x
            if best > start:
                X[a] = x; moved += 1; gain.append(best[0] - start[0])
    what = f"apex points < {thr}°" if target == "orth" else f"added points next to sub-volume ratio > {thr:g}"
    log(f"point optimisation ({target}): {len(cand)} {what} (of {len(X) - n_orig}), moved {moved}, "
        f"gain median {np.median(gain) if gain else 0:.3g}")
    return {"target": target, "candidates": len(cand), "moved": moved}


def su2_quality_summary(q):
    o, ar, vr = q["orth"], q["ar"], q["vr"]
    ok = np.isfinite(o)
    o, ar, vr = o[ok], ar[ok], vr[ok]
    return {"points": int(ok.sum()), "orth_min_deg": float(o.min()),
            "orth_p01_p1_p5_median": [float(v) for v in np.percentile(o, [0.1, 1, 5, 50])],
            "points_orth_lt_20": int((o < 20).sum()), "points_orth_lt_25": int((o < 25).sum()),
            "points_orth_lt_30": int((o < 30).sum()), "ar_max": float(ar.max()), "ar_p99": float(np.percentile(ar, 99)),
            "vr_max": float(vr.max()), "edge_orth_min_deg": float(q["orth_edge"].min())}


def dual_quality(X, nodes, types):
    """Stricter variant (kept for comparison): per point the MIN over its edges of the edge orthogonality (SU2 uses the
    area-weighted MEAN, see su2_quality) and max/min dual-face area without boundary faces."""
    keys, S = [], []
    for t, E in EDGES.items():
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        P = X[nodes[idx, :NODES[t]]]                                 # (n,k,3)
        ec = P.mean(1)
        fcs = []
        for f in FACES[t]:
            k = [i for i in f if i >= 0]
            fcs.append(P[:, k].mean(1))
        fcs = np.stack(fcs, 1)                                       # (n,nf,3)
        for a, b, fab, fba in E:
            mid = 0.5 * (P[:, a] + P[:, b])
            s = -0.5 * (np.cross(fcs[:, fab] - mid, ec - mid) + np.cross(ec - mid, fcs[:, fba] - mid))
            na, nb = nodes[idx, a], nodes[idx, b]
            sg = np.where(na < nb, 1.0, -1.0)[:, None]
            keys.append(np.column_stack([np.minimum(na, nb), np.maximum(na, nb)])); S.append(s * sg)
    keys, S = np.vstack(keys), np.vstack(S)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    Ssum = np.zeros((len(u), 3))
    np.add.at(Ssum, inv, S)
    e = X[u[:, 1]] - X[u[:, 0]]
    cosang = np.einsum("ij,ij->i", e, Ssum) / (np.linalg.norm(e, axis=1) * np.linalg.norm(Ssum, axis=1) + 1e-300)
    orth = 90.0 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))      # 90 = orthogonal
    area = np.linalg.norm(Ssum, axis=1)
    npnt = len(X)
    pmin = np.full(npnt, 90.0)
    np.minimum.at(pmin, u[:, 0], orth); np.minimum.at(pmin, u[:, 1], orth)
    amax = np.zeros(npnt); amin = np.full(npnt, np.inf)
    for col in (0, 1):
        np.maximum.at(amax, u[:, col], area); np.minimum.at(amin, u[:, col], area)
    used = np.isfinite(amin)
    ar = amax[used] / np.maximum(amin[used], 1e-300)
    return {"edges": int(len(u)), "orth_min_deg": float(orth.min()),
            "orth_p1_deg": float(np.percentile(orth, 1)),
            "points_orth_lt_20": int((pmin[used] < 20).sum()), "points_orth_lt_40": int((pmin[used] < 40).sum()),
            "cv_face_ar_max": float(ar.max()), "cv_face_ar_p99": float(np.percentile(ar, 99)),
            "_point_orth": pmin, "_point_ar": np.where(used, amax / np.maximum(amin, 1e-300), np.nan)}


# =========================================================================================================== writer
def _fmt_rows(f, arr, fmt):
    if len(arr):
        np.savetxt(f, arr, fmt=fmt)


def write_su2(path, X, types, nodes, bfaces, bpatch, names, zone_header=None, fh=None):
    used = np.zeros(len(X), bool)
    for t in NODES:
        used[nodes[types == t, :NODES[t]].ravel()] = True
    new = np.cumsum(used) - 1
    close = fh is None
    f = fh or open(path, "w")
    if zone_header:
        f.write(zone_header)
    f.write(f"NDIME= 3\nNELEM= {len(types)}\n")
    eid = 0
    for t in (VTK_TET, VTK_HEX, VTK_PRISM, VTK_PYR):
        idx = np.flatnonzero(types == t)
        if len(idx):
            _fmt_rows(f, np.column_stack([np.full(len(idx), t), new[nodes[idx, :NODES[t]]],
                                          np.arange(eid, eid + len(idx))]), "%d")
            eid += len(idx)
    Xu = X[used]
    f.write(f"NPOIN= {len(Xu)}\n")
    _fmt_rows(f, np.column_stack([Xu, np.arange(len(Xu))]), ["%.17g", "%.17g", "%.17g", "%d"])
    present = [i for i in range(len(names)) if np.any(bpatch == i)]
    f.write(f"NMARK= {len(present)}\n")
    for i in present:
        fm = bfaces[bpatch == i]
        f.write(f"MARKER_TAG= {names[i]}\nMARKER_ELEMS= {len(fm)}\n")
        tri = fm[:, 3] < 0
        # SU2 wants the right-hand normal of a boundary face pointing INTO the domain (Check_BoundElem_Orientation);
        # OpenFOAM's points out → write reversed (internal arrays keep OpenFOAM's outward convention)
        _fmt_rows(f, np.column_stack([np.full(tri.sum(), 5), new[fm[tri, :3][:, ::-1]]]), "%d")
        _fmt_rows(f, np.column_stack([np.full((~tri).sum(), 9), new[fm[~tri][:, ::-1]]]), "%d")
    if close:
        f.close()
    return int(len(Xu))


def merge_patches(res, mesh, a, b, tol_rel=1e-9):
    """Glue patch b onto the coincident points of patch a (one zone): point map + drop both patches' faces."""
    names = res["names"]
    ia, ib = names.index(a), names.index(b)
    X = res["X"]
    pa = np.unique(res["bfaces"][res["bpatch"] == ia]); pa = pa[pa >= 0]
    pb = np.unique(res["bfaces"][res["bpatch"] == ib]); pb = pb[pb >= 0]
    if len(pa) != len(pb):
        raise ValueError(f"{a}: {len(pa)} points, {b}: {len(pb)} — not a conformal pair")
    tol = tol_rel * np.abs(X).max()
    oa = np.lexsort(np.round(X[pa] / tol).T[::-1]); ob = np.lexsort(np.round(X[pb] / tol).T[::-1])
    gap = float(np.abs(X[pa[oa]] - X[pb[ob]]).max())
    if gap > 1e3 * tol:
        raise ValueError(f"{a}/{b} points do not coincide (max gap {gap:g})")
    pmap = np.arange(len(X)); pmap[pb[ob]] = pa[oa]
    keep = (res["bpatch"] != ia) & (res["bpatch"] != ib)
    res["nodes"] = np.where(res["nodes"] >= 0, pmap[np.maximum(res["nodes"], 0)], -1)
    res["bfaces"] = res["bfaces"][keep]
    res["bpatch"] = res["bpatch"][keep]
    return gap


def coupled_pairs(polymesh_dir):
    """[(A, B)] for cyclicAMI / cyclic patches that name each other as neighbourPatch (each pair once)."""
    text = _strip_comments((Path(polymesh_dir) / "boundary").read_bytes()).decode(errors="ignore")
    nb = {}
    for m in re.finditer(r"(\S+)\s*\{([^}]*)\}", text[text.index("(", text.index("}", text.index("FoamFile"))):]):
        t = re.search(r"\btype\s+([^;]+);", m.group(2))
        n = re.search(r"\bneighbourPatch\s+([^;]+);", m.group(2))
        if t and n and t.group(1).strip() in ("cyclicAMI", "cyclic"):
            nb[m.group(1)] = n.group(1).strip()
    return sorted({tuple(sorted((a, b))) for a, b in nb.items() if nb.get(b) == a})


def bad_points(res, n_orig, q, thr=25.0):
    """Where the SU2-poor points are: added centre points vs original mesh points, and next to native/decomposed cells."""
    o = q["orth"]
    bad = np.flatnonzero(o < thr)
    dec = np.zeros(len(res["X"]), bool)                               # points of decomposed cells (they own an apex)
    d = res["nodes"][(res["nodes"] >= n_orig).any(1)]
    dec[d[d >= 0]] = True
    return {"threshold_deg": thr, "points": int(len(bad)), "added_centre_points": int((bad >= n_orig).sum()),
            "original_points": int((bad < n_orig).sum()),
            "original_points_touching_decomposed_cells": int(((bad < n_orig) & dec[bad]).sum())}


# ============================================================================ zones, interfaces and mesh information
_SU2_BAD = re.compile(r"[\s(){}:,%=]")


def su2_name(name, used):
    """SU2-safe, unique marker/zone name: SU2's config tokenizer splits on space ( ) { } : , (CConfig.cpp:8406-8408),
    % starts a comment, = separates; binary names max 64 chars (option_structure.hpp:271)."""
    n = _SU2_BAD.sub("_", str(name))[:64] or "_"
    base, k = n, 2
    while n in used:
        n = f"{base[:60]}_{k}"; k += 1
    used.add(n)
    return n


def mesh_info(mesh, res, multizone, n_conf=0):
    """Everything about the OpenFOAM mesh that the SU2 file cannot hold, plus what is not carried over (written to
    <out>.info.json and printed) — nothing is dropped silently."""
    used, names, renamed = set(), [], {}
    for p in res["names"]:
        n = su2_name(p, used); names.append(n)
        if n != p:
            renamed[p] = n
    pinfo = mesh.get("patch_info", {})
    info = {"tool": "of2su2", "openfoam_mesh": mesh["dir"], "multizone": bool(multizone),
            "markers": [{"marker": names[i], "openfoam_patch": p, "type": pinfo.get(p, {}).get("type"),
                         "inGroups": pinfo.get(p, {}).get("inGroups", []),
                         "neighbourPatch": pinfo.get(p, {}).get("neighbourPatch")} for i, p in enumerate(res["names"])],
            "cellZones": {k: int(len(v)) for k, v in mesh["zones"].items()},
            "faceZones": {k: int(len(v)) for k, v in mesh.get("face_zones", {}).items()},
            "pointZones": {k: int(len(v)) for k, v in mesh.get("point_zones", {}).items()},
            "renamed": renamed, "zones": [], "interfaces": [], "not_carried_over": [],
            "limits": ["3D meshes only (2D 'empty' patches are not supported)",
                       "reads constant/polyMesh only (reconstructPar a decomposed case first)",
                       "periodic 'cyclic' patches with a transform are not converted (cyclic/AMI pairs with coincident "
                       "points are glued in single-zone output or kept as interface markers in multizone output)",
                       "mesh only: no field data, no boundary conditions, no solver settings"],
            "_marker_names": names}
    nc = info["not_carried_over"]
    for k, v in info["pointZones"].items():
        nc.append(f"pointZone '{k}' ({v} points): the SU2 mesh format has no point sets")
    for k in mesh.get("sets", []):
        nc.append(f"polyMesh/sets/{k}: checkMesh/topoSet diagnostics, not mesh data — ignored")
    nc.append("patch types / inGroups / neighbourPatch: SU2 markers carry names only — kept in this file ('markers')")
    if not multizone:
        for k, v in info["cellZones"].items():
            why = (f"its boundary is conformal in OpenFOAM ({n_conf} internal faces between zones); of2su2 does not split "
                   f"meshes — baffle it in OpenFOAM (createBaffles + mergeOrSplitBaffles -split) to get an SU2 zone"
                   if n_conf else "single-zone output requested")
            nc.append(f"cellZone '{k}' ({v} cells): {why} — not written as an SU2 zone")
        for k, v in info["faceZones"].items():
            nc.append(f"faceZone '{k}' ({v} faces): single-zone output — interior faces, not written as markers")
    return info


def zone_split(mesh, res, info):
    """Multizone output (SU2 'NZONE=' file): one SU2 zone per OpenFOAM cellZone + 'rest'. Faces between zones are
    written on BOTH sides as markers '<faceZone or interface>_<zone>' (outward from each zone; the same pieces on both
    sides → coincident points). Patches spanning zones are split per zone ('<patch>_<zone>'). (SU2 multizone mesh
    format: CSU2ASCIIMeshReaderBase.cpp:36-80)"""
    names = info["_marker_names"]; used = set(names)
    zused = set()
    zlist = [su2_name(z, zused) for z in mesh["zones"]]
    nc = res["ncells"]
    zone_of = np.full(nc, -1, np.int64)
    for k, cells in enumerate(mesh["zones"].values()):
        zone_of[cells] = k
    if (zone_of < 0).any():
        zone_of[zone_of < 0] = len(zlist); zlist.append(su2_name("rest", zused))
    ez = zone_of[res["cell"]]
    bz = zone_of[res["bcell"]]
    bp = res["bpatch"].copy()
    for pi in np.unique(res["bpatch"]):
        zs = np.unique(bz[res["bpatch"] == pi])
        if len(zs) > 1:
            for z in zs:
                n = su2_name(f"{names[pi]}_{zlist[z]}", used); names.append(n)
                bp[(res["bpatch"] == pi) & (bz == z)] = len(names) - 1
            info["not_carried_over"].append(f"patch '{names[pi]}' spans zones {[zlist[z] for z in zs]} → split into "
                                            f"per-zone markers '<patch>_<zone>'")
    own, nei = mesh["owner"], mesh["neighbour"]; nint = len(nei)
    foff, flab = mesh["foff"], mesh["flab"]
    fz_of = np.full(len(foff) - 1, -1, np.int64)
    fzn = list(mesh.get("face_zones", {}))
    for k, faces in enumerate(mesh.get("face_zones", {}).values()):
        fz_of[faces] = k
    zi = np.flatnonzero(zone_of[own[:nint]] != zone_of[nei])
    fp = res["fpieces"]
    groups = {}                                                       # (zone, other zone, base) → [pieces]
    for f in zi:
        za, zb = int(zone_of[own[f]]), int(zone_of[nei[f]])
        base = fzn[fz_of[f]] if fz_of[f] >= 0 else "interface"
        pcs = fp[f] if f in fp else [tuple(flab[foff[f]:foff[f + 1]].tolist())]
        groups.setdefault((za, zb, base), []).extend(pcs)
        groups.setdefault((zb, za, base), []).extend(tuple(p[::-1]) for p in pcs)
    partners = {}
    for za, zb, base in groups:
        partners.setdefault((za, base), set()).add(zb)
    mname = {}
    for za, zb, base in groups:
        tag = f"{base}_{zlist[za]}" + (f"_{zlist[zb]}" if len(partners[(za, base)]) > 1 else "")
        mname[(za, zb, base)] = su2_name(tag, used); names.append(mname[(za, zb, base)])
    done = set()
    for (za, zb, base), pcs in groups.items():
        if (zb, za, base) in done:
            continue
        done.add((za, zb, base))
        info["interfaces"].append({"kind": "conformal internal faces", "faceZone": base if base != "interface" else None,
                                   "zones": [zlist[za], zlist[zb]],
                                   "markers": [mname[(za, zb, base)], mname[(zb, za, base)]],
                                   "pieces_per_side": len(pcs)})
    in_iface = np.zeros(len(foff) - 1, bool); in_iface[zi] = True
    for k, faces in enumerate(mesh.get("face_zones", {}).values()):
        on_patch = faces >= nint
        inside = int((~in_iface[faces] & ~on_patch).sum())
        if on_patch.any():
            pn = sorted({res["names"][i] for i in range(len(mesh["patches"]))
                         if np.any((faces >= mesh["patches"][i][2]) & (faces < mesh["patches"][i][2] + mesh["patches"][i][3]))})
            info["faceZone_notes"] = info.get("faceZone_notes", []) + [
                f"faceZone '{fzn[k]}': {int(on_patch.sum())} faces are patch faces of {pn} (written as those markers)"]
        if inside:
            info["not_carried_over"].append(f"faceZone '{fzn[k]}': {inside} of {len(faces)} faces lie inside one zone "
                                            f"(not between zones, not on a patch) → not written as markers")
    pinfo = mesh.get("patch_info", {})
    pz = {res["names"][pi]: sorted({zlist[z] for z in np.unique(bz[res["bpatch"] == pi])}) for pi in np.unique(res["bpatch"])}
    for a, d in pinfo.items():
        b = d.get("neighbourPatch")
        if b and a < b and a in pz and b in pz and pz[a] != pz[b]:
            info["interfaces"].append({"kind": f"coupled patches ({d.get('type')})", "faceZone": None,
                                       "zones": [pz[a][0], pz[b][0]], "markers": [names[res["names"].index(a)],
                                                                                  names[res["names"].index(b)]]})
    parts = []
    for k, zname in enumerate(zlist):
        sel = ez == k
        if not sel.any():
            continue
        bf = [res["bfaces"][bz == k]]; bpz = [bp[bz == k]]
        for (za, zb, base), pcs in groups.items():
            if za == k:
                bf.append(np.array([list(p) + [-1] * (4 - len(p)) for p in pcs], np.int64))
                bpz.append(np.full(len(pcs), names.index(mname[(za, zb, base)])))
        parts.append({"name": zname, "sel": sel, "bfaces": np.vstack(bf), "bpatch": np.concatenate(bpz)})
        info["zones"].append({"zone": zname, "openfoam_cellZone": list(mesh["zones"])[k] if k < len(mesh["zones"]) else None,
                              "cells": int((zone_of == k).sum()), "elements": int(sel.sum())})
    return parts


def convert(case, out, single_zone=False, merge=None, apex="centroid", opt_apex=None, auto_merge=True, patterns=True,
            nproc=None, opt_vr=None, tri_interface=False, log=print):
    t0 = time.time()
    mesh = read_polymesh(case)
    log(f"read {mesh['dir']}: {len(mesh['points']):,} points, {len(mesh['foff']) - 1:,} faces, "
        f"{len(mesh['patches'])} patches, cellZones {list(mesh['zones'])}  ({time.time() - t0:.1f} s)")
    res = convert_mesh(mesh, apex=apex, patterns=patterns, nproc=nproc, tri_interface=tri_interface, log=log)
    out = Path(out)
    report = {"case": str(case), "su2": str(out), **res["stats"], "parts": {}}
    n_conf = conformal_zone_faces(mesh)
    multizone = (not single_zone) and any(len(v) for v in mesh["zones"].values()) and n_conf == 0
    if n_conf and not single_zone:
        log(f"cellZone boundary is CONFORMAL in OpenFOAM ({n_conf:,} internal faces between zones) → one SU2 zone; "
            f"for an SU2 zone per cellZone baffle it in OpenFOAM first (createBaffles + mergeOrSplitBaffles -split)")
    pairs = [tuple(merge)] if merge else ([] if (multizone or not auto_merge) else coupled_pairs(mesh["dir"]))
    report["merged_pairs"] = {}
    for a, b in pairs:
        try:
            report["merged_pairs"][f"{a}+{b}"] = merge_patches(res, mesh, a, b)
            log(f"glued coupled patches {a} + {b} (max point gap {report['merged_pairs'][f'{a}+{b}']:.3g}) → one zone")
        except ValueError as err:
            if merge:
                raise
            log(f"NOT glued {a} + {b}: {err} → kept as markers (non-conformal interface: use SU2 multizone)")
    if opt_apex or opt_vr:
        t1 = time.time()
        q0 = su2_quality(res["X"], res["nodes"], res["types"], res["bfaces"], res["bpatch"])
        report["point_optimisation"] = []
        if opt_vr:
            report["point_optimisation"].append(optimise_apex(res, len(mesh["points"]), thr=float(opt_vr), target="vr",
                                                              q=q0, log=log))
        if opt_apex:
            q1 = q0 if not opt_vr else None
            report["point_optimisation"].append(optimise_apex(res, len(mesh["points"]), thr=float(opt_apex),
                                                              target="orth", q=q1, log=log))
        log(f"  point optimisation ({time.time() - t1:.1f} s)")
    t1 = time.time()
    orient = su2_relabel(res["X"], res["nodes"], res["types"])           # SU2's own orientation tests (no flips)
    log(f"  SU2 orientation: {orient['relabelled']:,} elements relabelled; flip {orient['flip']} "
        f"(before {orient['flip_before']}), distorted {orient['distorted']:,}  ({time.time() - t1:.1f} s)")
    info = mesh_info(mesh, res, multizone, n_conf=n_conf if not single_zone else 0)
    if res["stats"].get("tri_interface_faces"):
        info.setdefault("modified", []).append(
            f"--tri-interface: {res['stats']['tri_interface_faces']} interface faces split into triangles (same on both "
            f"sides), {res['stats']['tri_interface_cells']} cells touching them split by a centre point (pyramids + tets)")
    X = res["X"]
    if multizone:
        parts = zone_split(mesh, res, info)
    else:
        sel = np.ones(len(res["types"]), bool)
        parts = [{"name": "all", "sel": sel, "bfaces": res["bfaces"], "bpatch": res["bpatch"]}]
    names = info["_marker_names"]
    with open(out, "w") as fh:
        if multizone:
            fh.write(f"NZONE= {len(parts)}\n")
        for iz, z in enumerate(parts):
            t1 = time.time()
            nodes_z, types_z = res["nodes"][z["sel"]], res["types"][z["sel"]]
            nfix, bq = su2_fix_boundary_flips(X, nodes_z, types_z, z["bfaces"])
            res["nodes"][z["sel"]] = nodes_z
            fl, di = su2_flip_distorted(X, nodes_z, types_z, relabel=False)
            zo = {"flip": int(fl.sum()), "distorted": int(di.sum()), "boundary_quads_flip": bq["flip"],
                  "boundary_quads_distorted": bq["distorted"], "boundary_quads_fixed_by_relabel": nfix}
            npts = write_su2(None, X, types_z, nodes_z, z["bfaces"], z["bpatch"], names,
                             zone_header=f"IZONE= {iz + 1}\n" if multizone else None, fh=fh)
            sub = {"X": X, "types": types_z, "nodes": nodes_z, "bfaces": z["bfaces"]}
            chk = check(sub)
            q = su2_quality(X, nodes_z, types_z, z["bfaces"], z["bpatch"])
            part = {"points": npts, **chk, "markers": {names[i]: int((z["bpatch"] == i).sum())
                                                       for i in np.unique(z["bpatch"])},
                    "su2_quality": su2_quality_summary(q), "su2_orientation": zo}
            if not multizone:
                part["bad_points"] = bad_points(res, len(mesh["points"]), q)
                res["quality"] = q
            report["parts"][z["name"]] = part
            log(f"  zone {z['name']}: {npts:,} points, {len(types_z):,} elements, markers {list(part['markers'])} "
                f"({time.time() - t1:.1f} s)")
    report["parts_order"] = [z["name"] for z in parts]
    info.pop("_marker_names", None)
    report["info"] = info
    Path(str(out) + ".info.json").write_text(json.dumps(info, indent=1))
    report["_res"] = res
    report["seconds"] = round(time.time() - t0, 2)
    Path(str(out) + ".json").write_text(json.dumps({k: v for k, v in report.items() if k != "_res"}, indent=1))
    return report


# ============================================================================================ --check (any .su2 file)
def read_su2(path):
    """Single-zone SU2 ASCII mesh → X, types, nodes(n,8), bfaces(n,4; -1 pad), bpatch, marker names."""
    L = open(path).read().split("\n")
    i, X, el, bf, bp, names = 0, None, [], [], [], []
    while i < len(L):
        s = L[i].split("%")[0].strip()
        if s.startswith("NZONE="):
            raise SystemExit("--check reads single-zone .su2 files only")
        if s.startswith("NELEM="):
            n = int(s.split("=")[1]); el = [list(map(int, l.split())) for l in L[i + 1:i + 1 + n]]; i += n
        elif s.startswith("NPOIN="):
            n = int(s.split("=")[1].split()[0])
            X = np.array([l.split()[:3] for l in L[i + 1:i + 1 + n]], float); i += n
        elif s.startswith("MARKER_TAG="):
            names.append(s.split("=")[1].strip()); n = int(L[i + 1].split("=")[1])
            for l in L[i + 2:i + 2 + n]:
                v = list(map(int, l.split()))[1:]
                bf.append((v + [-1])[:4]); bp.append(len(names) - 1)
            i += n + 1
        i += 1
    types = np.array([e[0] for e in el])
    nodes = np.full((len(el), 8), -1, np.int64)
    for k, e in enumerate(el):
        nodes[k, :NODES[e[0]]] = e[1:1 + NODES[e[0]]]
    return X, types, nodes, np.array(bf, np.int64).reshape(-1, 4), np.array(bp, np.int64), names


def read_su2_vtk(path, npnt):
    """SU2 legacy ASCII VTK (PARAVIEW_ASCII): POINTS + SCALARS, parsed as text (SU2 v8.5.0 writes 'nan', which VTK's
    reader rejects)."""
    L = open(path).read().split("\n")
    Xs, ref = None, {}
    for i, line in enumerate(L):
        if line.startswith("POINTS"):
            n = int(line.split()[1]); vals = []; j = i + 1
            while len(vals) < 3 * n:
                vals += L[j].split(); j += 1
            Xs = np.array(vals, float).reshape(n, 3)
        elif line.startswith("SCALARS"):
            vals = []; j = i + 2
            while j < len(L) and not L[j][:1].isalpha() and len(vals) < npnt:
                vals += L[j].split(); j += 1
            ref[line.split()[1].lower().replace("_", "")] = np.array(vals, float)
    return Xs, ref


def check_su2(path, vtk=None):
    X, types, nodes, bfaces, bpatch, names = read_su2(path)
    vol = element_volumes(X, nodes, types)
    q = su2_quality(X, nodes, types, bfaces, bpatch)
    out = {"su2": str(path), "points": int(len(X)), "elements": {n: int((types == t).sum()) for t, n in
           ((VTK_HEX, "hex"), (VTK_PRISM, "prism"), (VTK_PYR, "pyramid"), (VTK_TET, "tet"))},
           "markers": names, "n_nonpositive": int((vol <= 0).sum()), "su2_quality": su2_quality_summary(q)}
    fl, di = su2_flip_distorted(X, nodes, types, relabel=False)
    bf_of = bfaces.copy(); qm = bf_of[:, 3] >= 0
    bf_of[qm] = bf_of[qm][:, ::-1]                                       # back to OpenFOAM winding (function expects it)
    bq = su2_boundary_flips(X, nodes, types, bf_of)
    out["su2_orientation"] = {"flip": int(fl.sum()), "distorted": int(di.sum()),
                              "boundary_quads_flip": bq["flip"], "boundary_quads_distorted": bq["distorted"]}
    if vtk:
        Xs, ref = read_su2_vtk(vtk, len(X))
        cmp_ = {}
        for lab, key, mine in (("orthogonality", "orthogonality", q["orth"]), ("aspect_ratio", "aspectratio", q["ar"]),
                               ("volume_ratio", "volumeratio", q["vr"])):
            r = ref[key]; ok = np.isfinite(r) & np.isfinite(mine)
            cmp_[lab] = float((np.abs(mine[ok] - r[ok]) / np.maximum(np.abs(r[ok]), 1e-30)).max())
        out["compare_with_su2"] = {"max_rel_diff": cmp_, "coord_diff": float(np.abs(Xs - X).max()),
                                   "match": max(cmp_.values()) < 1e-6}
    return out


def _summary(rep):
    """Short human-readable verdict (per zone), interfaces, and everything NOT carried over."""
    lines = []
    for pair, gap in rep.get("merged_pairs", {}).items():
        lines.append(f"glued coupled patches {pair} into one zone (gap {gap:.2g})")
    parts = rep.get("parts") or {"all": rep}
    for zname in rep.get("parts_order", list(parts)):
        part = parts[zname]
        el = part.get("elements")
        el = f"{el:,} elements" if isinstance(el, int) else ", ".join(f"{v:,} {k}" for k, v in (el or {}).items() if v)
        tag = "" if zname == "all" else f"zone {zname}: "
        lines.append(f"{tag}{part.get('points', 0):,} points, {el}, markers {', '.join(part.get('markers', {}))}")
        valid = part.get("n_nonpositive", 1) == 0 and part.get("faces_used_more_than_twice", 0) == 0 and \
            part.get("faces_once_not_on_marker_or_reverse", 0) == 0
        lines.append(f"  valid: {'YES' if valid else 'NO'} (non-positive elements {part.get('n_nonpositive')}, "
                     f"unmatched faces {part.get('faces_once_not_on_marker_or_reverse', '-')})")
        sq = part.get("su2_quality", {})
        if sq:
            lines.append(f"  SU2 orthogonality min {sq['orth_min_deg']:.1f}° (points < 20°: {sq['points_orth_lt_20']}, "
                         f"< 25°: {sq['points_orth_lt_25']}), CV face AR max {sq['ar_max']:.0f}, "
                         f"sub-volume ratio max {sq['vr_max']:.0f}")
        so = part.get("su2_orientation")
        if so:
            lines.append(f"  SU2 orientation check (as SU2 v8.5.0 does it): flips {so['flip']}, distorted "
                         f"{so['distorted']:,}; boundary quads flips {so['boundary_quads_flip']}, distorted "
                         f"{so['boundary_quads_distorted']}")
        b = part.get("bad_points")
        if b and b["points"]:
            lines.append(f"  points < {b['threshold_deg']:.0f}°: {b['added_centre_points']} added centre points, "
                         f"{b['original_points']} original points → try --opt-apex 40 / --opt-vr 1000")
    info = rep.get("info", {})
    for it in info.get("interfaces", []):
        lines.append(f"interface {it['zones'][0]} | {it['zones'][1]}: markers {it['markers'][0]} / {it['markers'][1]} "
                     f"({it['kind']}" + (f", faceZone {it['faceZone']}" if it.get("faceZone") else "") + ")")
    for t in info.get("faceZone_notes", []):
        lines.append(t)
    if info.get("renamed"):
        lines.append(f"renamed for SU2: {info['renamed']}")
    for t in info.get("modified", []):
        lines.append(f"MODIFIED: {t}")
    for t in info.get("not_carried_over", []):
        lines.append(f"NOT carried over: {t}")
    if "compare_with_su2" in rep:
        lines.append(f"vs SU2 output: {'MATCH' if rep['compare_with_su2']['match'] else 'MISMATCH'} "
                     f"{rep['compare_with_su2']['max_rel_diff']}")
    if info:
        lines.append(f"mesh information (patch types, groups, zones, interfaces, limits): {rep.get('su2', '')}.info.json")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", help="OpenFOAM case folder (or its constant/polyMesh folder)")
    ap.add_argument("out", nargs="?", help="output .su2 file (report: <out>.json)")
    ap.add_argument("--check", metavar="MESH.su2", help="score an existing .su2 file instead of converting")
    ap.add_argument("--vtk", metavar="flow.vtk", help="with --check: compare against SU2's MESH_QUALITY output")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--merge", nargs=2, metavar=("A", "B"), help="glue this conformal patch pair (default: all "
                   "cyclicAMI/cyclic pairs are glued automatically when their points coincide)")
    g.add_argument("--no-merge", action="store_true", help="keep coupled patches as separate markers")
    ap.add_argument("--single-zone", action="store_true", help="one SU2 zone even if the mesh has cellZones (default: one SU2 "
                    "zone per cellZone + rest when OpenFOAM baffled the zone boundary; conformal boundary → one zone)")
    ap.add_argument("--opt-apex", type=float, metavar="DEG", default=None,
                    help="move added centre points whose SU2 orthogonality < DEG (e.g. 40) to raise it; never makes "
                         "aspect ratio or volume ratio worse (slow: ~4 min per 5,000 polyhedra)")
    ap.add_argument("--opt-vr", type=float, metavar="RATIO", default=None,
                    help="move added centre points of cells next to points with SU2 sub-volume ratio > RATIO (e.g. 1000) "
                         "to lower it; never makes orthogonality or aspect ratio worse")
    ap.add_argument("--apex", choices=("centroid", "kernel"), default="centroid", help=argparse.SUPPRESS)
    ap.add_argument("--no-patterns", action="store_true", help="do not use the MixedOcTree hanging-node patterns "
                    "(centre-point split for every hanging-node hex; for comparison)")
    ap.add_argument("--tri-interface", action="store_true",
                    help="split zone-interface quads/polygons into triangles (identical on both sides) and the cells "
                         "touching them by a centre point, so SU2 KIND_INTERPOLATION= WEIGHTED_AVERAGE works "
                         "(it fails on quad interfaces); changes only the cells at the interface")
    ap.add_argument("--nproc", type=int, default=None, help="worker processes (default: all CPUs; 1 = serial)")
    ap.add_argument("--json", action="store_true", help="print the full report instead of the summary")
    ap.add_argument("--version", action="version", version=f"of2su2 {__version__}")
    a = ap.parse_args(argv)
    if a.check:
        rep = check_su2(a.check, a.vtk)
    elif a.case and a.out:
        rep = convert(a.case, a.out, a.single_zone, a.merge, a.apex, a.opt_apex, auto_merge=not a.no_merge,
                      patterns=not a.no_patterns, nproc=a.nproc, opt_vr=a.opt_vr, tri_interface=a.tri_interface)
        rep = {k: v for k, v in rep.items() if k != "_res"}
    else:
        ap.error("give <case> <out.su2> to convert, or --check <mesh.su2>")
    print(json.dumps(rep, indent=1) if a.json else _summary(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
