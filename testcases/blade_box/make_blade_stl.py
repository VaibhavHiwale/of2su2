"""Blade STL for the blade_box test mesh: box chord 0.15 x span 0.55 (y 0.15-0.70) x thickness 0.02 m, 8 deg pitch about y.
Regions: wall (4 long faces), root_cap (y=0.15), tip_cap (y=0.70) (caps get no wall layers)."""
import numpy as np, pathlib
c, t, y0, y1, a = 0.15, 0.02, 0.15, 0.70, np.radians(8.0)
R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
def P(x, y, z): return R @ np.array([x, y, z])
xs, zs = (-c / 2, c / 2), (-t / 2, t / 2)
corner = {(i, j, k): P(xs[i], (y0, y1)[j], zs[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
def quad(a_, b_, c_, d_):  # outward order given
    return [(a_, b_, c_), (a_, c_, d_)]
q = lambda *ks: quad(*[corner[k] for k in ks])
solids = {
    "wall": q((0,0,0),(1,0,0),(1,1,0),(0,1,0)) + q((0,0,1),(0,1,1),(1,1,1),(1,0,1))      # lower / upper
          + q((0,0,0),(0,1,0),(0,1,1),(0,0,1)) + q((1,0,0),(1,0,1),(1,1,1),(1,1,0)),     # LE / TE
    "root_cap": q((0,0,0),(0,0,1),(1,0,1),(1,0,0)),
    "tip_cap":  q((0,1,0),(1,1,0),(1,1,1),(0,1,1)),
}
out = []
for name, tris in solids.items():
    out.append(f"solid {name}")
    for p0, p1, p2 in tris:
        n = np.cross(p1 - p0, p2 - p0); n /= np.linalg.norm(n)
        out += [f" facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}", "  outer loop"]
        out += [f"   vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in (p0, p1, p2)]
        out += ["  endloop", " endfacet"]
    out.append(f"endsolid {name}")
pathlib.Path("constant/triSurface/blade.stl").write_text("\n".join(out) + "\n")
