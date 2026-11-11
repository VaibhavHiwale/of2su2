# rotor_cut — regression mesh

2,441 cells cut from a 13.7 M-cell snappyHexMesh rotor mesh: up to 5 cells of **every** cell shape in that mesh that SU2
cannot use (64 shapes) plus their face neighbours; 13,383 points. Faces exposed by the cut form the patch `cut`; the
original patches are kept (coupled patches written as plain `patch`). It cannot be regenerated with OpenFOAM, so the
polyMesh is tracked.
