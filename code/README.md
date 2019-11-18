# Code

This folder contains the following elements:

- `alpha_shapes_2D.py`: the original module used to draw alpha shapes in the paper. It
  is based on an original (3D) [implementation](https://github.com/timkittel/alpha-shapes) 
  by Tim Kittel, adapted by Dani Arribas-Bel. The 2D functionality used here
  has now been [contributed to
  PySAL](https://github.com/pysal/pysal/blob/master/pysal/lib/cg/alpha_shapes.py)
  and that is the recommended implementation for future use.
- `tools.py`: collection of utilities written for the project. The module also
  contains the initial implementation of A-DBSCAN used to generate the
  published results. A later implementation has been contributed to PySAL and
  represents the preferrable choice for future use.
