# Code

This folder contains the following elements:

- `01_extract_merge_buildings.ipynb`: notebook with code to extract buildings
  from original Cadastre files
- `02_delineate_cities.ipynb`: notebook with steps to delineate city
  boundaries
- `03_identify_employment_centres.ipynb`: notebook with code to identify
  employment centres from building data in each city
- `04_join_labels.ipynb`: notebook merging results from our delineation with
  Municipality and AUDES delineations at the building data
- `05_paper_figures.ipynb`: notebook with code to replicate figures presented
  in the paper
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
