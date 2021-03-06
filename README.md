# *Building(s and) Cities*

[**Paper**](https://www.sciencedirect.com/science/article/pii/S0094119019300944) | [**Working Paper**](https://ieb.ub.edu/en/publication/2019-10-buildings-and-cities-delineating-urban-areas-with-a-machine-learning-algorithm/) | [**Map**](https://gdsl.carto.com/u/danigdsl/builder/fe01a0ff-f0ee-4df2-ba4a-a46815b87aab/embed) | [**Data**](https://figshare.com/articles/Building_s_and_cities_Delineating_urban_areas_with_a_machine_learning_algorithm_-_City_Employment_Centre_Boundaries_v1_/11384136)

![Cities](boundaries.png)

Repository for the paper ["Building(s and) cities: Delineating urban areas with a machine learning algorithm"](https://www.sciencedirect.com/science/article/pii/S0094119019300944), forthcoming in the *Journal of Urban Economics*.

Authors:

- [Dani Arribas-Bel](https://darribas.org) [`[@darribas]`](https://twitter.com/darribas)
- [Miquel-Angel Garcia-Lopez](http://gent.uab.cat/miquelangelgarcialopez/ca/content/home) [`[@MiquelAngelGL]`](https://twitter.com/MiquelAngelGL)
- [Elisabet Viladecans-Marsal](https://elisabetviladecans.wordpress.com/) [`[@eviladecans]`](https://twitter.com/eviladecans)

## Citation

If you use the code and/or data in this repository, we would appreciate if you
could cite the original paper as:

```
@article{ab_gl_vm2019joue,
  title={Building(s and) cities: 
         Delineating urban areas with a machine learning algorithm},
  author={Arribas-Bel, Daniel and 
          Garcia-L\`{o}pez, Miquel-\'{A}ngel and
          Viladecans-Marsal, Elisabet},
  journal={Journal of Urban Economics},
  year={\emph{forthcoming}},
}
```

## Interactive Map

You can explore the city delineations in the interactive map (click on the
image):

[![Map](map.png)](https://danigdsl.carto.com/builder/fe01a0ff-f0ee-4df2-ba4a-a46815b87aab/embed?state=%7B%22map%22%3A%7B%22ne%22%3A%5B31.738241084411285%2C-19.36323165893555%5D%2C%22sw%22%3A%5B46.2944086152281%2C9.377002716064455%5D%2C%22center%22%3A%5B39.3944178910168%2C-4.993114471435548%5D%2C%22zoom%22%3A6%7D%7D)

## Data

City and employment centre boundary delineations are available as open data
at:

> [https://figshare.com/articles/Building_s_and_cities_Delineating_urban_areas_with_a_machine_learning_algorithm_-_City_Employment_Centre_Boundaries_v1_/11384136](https://figshare.com/articles/Building_s_and_cities_Delineating_urban_areas_with_a_machine_learning_algorithm_-_City_Employment_Centre_Boundaries_v1_/11384136)

## Computational Platform

The analysis in this repository was performed on machines running the `gds_env` platform,
v3.0. The project's website is available at:

> [https://github.com/darribas/gds_env](https://github.com/darribas/gds_env)

And you can install the version used for this projec by first installing
[Docker](https://www.docker.com/) (the community edition, CE, works fine) and
then running:

```
docker pull darribas/gds:3.0
```

You can read more background about platforms like `gds_env` in [Chapter 1](https://geographicdata.science/book/notebooks/01_geospatial_computational_environment.html) of the upcoming book in Geographic Data Science.

