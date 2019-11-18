"""
Utilities and original A-DBSCAN implementation
...

Copyright 2019 Dani Arribas-Bel
"""

import os, math, time
import pandas as pd
import numpy as np
import geopandas as gpd
import alpha_shapes_2D as as2
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
from collections import Counter
from zipfile import ZipFile
from shapely.wkb import loads
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPoint, MultiLineString
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

def write_geoparquet(gdf, url, multi_cpu=True):
    '''
    Serialise a GeoDataFrame into a `parquet` file
    
    NOTE: it does NOT save CRS info
    ...
    
    Arguments
    ---------
    gdf     : GeoDataFrame
              Table to be written
    url     : str
              Path to file to be written
    multi_cpu : Boolean
              [Optional. Default=True] If True, serialising 
              of wkb is done in parallel
   
    Returns
    -------
    url     : str
              Path to file to be written
    '''
    if multi_cpu:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count())
        geos_chunked = np.array_split(gdf['geometry'], mp.cpu_count())
        new_geom = pd.concat(pool.map(_serialiser, geos_chunked))
    else:
        new_geom = gdf['geometry'].apply(lambda x: x.wkb_hex)
    gdf.drop('geometry', axis=1)\
       .assign(geometry=new_geom)\
       .to_parquet(url)
    return url

def _serial(g):
    return g.wkb_hex
def _serialiser(s):
    return s.apply(_serial)

def read_geoparquet(url, qry=None, crs=None, multi_cpu=True):
    '''
    De-serialise a `parquet` file into a `GeoDataFrame`
    
    NOTE: it does NOT fill the `crs` attribute unless explicitly passed
    ...
    
    Arguments
    ---------
    url     : str
              Path to file to be written
    qry     : str
              [Optional. Default=None] Query to filter rows before parsing geometries
    crs     : str
              [Optional. Default=True] CRS info for loaded GeoDataFrame
    multi_cpu : Boolean
              [Optional. Default=True] If True, serialising 
              of wkb is done in parallel    
    Returns
    -------
    gdf     : GeoDataFrame
              Table to be written
    '''
    gdf = pd.read_parquet(url)
    if qry is not None:
        gdf = gdf.query(qry)
    if multi_cpu:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count())
        hexs_chunked = np.array_split(gdf['geometry'], mp.cpu_count())
        gdf['geometry'] = pd.concat(pool.map(_deserialiser, hexs_chunked))
    else:
        gdf['geometry'] = gdf['geometry'].apply(lambda x: loads(x, hex=True))
    gdf = gpd.GeoDataFrame(gdf)
    if crs is not None:
        gdf.crs = crs
    return gdf

def _deserial(h):
    return loads(h, hex=True)
def _deserialiser(s):
    return s.apply(_deserial)

def p_sjoin(l_gdf, r_gdf, n_jobs=None, verbose=True):
    '''
    Parallel version of `geopandas.sjoin`
    '''
    import multiprocessing as mp
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)
    pts_polys_chunked = [(chunk, r_gdf, verbose) for chunk in \
                         np.array_split(l_gdf, n_jobs)]
    if verbose:
        print(f"Chunks created, spinning up {n_jobs} parallel tasks...")
    chunks = pool.map(joiner, pts_polys_chunked)
    pool.close()
    joined = pd.concat(chunks)
    return joined

def joiner(pts_polys_verb):
    pts, polys, verbose = pts_polys_verb
    if verbose:
        print(f"Working on rows {pts.index[0]} to {pts.index[-1]}")
    joined = gpd.sjoin(pts, 
                       polys,
                       how='left')
    if verbose:
        print(f"\tCompleted rows {pts.index[0]} to {pts.index[-1]}")
    return joined

def extract_geoms(f, ext='.building.gml', encode_geom=True,
                  t_crs=25830):
    '''
    Extract geometries from compressed `.gml` file and append X/Y coordinates
    
    NOTE: requires `ogr2ogr` installed
    ...
    
    Arguments
    ---------
    f       : str
              Path to compressed file
    
    Returns
    -------
    gdf     : GeoDataFrame
              Geotable with information from `.gml` file + X/Y coordinates in 
              separate columns
    '''
    fn = f.split('/')[-1].replace('.zip', '')
    _ = ZipFile(f, 'r').extract(fn + ext)
    cmd = "ogr2ogr -f 'GeoJSON' %s.geojson %s"%\
          (fn, fn+ext)
    os.system(cmd)
    db = gpd.read_file(fn+'.geojson')
    db = db.to_crs(epsg=t_crs)
    xys = pd.DataFrame([[pt.x, pt.y] for pt in db.centroid],
                       columns=['X', 'Y'])
    cmd = "rm %s.*"%fn
    os.system(cmd)
    if encode_geom:
        db['geometry'] = db['geometry'].apply(lambda g: g.to_wkb())
        db = pd.DataFrame(db)
    return db.join(xys)

def parse_municipalities(p):
    '''
    Parse xml file to extract municipalities
    ...
    
    Attributes
    ----------
    p       : str
              Path to file to parse
    
    Returns
    -------
    list    : Series
              Series with Municipality names indexed on Municipality code
    '''
    lines = open(p).readlines()
    s = {}
    for line in lines:
        try:
            int(line[:5])
            code, name = line.split('  ')[0].split('-')
            s[code] = name
        except:
            pass
    return pd.Series(s, name='code2name')

def load_municipality(engine, code=None, name=None, 
                      varlist=None, polys=True):
    '''
    Load a given municipality
    ...
    
    Arguments
    ---------
    engine  : Engine
    code    : str
    name    : str
    vars    : None/list
    
    Returns
    -------
    tab     : (Geo)DataFrame
    '''
    if not code:
        code = pd.read_sql("""
                           SELECT code \
                           FROM code2name \
                           WHERE name=='%s'
                           """%name,
                           engine)\
                 .loc[0, 'code']
    code = "A.ES.SDGC.BU." + str(code)
    if varlist is None:
        varlist = '*'
    else:
        varlist = ', '.join(varlist)
    tab = pd.read_sql("""
                      SELECT %s \
                      FROM cadastro \
                      WHERE file=='%s' \
                      """%(varlist, code),
                      engine)
    if 'geometry' in tab:
        tab = _build_geodf(tab, polys=polys)
    return tab

def load_bbox(engine, bbox, 
              varlist=None, polys=True):
    '''
    Load a given bounding box
    ...
    
    Arguments
    ---------
    engine  : Engine
    bbox    : list
              [minX, minY, maxX, maxY]
    vars    : None/list
    
    Returns
    -------
    tab     : (Geo)DataFrame
    '''

    if varlist is None:
        varlist = '*'
    else:
        varlist = ', '.join(varlist)
    tab = pd.read_sql("""
                      SELECT %s \
                      FROM cadastro \
                      WHERE (X > '%f') & \
                            (Y > '%f') & \
                            (X < '%f') & \
                            (Y < '%f')
                      """%(varlist, *bbox),
                      engine)
    if 'geometry' in tab:
        tab = _build_geodf(tab, polys=polys)
    return tab


def _build_geodf(tab, 
                 crs={'init' :'epsg:25830'}, 
                 polys=True):
    tab['geometry'] = tab['geometry'].apply(lambda x: loads(x, hex=False))
    
    tab = gpd.GeoDataFrame(tab, crs=crs)
    if not polys:
        tab['geometry'] = tab['geometry'].centroid    
    return tab


def lbls2polys(lbls, xys, noise='-1', 
               xy=['X', 'Y'], epsg=25830, 
               step=1, ignore=True, gdf=False):
    '''
    Convert `xys` points into polygons following groupings in `lbls`
    ...

    Arguments
    ---------
    lbls    : Series
              Labels with cluster membership
    xys     : DataFrame
    noise   : str/int/float
              Label used to signify non-membership
    xy      : list
              [Default=`['X', 'Y']`] Pair of column names in `xys`
              with coordinates
    epsg    : int
              [Default=25830] CRS of points
    step    : int
              [Default=1] Step for auto alpha shape
    ignore  : Boolean
              [Default=True] If True, remove clusters with less than four
              points (so every cluster can be turned into a valid polygon)
    gdf     : Boolean
              [Default=False] If True, return a `GeoDataFrame` with a column 
              polygon ID, and another one with a count of points

    Returns
    -------
    polys/gdf: GeoSeries/GeoDataFrame
    '''
    lbl_type = type(lbls.iloc[0])
    new_lbls = lbls
    if ignore:
        counts = lbls.groupby(lbls).size()
        not_polys = counts[counts<4].index.values
        if not_polys.shape[0] > 0:
            print('Ignoring %i clusters with less than 4 members'%not_polys.shape[0])
            new_lbls = lbls.replace(not_polys, noise)
    # If any clusters
    if new_lbls.unique().shape[0] > 1:
        cl_lbls = new_lbls[new_lbls != noise]
        polys = xys.loc[cl_lbls.index, xy]\
                   .groupby(cl_lbls)\
                   .apply(lambda xys:    
                          as2.alpha_shape_auto(xys.values,
                                               step=step,
                                               verbose=False))
        polys = gpd.GeoSeries(polys, crs={'init' :'epsg:%i'%epsg})
    else:
        polys = gpd.GeoSeries([], crs={'init' :'epsg:%i'%epsg})
    if gdf:
        sizes = xys.groupby(lbls).size().drop(lbl_type(-1), errors='ignore')
        gdf = gpd.GeoDataFrame({'geometry': polys, 'n_pts': sizes}, 
                               crs=polys.crs)\
                 .dropna()\
                 .reset_index() 
        return gdf
    else:
        return polys
    
def lbls2polys_parallel(lbls, xys, noise='-1', 
                        xy=['X', 'Y'], epsg=25830, 
                        step=1, ignore=True, gdf=False):
    '''
    Convert `xys` points into polygons following groupings in `lbls`
    ...

    Arguments
    ---------
    lbls    : Series
              Labels with cluster membership
    xys     : DataFrame
    noise   : str/int/float
              Label used to signify non-membership
    xy      : list
              [Default=`['X', 'Y']`] Pair of column names in `xys`
              with coordinates
    epsg    : int
              [Default=25830] CRS of points
    step    : int
              [Default=1] Step for auto alpha shape
    ignore  : Boolean
              [Default=True] If True, remove clusters with less than four
              points (so every cluster can be turned into a valid polygon)
    gdf     : Boolean
              [Default=False] If True, return a `GeoDataFrame` with a column 
              polygon ID, and another one with a count of points

    Returns
    -------
    polys/gdf: GeoSeries/GeoDataFrame
    '''
    import multiprocessing as mp
    lbl_type = type(lbls.iloc[0])
    noise_adj = lbl_type(noise)
    new_lbls = lbls
    if ignore:
        counts = lbls.groupby(lbls).size()
        not_polys = counts[counts<4].index.values
        if not_polys.shape[0] > 0:
            print('Ignoring %i clusters with less than 4 members'%not_polys.shape[0])
            new_lbls = lbls.replace(not_polys, noise_adj)
    # If any clusters
    if new_lbls.unique().shape[0] > 1:
        cl_lbls = new_lbls[new_lbls != noise_adj]
        # Chunk xys and cl_lbls
        tmp = xys.assign(cl_lbls=cl_lbls)\
                 .dropna()
        gtmp = tmp.groupby('cl_lbls')\
                  .groups
        id_order = list(gtmp.keys())
        chunks_steps = [(tmp.loc[gtmp[i], xy], step) \
                        for i in id_order]
        # Span multi-processes
        pool = mp.Pool(mp.cpu_count())
        polys = pool.map(_polygonize, chunks_steps)
        pool.close()
        polys = gpd.GeoSeries(polys, crs={'init' :'epsg:%i'%epsg},
                              index=id_order)
    else:
        polys = gpd.GeoSeries([], crs={'init' :'epsg:%i'%epsg})
    if gdf:
        sizes = xys.groupby(lbls).size().drop(lbl_type(-1), errors='ignore')
        gdf = gpd.GeoDataFrame({'geometry': polys, 'n_pts': sizes}, 
                               crs=polys.crs)\
                 .dropna()\
                 .reset_index() 
        return gdf
    else:
        return polys
    
def _polygonize(tmp_step):
    tmp, step = tmp_step
    if tmp.shape[0] < 4:
        return None
    else:
        poly = as2.alpha_shape_auto(tmp.values,
                                    step=step,
                                    verbose=False)
        return poly
    
def shade_pts(xys, z=None, w=600, h=600, x='X', y='Y', how='log',
        z_how=ds.mean, figsize=(9, 9), cmap='magma', alpha=1, ax=None):
    '''
    Rasterise a point pattern
    ...

    Attributes
    ----------
    xys
    z
    w
    h
    x
    y
    how
    figsize
    cmap
    ax
    '''
    cvs = ds.Canvas(plot_width=w, plot_height=h)
    if z:
        agg = cvs.points(xys, x, y, z_how(z))
    else:
        agg = cvs.points(xys, x, y)
    img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how=how)

    a = np.flipud(img.values.astype(float))
    mask = np.where(a==0)
    a[mask] = np.nan

    ext = xys[[x, y]].describe()\
                     .loc[['min', 'max'], :]\
                     .T\
                     .stack()\
                     .values

    plot = False
    if not ax:
        f, ax = plt.subplots(1, figsize=figsize)
        plot = True
    ax.imshow(a, cmap=cmap, alpha=alpha, extent=ext)
    if plot:
        return plt.show()
    else:
        return ax

def dbs_params(dens=None, eps=None, min_pts=None):
    '''
    Calculate DBSCAN parameters (eps, min_pts) and density triplets. 
    
    NOTE: Two parameters are required.
    ...
    
    Arguments
    ---------
    dens        : float/None
                  [Optional. Default=None] Minimum density required
    eps         : float/None
                  [Optional. Default=None] Maximum radious required
    min_pts     : float/None
                  [Optional. Default=None] Minimum amount of points required
    
    Returns
    -------
    params      : tuple
                  (dens, eps, min_pts)
    '''
    if eps and min_pts:
        dens = min_pts / (np.pi * eps**2)
    elif dens and eps:
        min_pts = int(np.round(dens * np.pi * eps**2))
    elif dens and min_pts:
        eps = np.sqrt(min_pts / (np.pi * dens))
    else:
        return print(("Please pass at least two parameters of "\
                      "(dens, eps, min_pts)"))
    return (dens, eps, min_pts)
    
def identify_city_centres(tab, log_file=None, 
                          dens=None, eps=None, min_pts=None,
                          reps=100,
                          gpkg_out=None, feather_out=None,
                          xy=['X', 'Y'], lbls='lbls',
                          units='numberOfBuildingUnits',
                          step=50):
    '''
    Identify centres and return as polygons (together with centre ID, city ID, centre size)
    ...
    
    Parameters
    ----------
    tab             : DataFrame
                      Table with coordinates and labels
    log_file        : None/str
                      [Optional. Default=None]
    dens            : float/None
                      [Optional. Default=None]
    eps             : float/None
                      [Optional. Default=None]
    min_pts         : int/None
                      [Optional. Default=None]
    reps            : int
                      [Optional. Default=100]
    gpkg_out        : str/None
                      [Optional. Default=None]
    feather_out     : str/None
                      [Optional. Default=None]
    xy              : list
                      [Optional. Default=['X', 'Y']]
    lbls            : str
                      [Optional. Default='lbls']
    units           : str
                      [Optional. Default='numberOfBuildingUnits']
    step            : int
                      [Optional. Default=50]
    
    Returns
    -------
    cnts        : GeoDataFrame
    '''
    t0 = time.time()
    dens, eps, min_pts = dbs_params(eps=eps, 
                                    dens=dens,
                                    min_pts=min_pts)
    ec = ADBSCAN(eps=eps, min_samples=min_pts,
                 pct_exact=0.5, reps=reps)
    if units:
        ec.fit(tab[xy], sample_weight=tab[units])
    else:
        ec.fit(tab[xy])
    lbls = pd.Series(ec.labels_.astype(str),
                     index=tab.index)
    cnts = lbls2polys(lbls, tab, step=step, gdf=True)
    cnts = cnts.rename(columns={'index': 'centre_id'})
    cnts['area_sqm'] = cnts.area
    t1 = time.time()
    if log_file is not None:
        msg = f"City {tab['lbls'].iloc[0]} completed in {np.round(t1-t0, 3)} seconds\n"
        logger(msg, log_file)
    return cnts

def print_coords(longitude, latitude):
    #https://glenbambrick.com/2015/06/24/dd-to-dms/
    import math
    # math.modf() splits whole number and decimal into tuple
    # eg 53.3478 becomes (0.3478, 53)
    split_degx = math.modf(longitude)
    
    # the whole number [index 1] is the degrees
    degrees_x = int(split_degx[1])

    # multiply the decimal part by 60: 0.3478 * 60 = 20.868
    # split the whole number part of the total as the minutes: 20
    # abs() absoulte value - no negative
    minutes_x = abs(int(math.modf(split_degx[0] * 60)[1]))

    # multiply the decimal part of the split above by 60 to get the seconds
    # 0.868 x 60 = 52.08, round excess decimal places to 2 places
    # abs() absoulte value - no negative
    seconds_x = abs(round(math.modf(split_degx[0] * 60)[0] * 60,2))

    # repeat for latitude
    split_degy = math.modf(latitude)
    degrees_y = int(split_degy[1])
    minutes_y = abs(int(math.modf(split_degy[0] * 60)[1]))
    seconds_y = abs(round(math.modf(split_degy[0] * 60)[0] * 60,2))

    # account for E/W & N/S
    if degrees_x < 0:
        EorW = "W"
    else:
        EorW = "E"

    if degrees_y < 0:
        NorS = "S"
    else:
        NorS = "N"

    # abs() remove negative from degrees, was only needed for if-else above
    out = (f"{str(abs(degrees_x)).rjust(3)} u\u00b0 "\
           f"{str(minutes_x).rjust(3)} "\
           f"{str(seconds_x).rjust(3)} {EorW} \n"\
           f"{str(abs(degrees_y)).rjust(3)} u\u00b0 "\
           f"{str(minutes_y).rjust(3)} "\
           f"{str(seconds_y).rjust(3)} {NorS}")
    return out

                            #--------------#
                            #--- ADBSCAN --#
                            #--------------#

# IMPORTANT: this contains the original ADBSCAN implementation used in the
# paper. A more user-friendly, robust implementation is available in PySAL and
# the user is advised to rely on that implementation over this one.

class ADBSCAN():
    '''
    Approximated DBSCAN using NN Regression
    ...

    Parameters
    ----------
    eps         : float
                  The maximum distance between two samples for them to be considered
                  as in the same neighborhood.
    min_samples : int
                  The number of samples (or total weight) in a neighborhood
                  for a point to be considered as a core point. This includes the
                  point itself.
    algorithm   : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
                  The algorithm to be used by the NearestNeighbors module
                  to compute pointwise distances and find nearest neighbors.
                  See NearestNeighbors module documentation for details.
    n_jobs      : int
                  [Optional. Default=1] The number of parallel jobs to run. If
                  -1, then the number of jobs is set to the number of CPU
                  cores.
    pct_exact   : float
                  [Optional. Default=0.1] Percentage of the entire dataset
                  used to calculate DBSCAN in each draw
    reps        : int
                  [Optional. Default=100] Number of random samples to draw in order to
                  build final solution
    keep_solus  : Boolean
                  [Optional. Default=False] If True, the `solus` object is
                  kept, else it is deleted to save memory
    pct_thr     : float
                  [Optional. Default=0.9] Minimum percentage of replications that a non-noise 
                  label need to be assigned to an observation for that observation to be labelled
                  as such

    Attributes
    ----------
    labels_     : array
                  Cluster labels for each point in the dataset given to fit().
                  Noisy (if the proportion of the most common label is < pct_thr) samples are given
                  the label -1.

    votes       : DataFrame
                  Table indexed on `X.index` with `labels_` under the `lbls`
                  column, and the frequency across draws of that label under
                  `pct`
    solus       : DataFrame, shape = [n, reps]
                  Each solution of labels for every draw
    '''
    def __init__(self, eps, min_samples, algorithm='auto', n_jobs=1, pct_exact=0.1, reps=100,
                 keep_solus=False, pct_thr=0.9):
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.reps = reps
        self.n_jobs = n_jobs
        self.pct_exact = pct_exact
        self.pct_thr = pct_thr
        self.keep_solus = keep_solus

    def fit(self, X, y=None, sample_weight=None, xy=['X', 'Y'], multi_cpu=False):
        '''
        Perform ADBSCAN clustering from fetaures
        ...

        Parameters
        ----------
        X               : DataFrame
                          Features
        sample_weight   : Series, shape (n_samples,)
                          [Optional. Default=None] Weight of each sample, such
                          that a sample with a weight of at least ``min_samples`` 
                          is by itself a core sample; a sample with negative
                          weight may inhibit its eps-neighbor from being core.
                          Note that weights are absolute, and default to 1.
        xy              : list
                          [Default=`['X', 'Y']`] Ordered pair of names for XY
                          coordinates in `xys`
        y               : Ignored
        multi_cpu       : Boolean
                          [Default=False] If True, paralelise where possible
        '''
        n = X.shape[0]
        zfiller = len(str(self.reps))
        solus = pd.DataFrame(np.zeros((X.shape[0], self.reps), dtype=str),
                     index=X.index,
                     columns=['rep-%s'%str(i).zfill(zfiller) 
                              for i in range(self.reps)])
        if multi_cpu is True:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count())
            print('On multiple cores...')
            # Set different parallel seeds!!!
            raise NotImplementedError
        else:
            for i in range(self.reps):
                pars = (n, X, sample_weight, xy, \
                        self.pct_exact, self.eps, self.min_samples, \
                        self.algorithm, self.n_jobs)
                lbls_pred = _one_draw(pars)
                solus.iloc[:, i] = lbls_pred

        self.votes = ensemble(solus, X, multi_cpu=multi_cpu)
        lbls = self.votes['lbls'].values
        lbl_type = type(solus.iloc[0, 0])
        lbls[self.votes['pct'] < self.pct_thr] = lbl_type(-1)
        self.labels_ = lbls
        if not self.keep_solus:
            del solus
        else:
            self.solus = solus
        return self
    
def _one_draw(pars):
    n, X, sample_weight, xy, pct_exact, eps, min_samples, algorithm, n_jobs = pars
    rids = np.arange(n)
    np.random.shuffle(rids)
    rids = rids[:int(n*pct_exact)]

    X_thin = X.iloc[rids, :]

    thin_sample_weight = None
    if sample_weight is not None:
        thin_sample_weight = sample_weight.iloc[rids]

    dbs = DBSCAN(eps=eps, min_samples=int(np.round(min_samples*pct_exact)), 
                 algorithm=algorithm, n_jobs=n_jobs)\
          .fit(X_thin[xy], sample_weight=thin_sample_weight)
    lbls_thin = pd.Series(dbs.labels_.astype(str),
                     index=X_thin.index)

    NR = KNeighborsClassifier(n_neighbors=1)
    NR.fit(X_thin[['X', 'Y']], lbls_thin)
    lbls_pred = pd.Series(NR.predict(X[['X', 'Y']]), \
                                     index=X.index)
    return lbls_pred

def remap_lbls(solus, xys, xy=['X', 'Y'], multi_cpu=True):
    '''
    Remap labels in solutions so they are comparable (same label
    for same cluster)
    ...

    Arguments
    ---------
    solus       : DataFrame
                  Table with labels for each point (row) and solution (column)
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    multi_cpu   : Boolean
                  [Default=False] If True, paralelise remapping
    

    Returns
    -------
    onel_solus  : DataFrame
    '''
    lbl_type = type(solus.iloc[0, 0])
    # N. of clusters by solution
    ns_clusters = solus.apply(lambda x: x.unique().shape[0])
    # Pic reference solution as one w/ max N. of clusters
    ref = ns_clusters[ns_clusters==ns_clusters.max()]\
                     .iloc[[0]]\
                     .index[0]
    # Obtain centroids of reference solution
    ref_centroids = xys.groupby(solus[ref])\
                       [xy]\
                       .apply(lambda xys: xys.mean())\
                       .drop(lbl_type(-1), errors='ignore')
    # Only continue if any solution
    if ref_centroids.shape[0] > 0:
        # Build KDTree and setup results holder
        ref_kdt = cKDTree(ref_centroids)
        remapped_solus = pd.DataFrame(np.zeros(solus.shape, dtype=str),
                                      index=solus.index,
                                      columns=solus.columns)
        if multi_cpu is True:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count())
            s_ids = solus.drop(ref, axis=1).columns.tolist()
            to_loop_over = [(solus[s], ref_centroids, ref_kdt, xys, xy) \
                            for s in s_ids]
            remapped = pool.map(_remap_n_expand, to_loop_over)
            remapped_df = pd.concat(remapped, axis=1)
            remapped_solus.loc[:, s_ids] = remapped_df
        else:
            for s in solus.drop(ref, axis=1):
                #-
                pars = (solus[s], ref_centroids, ref_kdt, xys, xy)
                remap_ids = remap_lbls_single(pars)
                #-
                remapped_solus.loc[:, s] = solus[s].map(remap_ids)
        remapped_solus.loc[:, ref] = solus.loc[:, ref]
        return remapped_solus.fillna('-1')
    else:
        print("WARNING: No clusters identified")
        return solus
    
def _remap_n_expand(pars):
    solus_s, ref_centroids, ref_kdt, xys, xy = pars
    remap_ids = remap_lbls_single(pars)
    expanded = solus_s.map(remap_ids)
    return expanded
    

def remap_lbls_single(pars):
    new_lbls, ref_centroids, ref_kdt, xys, xy = pars
    lbl_type = type(new_lbls.iloc[0])
    # Cross-walk to cluster IDs
    ref_centroids_ids = pd.Series(ref_centroids.index.values)
    # Centroids for new solution
    solu_centroids = xys.groupby(new_lbls)\
                        [xy]\
                        .apply(lambda xys: xys.mean())\
                        .drop(lbl_type(-1), errors='ignore')
    # Remapping from old to new labels
    _, nrst_ref_cl = ref_kdt.query(solu_centroids.values)
    remap_ids = pd.Series(nrst_ref_cl, 
                          index=solu_centroids.index)\
                         .map(ref_centroids_ids)
    return remap_ids

def ensemble(solus, xys, xy=['X', 'Y'], multi_cpu=False):
    '''
    Generate unique class prediction based on majority/hard voting
    ...

    Arguments
    ---------
    solus       : DataFrame
                  Table with labels for each point (row) and solution (column)

    Returns
    -------
    pred        : DataFrame
                  Table with predictions (`pred`) and proportion of votes 
                  that elected it (`pct`)
    xys         : DataFrame
                  Table including coordinates
    xy          : list
                  [Default=`['X', 'Y']`] Ordered pair of names for XY
                  coordinates in `xys`
    multi_cpu   : Boolean
                  [Default=False] If True, paralelise remapping
    '''
    f = lambda a: Counter(a).most_common(1)[0]
    remapped_solus = remap_lbls(solus, xys, xy=xy, multi_cpu=multi_cpu)
    counts = np.array(list(map(f, remapped_solus.values)))
    winner = counts[:, 0]
    votes = counts[:, 1].astype(int) / solus.shape[1]
    pred = pd.DataFrame({'lbls': winner, 'pct': votes})
    return pred

def logger(txt, f='log_revision.txt', mode='a'):
    fo = open(f, 'a')
    fo.write(txt)
    fo.close()
    return txt

