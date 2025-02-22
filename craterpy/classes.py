import warnings
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
import craterpy.helper as ch
from rasterstats import zonal_stats


import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs


# Default stats for rasterstats
STATS = ("mean", "std", "count")

# CRS for units / coord transformations (convention is only Ocentric )
CRS_DICT = {
    # body: (Geographic2D [deg], Equirect [m] (clon=0), Equirect [m] (clon=180),  Npole Stereo [m], Spole Stereo [m])
    "moon": (
        "IAU_2015:30100",
        "IAU_2015:30110",
        "IAU_2015:30115",
        "IAU_2015:30130",
        "IAU_2015:30135",
    ),
    "mars": (
        "IAU_2015:49900",
        "IAU_2015:49910",
        "IAU_2015:49915",
        "IAU_2015:49930",
        "IAU_2015:49935",
    ),
    "mercury": (
        "IAU_2015:19900",
        "IAU_2015:19910",
        "IAU_2015:19915",
        "IAU_2015:19930",
        "IAU_2015:19935",
    ),
    "venus": (
        "IAU_2015:29900",
        "IAU_2015:29910",
        "IAU_2015:29915",
        "IAU_2015:29930",
        "IAU_2015:29935",
    ),
    "europa": (
        "IAU_2015:50200",
        "IAU_2015:50210",
        "IAU_2015:50215",
        "IAU_2015:50230",
        "IAU_2015:50235",
    ),
    "ceres": (
        "IAU_2015:200000100",
        "IAU_2015:200000110",
        "IAU_2015:200000115",
        "IAU_2015:200000130",
        "IAU_2015:200000135",
    ),
    "vesta": (
        "IAU_2015:200000400",
        "IAU_2015:200000410",
        "IAU_2015:200000415",
        "IAU_2015:200000430",
        "IAU_2015:200000435",
    ),
}


class CraterDatabase:
    """Database of crater locations and shapefiles.

    Attributes:
        data (GeoDataFrame): GeoDataFrame containing the crater data.
        lat (Series): Crater latitudes.
        lon (Series): Crater longitudes.
        rad (Series): Crater radii.
        center (GeoSeries): Crater center (shapely.geometry Point).
    """

    # Philosophy for database: Geopandas only allows 1 shape geometry per row
    # So, use the _point geometry for center of each crater by default
    # Then switch to _rim / _annulus geometry on the fly for computing stats

    # Private Attrs:
    # _crs (str): Coordinate reference system for the body.
    # _crs180 (str): CRS with longitude in -180 to 180 degrees.
    # _crs360 (str): CRS with longitude in 0 to 360 degrees.
    # _crsnorth (str): CRS for the northern hemisphere.
    # _crssouth (str): CRS for the southern hemisphere.
    # _latcol (str): Column name for latitude.
    # _loncol (str): Column name for longitude.
    # _radcol (str): Column name for radius.
    # _vesta_coord (str, optional): Coordinate system for Vesta, if applicable.
    def __init__(self, dataset, body="Moon", units="m"):
        """
        Initialize a CraterDatabase.

        Parameters:
            filepath (str): Path to the file containing crater data.
            body (str): Planetary body, e.g. Moon, Vesta (default: Moon)
            units (str): Length units of radius/diameter, m or km (default: m)
        """
        lon_offset = 0
        if "vesta" in body.lower():
            body, lon_offset = self._vesta_check(body)
            self._vesta_coord = body
            body = "vesta"
        (
            self._crs,
            self._crs180,
            self._crs360,
            self._crsnorth,
            self._crssouth,
        ) = self._load_crs(body)
        
        if isinstance(dataset, pd.DataFrame):
            self.data = dataset
        elif Path(dataset).is_file():
            # TODO: handle craterstats .scc files here
            self.data = gpd.read_file(dataset)
        else:
            raise ValueError("Could not read crater dataset.")

        # Store reference to lat, lon, rad columns
        cols = ch.get_crater_cols(self.data)
        for col in cols:
            # Fixes lat, lon, rad columns if loaded as a string or object
            self.data[col] = pd.to_numeric(self.data[col])
        self._latcol, self._loncol, self.radcol = cols

        # Ensure lon is in -180 to 180
        self.data[self._loncol] = self.lon - lon_offset
        self.data[self._loncol] = ch.lon180(self.lon)

        # Convert to meters
        if units == "km":
            self.data[self.radcol] *= 1000

        # Generate point geometry for each row
        self.data["_point"] = self._generate_point()

        # Set geometry and covert to GeoDataFrame if not already
        if not isinstance(self.data, gpd.GeoDataFrame):
            self.data = gpd.GeoDataFrame(
                self.data, geometry="_point", crs=self._crs
            )
        elif "geometry" not in self.data.columns:
            self.data.set_geometry("_point", inplace=True)


    def __repr__(self):
        return self.data.__repr__()

    def _load_crs(self, body):
        """Return the pyproj CRSs for the body."""
        return [CRS.from_user_input(crs) for crs in CRS_DICT[body.lower()]]

    def _generate_point(self):
        """Return point geometry (lon, lat) for each row."""
        # Note: list comprehension is faster than df.apply
        return [Point(xy) for xy in zip(self.lon, self.lat)]

    def _generate_annulus(self, inner, outer, precise=False):
        """Return annular geometry for each row."""
        # Assumes radius is in same units as projection base unit (e.g., meters)
        if precise:
            out = self._generate_annulus_precise(self.center, self.rad, 0, 1)
        else:
            out = self.data._point.copy()
            # Draws circles in simple cylindrical for all craters
            lat_eq = 50  # +/-lat where annuli in simple cylindrical are ok
            is_eq = (-lat_eq <= self.lat) & (self.lat <= lat_eq)
            out.loc[is_eq] = self._generate_annulus_simple(
                self.center[is_eq], self.rad[is_eq], inner, outer, self._crs, self._crs180
            )

            # Re-process annuli in the north and south using N/S stereographic
            is_n = self.lat > lat_eq
            out.loc[is_n] = self._generate_annulus_simple(
                self.center[is_n],
                self.rad[is_n],
                inner,
                outer,
                self._crs,
                self._crsnorth,
            )
            is_s = self.lat < -lat_eq
            out.loc[is_s] = self._generate_annulus_simple(
                self.center[is_s],
                self.rad[is_s],
                inner,
                outer,
                self._crs,
                self._crssouth,
            )

            # Finally, correct very large annuli
            # Suppress warn: Usually shouldn't compute area in unprojected coords 
            #   because of stretching but that's what we want here - how messed 
            #   up does an annulus get when stretched near poles
            warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS*')
            is_large = out.area > 25  # [deg^2] TODO: Is this a good threshold?
            out.loc[is_large] = self._generate_annulus_precise(
                self.center[is_large], self.rad[is_large], inner, outer
            )
        return out

    def _get_annular_buffer(self, pt, rad, inner, outer):
        """Generate annulus for each row in dataframe pt must be projected to crs in rad units."""
        # TODO: option to make circle more precise with more vertices at cost of memory? e.g. pt.buffer(rad, 128)
        if inner == 0:
            return pt.buffer(rad * outer)
        else:
            return pt.buffer(rad * outer).difference(pt.buffer(rad * inner))

    def _generate_annulus_simple(
        self, pt, rad, inner, outer, src_crs, dst_crs
    ):
        """Generate annulus naievely in one projection"""
        annuli = self._get_annular_buffer(
            pt.to_crs(dst_crs), rad, inner, outer
        )
        out = annuli.to_crs(src_crs)

        # Correct annuli that cross the antimeridian (make a huge poly stretching across the globe, i.e. lon>180)
        oob = out.bounds["maxx"] - out.bounds["minx"] >= 179
        out.loc[oob] = ch.unproject_split_meridian(
            annuli.loc[oob], src_crs, dst_crs
        )
        return out

    def _generate_annulus_precise(self, centers, rads, inner, outer):
        """
        Generate annuli using a local azimuthal equidistant projection for each crater.

        This method is more precise than the simple method, but slower. It is
        recommended for high-precision work or when the annuli are large and
        span many degrees of latitude.
        """
        # Generate annulus centered on each crater in the dataframe
        annuli = []
        for (center, rad) in zip(centers, rads):
            annulus = self._get_annular_buffer(Point(0, 0), rad, inner, outer)

            # TODO apply antimeridian cutting / wrap around 180 correction
            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f"AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)",
                conversion=AzimuthalEquidistantConversion(center.y, center.x),
                geodetic_crs=self._crs,
            )

            # Project the annulus from local azeq centered to main CRS
            transformer = Transformer.from_crs(
                local_crs, self._crs, always_xy=True
            )
            annuli.append(transform(transformer.transform, annulus))
        return annuli

    def _vesta_check(self, body):
        """
        Return the lon offset for craters from claudia_dp coord system.

        Default: claudia_dp (Claudia Double Prime / PDS-Vesta-2012), in use
        by the Dawn mission and accepted by the IAU. Each coordinate system
        requires an offset, seen by the shift in the reference crater Claudia:
        - vesta_claudia_dp (Claudia Double Prime): Claudia at (-1.6N, 146E)
        - vesta_claudia_p (Claudia Prime): Claudia at (-1.6N, 136E)
        - vesta_claudia (Dawn-Claudia): Claudia at (-1.6N, 356E)
        - vesta_iau_2000 (IAU-2000): Claudia at (4.3N, 145E) - not supported

        See the NASA PDS small bodies node notes on Vesta coordinate systems:
        https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1A/DOCUMENT/VESTA_COORDINATES/VESTA_COORDINATES_131018.PDF
        """
        body = body.lower()
        if body in ["vesta_claudia_dp", "vesta_claudia_double_prime"]:
            return "vesta_claudia_dp", 0
        elif body in ["vesta_claudia_p", "vesta_claudia_prime"]:
            return "vesta_claudia_p", 190
        elif body in ["vesta_claudia", "vesta_dawn_claudia"]:
            return "vesta_claudia", 150
        elif body in ["vesta_iau_2000", "vesta_iau2000"]:
            raise NotImplementedError(
                "Vesta IAU 2000 coordinate system is not supported."
            )

        # Default to claudia_dp if no match
        warnings.warn(
            "Vesta has multiple coordinate systems. Defaulting to vesta_claudia_dp... "
            "Specify one of (vesta_claudia_dp, vesta_claudia_p, vesta_claudia). to "
            "avoid this warning. Type help(CraterDatabase()._vesta_check)"
            "for more info."
        )
        return "vesta_claudia_dp", 0

    @property
    def lat(self):
        """Crater latitudes."""
        return self.data[self._latcol]

    @property
    def lon(self):
        """Crater longitudes."""
        return self.data[self._loncol]

    @property
    def rad(self):
        """Crater radii."""
        return self.data[self.radcol]

    @property
    def center(self):
        """Crater center point geometry."""
        return self.data["_point"]

    # def zonal_stats(self, rasters, roi='ejecta', **kwargs):
    #     """Compute zonal statistics on all craters."""
    #     # TODO: finish writing this and test
    #     tmpdf = self.data
    #     if roi == 'ejecta':
    #         tmpdf = tmpdf['annulus'].rename({'annulus': 'geometry'}).set_geometry('geometry')
    #     elif roi == 'crater':
    #         tmpdf = tmpdf['_rim'].rename({'_rim': 'geometry'}).set_geometry('geometry')

    #     # Yields generator - figure out where we want to store all the data
    #     for raster in rasters:
    #         out = gen_zonal_stats(tmpdf, raster, **kwargs)
    #         break
    #     return list(out)
    def add_annuli(self, inner, outer, label=None, precise=False):
        """Generate annular shapefiles for each crater in database.

        inner: Num crater radii to inner edge (from center).
        outer: Num crater radii to outer edge (from center).
        precise: Use local projection for all craters at cost of speed. 

        Examples: 
        - cdb.add_annuli(0, 1) is a circle capturing the interior of the crater.
        - cdb.add_annuli(1, 3) is the annulus from the rim to 1 crater diameter beyond the rim

        """
        if label is None:
            label = f'Annulus({inner}, {outer})'
        self.data[label] = self._generate_annulus(inner, outer, precise)

    def get_stats(self, fraster, region, stats=STATS, nodata=None, suffix=None):
        """Compute stats on polygons in a GeoDataFrame."""
        zstats = zonal_stats(self.data.set_geometry(region), fraster, stats=stats, nodata=nodata)
        out = pd.DataFrame({stat: [z[stat] for z in zstats] for stat in stats}, index=self.data.index)
        if suffix:
            out = out.add_suffix(f"_{suffix}")
        return out

    def get_stats_parallel(self, raster_dict, regions, stats=STATS, nodata=None, n_jobs=None):
        """Compute stats on polygons in a GeoDataFrame in parallel."""
        # Convert numeric nodata value to dict
        if not isinstance(nodata, dict):
            nodata = {k: nodata for k in raster_dict.keys()}
        with mp.Pool(n_jobs) as pool:
            results = []
            for name, f in raster_dict.items():
                for region in regions:
                    results.append(
                        pool.apply_async(self.get_stats, 
                                 args=(f, region, stats, nodata.get(name, None), f'{name}_{region}'))
                    )
            results = [r.get() for r in results]
        return pd.concat(results, axis=1)
    
    # def plot_region(self, fraster, region, subset=range(5)):
    #     """Display the regions on the raster given for rows in subset."""
        
    #     def b2e(bounds):
    #         """Reorder bounds from [w, s, e, n] to extent [x0, x1, y0, y1]."""
    #         return bounds[0], bounds[2], bounds[1], bounds[3]

    #     # TODO: Switch to only plotting what is clipped by zonal stats
    #     img = plt.imread(fraster)  # Will crash for very large rasters
    #     # rois = zonal_stats(data.set_geometry(region), fraster, stats='count',
    #     #                    raster_out=True)
        
    #     for i, crater in self.data.loc[subset].iterrows():
    #         lat = self.lat.loc[i]
    #         lon = self.lon.loc[i]
    #         crs = ProjectedCRS(name=f'AzimuthalEquidistant({lat:.2f}N, {lon:.2f}E)',     
    #                                   conversion=AzimuthalEquidistantConversion(lat, lon), 
    #                                   geodetic_crs=self._crs)
    #         globe = ccrs.Globe(semimajor_axis=crs.ellipsoid.semi_major_metre, 
    #                            semiminor_axis=crs.ellipsoid.semi_minor_metre, 
    #                            ellipse=None)
    #         ae = ccrs.AzimuthalEquidistant(lon, lat, globe=globe)
    #         pc = ccrs.PlateCarree(globe=globe)

    #         # Plot
    #         fig = plt.figure(figsize=(4, 4))
    #         ax = plt.subplot(111, projection=ae)
    #         ax.imshow(img, extent=(-180,180,-90,90), transform=pc, cmap='gray')
            
    #         # Crop to region
    #         ax.set_extent(b2e(crater[region].bounds), crs=pc)
    #         ax.set_boundary(mpath.Path(crater[region].exterior.coords), transform=pc)
            
    #         # Draw region
    #         ax.add_geometries(crater[region], crs=pc, facecolor='none', edgecolor='r', linewidth=2, alpha=0.8)
    #         ax.gridlines()