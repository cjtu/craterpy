import json
from copy import deepcopy
from typing import Union
import warnings
from pathlib import Path
import multiprocessing
import antimeridian
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import shapely
from shapely.geometry import Point
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
from rasterstats import zonal_stats, gen_zonal_stats
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import craterpy.helper as ch

# Suppress antimeridian warning
warnings.filterwarnings("ignore", category=antimeridian.FixWindingWarning)

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

    Attributes
    ----------
    data : geopandas.GeoDataFrame
        Dataframe containing the crater data.
    lat : pandas.Series
        Crater latitudes.
    lon : pandas.Series
        Crater longitudes.
    rad : pandas.Series
        Crater radii.
    center : geopandas.GeoSeries of shapely.geometry.Point objects
        Crater centers.
    """

    # Philosophy for database: Geopandas only allows 1 shape geometry per row
    # So, use the _center of crater as default geometry. Switch to other
    # geometry columns on the fly for computing stats

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
    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        body: str = "",
        units: str = "m",
    ):
        """
        Initialize a CraterDatabase.

        Parameters
        ----------
        dataset : str or pandas.DataFrame
            if str, path to the file containing crater data.
            if DataFrame, DataFrame containing crater data.
        body : str
            Planetary body, e.g. 'Moon', 'Vesta' (default: 'Moon')
        units : str
            Length units of radius/diameter, 'm' or 'km' (default: 'm')

        Raises
        ------
            ValueError
                If dataset is not a file or is not a pandas.DataFrame.
        """
        if not body:
            raise ValueError(
                f"Please specify a planetary body from, {list(CRS_DICT.keys())}"
            )
        lon_offset = 0
        if "vesta" in body.lower():
            body, lon_offset = self._vesta_check(body)
            self._vesta_coord = body
            body = "vesta"
        self.body = body
        (
            self._crs,
            self._crs180,
            self._crs360,
            self._crsnorth,
            self._crssouth,
        ) = self._load_crs(self.body)

        if isinstance(dataset, pd.DataFrame):
            self.data = dataset.copy(True)
        elif Path(dataset).is_file():
            # TODO: handle craterstats .scc files here
            self.data = gpd.read_file(dataset)
        else:
            raise ValueError("Could not read crater dataset.")
        self.orig_cols = self.data.columns

        # Store reference to lat, lon columns
        self._latcol = ch.findcol(self.data, ["latitude", "lat"])
        self._loncol = ch.findcol(self.data, ["longitude", "lon"])
        self.data[self._latcol] = pd.to_numeric(self.data[self._latcol])
        self.data[self._loncol] = pd.to_numeric(self.data[self._loncol])
        # Look for radius / diam column, store as _radius_m
        rcol = ch.find_rad_or_diam_col(self.data)
        div = 2 if rcol.startswith("d") else 1  # diam -> rad conversion
        self.data["_radius_m"] = pd.to_numeric(self.data[rcol]) / div
        self._radcol = "_radius_m"

        # Ensure lon is in -180 to 180
        self.data[self._loncol] = self.lon + lon_offset
        self.data[self._loncol] = ch.lon180(self.lon)

        # Convert to meters
        if units == "km":
            self.data[self._radcol] *= 1000

        # Generate point geometry for each row
        self.data["_center"] = self._gen_point()

        # Set geometry and covert to GeoDataFrame if not already
        if not isinstance(self.data, gpd.GeoDataFrame):
            self.data = gpd.GeoDataFrame(
                self.data, geometry="_center", crs=self._crs
            )
        elif "geometry" not in self.data.columns:
            self.data.set_geometry("_center", inplace=True)

    def __repr__(self):
        attrs = ", ".join(
            [p for p in self._get_properties() if not p.startswith("_")]
        )
        return f"CraterDatabase of length {len(self.data)} with attributes {attrs}."

    def __eq__(self, other):
        """
        Compare this CraterDatabase with another for equality.

        Two CraterDatabase objects are considered equal if all of the below are true:
        - They are instances of CraterDatabase.
        - Their underlying data GeoDataFrames are equal.
        - Their coordinate reference systems (_crs, _crs180, _crs360, _crsnorth, _crssouth)
            are equal.
        - The names of the latitude, longitude, and radius columns (_latcol, _loncol, _radcol)
            are equal.
        - The optional _vesta_coord attribute (if present) is equal.

        Parameters
        ----------
        other : object
            The other database (or object) to compare against

         Returns
        -------
        bool or NotImplemented
            - `True` if `other` is a CraterDatabase with equivalent data and metadata.
            - `False` if `other` is a CraterDatabase but differs in any of the required attributes.
            - `NotImplemented` if `other` is not a CraterDatabase instance.
        """
        if not isinstance(other, CraterDatabase):
            return NotImplemented

        return (
            self.data.equals(other.data)
            and self._crs == other._crs
            and self._crs180 == other._crs180
            and self._crs360 == other._crs360
            and self._crsnorth == other._crsnorth
            and self._crssouth == other._crssouth
            and self._latcol == other._latcol
            and self._loncol == other._loncol
            and self._radcol == other._radcol
            and getattr(self, "_vesta_coord", None)
            == getattr(other, "_vesta_coord", None)
        )

    def __len__(self):
        """Return the number of crater records in the database."""
        return len(self.data)

    def __bool__(self):
        """Return True if the database contains any crater records."""
        return not self.data.empty

    def _load_crs(self, body):
        """Return the pyproj CRSs for the body."""
        return [CRS.from_user_input(crs) for crs in CRS_DICT[body.lower()]]

    def _gen_point(self):
        """Return point geometry (lon, lat) for each row."""
        # Note: list comprehension is faster than df.apply
        return [Point(xy) for xy in zip(self.lon, self.lat)]

    def _gen_annulus(self, inner, outer, precise=False, **kwargs):
        """Return annular geometry for each row."""
        # Don't know why it prefers to edit a copy, but _precise was giving None geometries with GeoSeries([polys_out])
        out = self.center.copy()
        if precise:
            out.loc[:] = self._gen_annulus_precise(
                self.center, self.rad, inner, outer, **kwargs
            )
        else:
            # Compute simple circles for all craters (fast)
            out.loc[:] = self._gen_annulus_simple(
                inner, outer, self._crs, self._crs180, **kwargs
            )
            # Be more precise for craters that are warped or go out of bounds
            # find craters warped more than 10% in plate caree. Horizonal stretch
            # goes as cos and cos(25)=0.906, so circles within 25N-25S are <10% err
            equatorial = (-25 < out.bounds.miny) & (out.bounds.maxy < 25)
            oob = out.bounds.maxx - out.bounds.minx >= 300
            out.loc[~equatorial | oob] = self._gen_annulus_precise(
                self.center[~equatorial | oob],
                self.rad[~equatorial | oob],
                inner,
                outer,
            )
        return out

    def _get_annular_buffer(self, pt, rad, inner, outer, nvert=32):
        """Generate annulus for each row in dataframe pt must be projected to crs in rad units."""
        qs = nvert // 4  # quad_segs (n linear segments / quarter circle)
        buf = pt.buffer(rad * outer, qs)
        if inner > 0:
            buf = buf.difference(pt.buffer(rad * inner), qs)
        return buf

    def _gen_annulus_simple(self, inner, outer, src_crs, dst_crs, **kwargs):
        """Generate annulus in one dst_crs and reproject to src_crs.

        Usage
        =====
        Generate circular crater rims in simple cylindrical (PlateCaree)
            self._gen_annulus_simple(0, 1, self._crs, self._crs180)
        Generate circular crater rims in north pole stereographic
            self._gen_annulus_simple(0, 1, self._crs, self._crsnorth)
        Generate circular crater rims in south pole stereographic
            self._gen_annulus_simple(0, 1, self._crs, self._crssouth)
        """
        # Assumes radius is in same units as crs base unit (usually meters)
        annuli = self._get_annular_buffer(
            self.center.to_crs(dst_crs), self.rad, inner, outer, **kwargs
        )
        return annuli.to_crs(src_crs)

    def _gen_annulus_precise(self, centers, rads, inner, outer, **kwargs):
        """
        Generate annuli using a local azimuthal equidistant projection for each crater.

        This method is more precise than the simple method, but slower. It is
        recommended for high-precision work or when the annuli are large and
        span many degrees of latitude.
        """
        # Generate annulus centered on each crater in the dataframe
        annuli = []
        for center, rad in zip(centers, rads):
            buf = self._get_annular_buffer(
                Point(0, 0), rad, inner, outer, **kwargs
            )

            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f"AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)",
                conversion=AzimuthalEquidistantConversion(center.y, center.x),
                geodetic_crs=self._crs,
            )

            # Unproject the annular buffer from local azeq centered to geodetic crs
            to_geodetic = Transformer.from_crs(
                local_crs, self._crs, always_xy=True
            ).transform

            annulus = transform(to_geodetic, buf)

            # Fix antimeridian (if xmax-xmin is huge, it wrapped around the globe the wrong way)
            if annulus.bounds[2] - annulus.bounds[0] >= 300:
                # Check if north or south crossing
                # Need to check in projected coords because geodetic coords are broken
                # Also makes circlar buf with no hole in center
                to_projected = Transformer.from_crs(
                    self._crs, local_crs, always_xy=True
                ).transform
                sbuf = self._get_annular_buffer(
                    Point(0, 0), rad, 0, outer, **kwargs
                )
                isn = sbuf.contains(transform(to_projected, Point(20, 90)))
                iss = sbuf.contains(transform(to_projected, Point(20, -90)))

                annulus = antimeridian.fix_polygon(
                    annulus, force_north_pole=isn, force_south_pole=iss
                )
            annuli.append(annulus)
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

    def _make_data_property(self, col):
        """Make a column of self.data accessible directly as an attribute."""
        c = col.replace(" ", "_")  # attr can't have spaces
        setattr(self.__class__, c, property(fget=lambda self: self.data[col]))

    def _get_properties(self):
        """Return list of property names."""
        class_items = self.__class__.__dict__.items()
        return list(k for k, v in class_items if isinstance(v, property))

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
        """Crater radii in meters."""
        return self.data[self._radcol]

    @property
    def center(self):
        """Crater center point geometry."""
        return self.data["_center"]

    @property
    def _rbody(self):
        """Return average planetary radius from crs in meters."""
        ellipse = self._crs.ellipsoid
        return (ellipse.semi_major_metre + ellipse.semi_minor_metre) / 2

    def copy(self):
        """Return a deepcopy of a CraterDatabase."""
        return deepcopy(self)

    def add_annuli(self, name, inner, outer, precise=True):
        """Generate annular geometries for each crater in database.

        (slower but more precise than using global, Npolar and Spolar stereo)

        Parameters
        ----------
        name: str
            Name of geometry column.
        inner: int or float
            Distance from center to inner edge of annulus in crater radii.
        outer: int or float
            Distance from center to outer edge of annulus in crater radii.
        precise: bool
            Precisely calculate each geometry in a local projection (default: True).

        Examples
        --------
        .. code-block:: python

            >>> cdb.add_annuli("name", 1, 2)  # generates annuli from each crater rim to 1 crater radius beyond the rim.
            >>> cdb.add_annuli("name", 1, 3)  # generates annuli from each crater rim to 1 crater diameter beyond the rim.
            >>> cdb.add_annuli("name", 0, 1)  # generates a cicle capturing the interior of the crater rim.

        """
        name = name or f"annulus_{inner}_{outer}"
        self.data[name] = self._gen_annulus(inner, outer, precise)
        self._make_data_property(name)

    def add_circles(self, name="", size=1, precise=True):
        """Generate circluar geometries for each crater in database.

        Parameters
        ----------
        name : str
            Name of geometry column (default: circle_{size}).
        size : int or float
            Radius of circle around each crater in crater radii (default: 1).
        precise : bool
            Precisely calculate each geometry in a local projection (default: True).
        """
        name = name or f"circle_{size}"
        self.add_annuli(inner=0, outer=size, name=name, precise=precise)

    def _get_stats(
        self, fraster, region, stats=STATS, nodata=None, suffix=None
    ):
        """Return DataFrame of zonal stats on region from fraster."""
        zstats = zonal_stats(
            self.data.set_geometry(region),
            fraster,
            stats=stats,
            nodata=nodata,
            all_touched=True,
        )
        out = pd.DataFrame(
            {stat: [z[stat] for z in zstats] for stat in stats},
            index=self.data.index,
        )
        if suffix:
            out = out.add_suffix(f"_{suffix}")
        return out

    def get_stats(self, rasters, regions, stats=STATS, nodata=None, n_jobs=1):
        """
        Compute stats on polygons in a GeoDataFrame in parallel.

        Parameters
        ----------
        rasters : str, rasterio.DatasetReader, or dict of {str: str or rasterio.DatasetReader}
            Single raster path or open raster dataset, or a mapping of names to rasters.
            If a single raster is provided, it will be wrapped in a dict with key '_'.
        regions : str or list of str
            Name or list of names of geometry columns in the crater database to use as regions.
        stats : tuple of str, optional
            Statistics to compute for each raster-region combination.
            Default is ("mean", "std", "count").
        nodata : number, None, or dict of {str: number}, optional
            Nodata value to use for masking, or mapping of raster names to their nodata values.
            If None, no nodata masking is applied.
        n_jobs : int, optional
            Number of parallel worker processes to use. Default is 1.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the original crater columns along with appended
            statistics columns for each raster and region combination.
        """

        if not isinstance(rasters, dict):
            rasters = {"_": rasters}  # Will be stripped from stats colname
        if isinstance(regions, str):
            regions = [regions]
        # Convert numeric nodata value to dict
        if not isinstance(nodata, dict):
            nodata = {k: nodata for k in rasters.keys()}
        args = [
            [
                f,
                region,
                stats,
                nodata.get(name, None),
                f"{name}_{region}".strip("_"),
            ]
            for region in regions
            for name, f in rasters.items()
        ]
        with multiprocessing.Pool(n_jobs) as pool:
            result = pool.starmap(self._get_stats, args)
        return pd.concat([self.data[self.orig_cols], *result], axis=1)

    def plot(
        self,
        fraster=None,
        region="",
        ax=None,
        size=6,
        dpi=100,
        band=1,
        alpha=0.5,
        color="tab:blue",
        **kwargs,
    ):
        """Plot crater geometries.

        Parameters
        ----------
        fraster : str
            A path to raster image data to overlay craters on.
        region : str
            The CraterDatabase region geometries to plot (default: crater rims).
        ax : matplotlib.Axes
            Axes on which to plot data.
        size : int, float, or 2-tuple
            Width of plot in inches or figsize tuple (width, height).
        dpi : int
            Resolution of plot (dots/inch). Lower resolutions load faster.
        band : int
            Band to use from fraster (default: first).
        alpha: float
            Transparancy of the ROI geometries (0-1, default: 0.2).
        color : str
            Color of the ROI geometries.
        **kwargs
            Keyword arguments supplied to GeoSeries.plot().

        Returns
        -------
        ax : matplotlib.Axes
            Original axes, now with data plotted.
        """
        if fraster:
            if ax is not None:
                size = ax.figure.get_size_inches()
                dpi = ax.figure.get_dpi()
            with rio.open(fraster) as src:
                if isinstance(size, (int, float)):
                    aspect = src.width / src.height
                    size = (size, size / aspect)
                height_npix = int(size[1] * dpi)
                width_npix = int(size[0] * dpi)
                # By default does nearest-neighbor interp to out_shape (super fast reads for low dpi)
                data = src.read(
                    indexes=band, out_shape=(height_npix, width_npix)
                )
                extent = ch.bbox2extent(src.bounds)
            if ax is None:
                fig, ax = plt.subplots(figsize=size, dpi=dpi)
            ax.imshow(data, cmap="gray", extent=extent)
        if not region:
            # Store crater circles in .data with leading "_" to not be mistaken
            #  for user-defined ROIs
            region = "_plot_circles"
            if region not in self.data.columns:
                self.data[region] = self._gen_annulus(0, 1, precise=False)
        rois = self.data.loc[:, region].boundary  # plot outline of ROI
        ax = rois.plot(ax=ax, alpha=alpha, color=color, **kwargs)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        label = " " + region if not region.startswith("_") else ""
        ax.set_title(f"CraterDatabase{label} (N={len(self.data)})")
        return ax

    def to_geojson(
        self,
        filename,
        region=None,
        crs=None,
        keep_cols=None,
        keep_all=False,
    ):
        """
        Export the crater database to GeoJSON file.

        Parameters
        ----------
        filename : str
            Path to output GeoJSON file.
        region : str, optional
            Name of the region geometries to export. Default: crater centers point geometry.
        crs : str or pyproj.CRS, optional
            Target coordinate reference system. If None, uses the current CRS.
        keep_cols : list, optional
            List of column names to keep. If None, includes all original columns
            and excludes region geometries.
        keep_all : bool, optional
            If True, keep hidden columns (e.g. all region geometries). Default is False.
            Only if keep_cols is None (specific columns requested take precedence).
        """
        if region is None:
            geom_col = "_center"
        elif region in self.data.columns:
            geom_col = region
        else:
            raise ValueError(f"Geometry column '{region}' not found.")

        # Determine which columns to keep
        must_keep = {self._latcol, self._loncol, self._radcol}
        if keep_cols is None:
            # Take all columns if keep cols otherwise only take cols without leading "_"
            keep_cols = [
                col
                for col in self.data.columns
                if keep_all or col in must_keep or not col.startswith("_")
            ]
        else:
            # Check if any specified cols don't exist
            missing_cols = [
                col for col in keep_cols if col not in self.data.columns
            ]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")

        if geom_col not in keep_cols:
            keep_cols = list(set(keep_cols).union({geom_col}))
        keep_cols = list(must_keep.union(set(keep_cols)))
        # Prep for export - always writes active geometry with the name 'geometry'
        # So copy the active col to 'geometry' and keep the original with its name
        # All other geometry columns need to be written as string (WKT format)
        data = self.data.copy()
        data = data[keep_cols]
        data["geometry"] = data[geom_col]
        data = data.set_geometry("geometry")

        # Get the rest of the geometry type colums (minus the one called geometry)
        other_geom_cols = set(data.select_dtypes("geometry")) - {"geometry"}
        for col in other_geom_cols:
            data[col] = data.apply(lambda x: x[col].wkt, axis=1)
            # TODO: there should be a better way to store these without all the naming hacks
            data.rename(columns={col: col + "_wkt"}, inplace=True)
        data.rename(
            columns={geom_col + "_wkt": geom_col + "_active_wkt"}, inplace=True
        )

        if crs is not None:
            data = data.to_crs(crs)
        data.to_file(filename, driver="GeoJSON")

    def to_crs(self, crs, inplace: bool = False):
        """
        Convert the crater database to a different coordinate reference system.

        Parameters
        ----------
        crs : str or pyproj.CRS
            Target coordinate reference system.
        inplace : bool, optional
            If True, modifies the current instance. If False, returns a new instance.
            Default is False.

        Returns
        -------
        CraterDatabase
            A new CraterDatabase instance with the converted data if inplace is False.
            Otherwise, modifies the current instance.
        """

        if inplace:
            self.data.to_crs(crs, inplace=True)
            return
        else:
            # Make a copy of the whole instance and do to_crs inplace on the copy
            new_crater_db = self.copy()
            new_crater_db.to_crs(crs, inplace=True)
            return new_crater_db

    def plot_rois(self, fraster, region, index=9, grid_kw=None, **kwargs):
        """
        Plot CraterDatabase regions of interest (ROIs) clipped from raster.

        Plots the first 9 craters supplied in index.

        Parameters
        ----------
        cdb : object
            A CraterDatabase with region defined, e.g., with add_annuli().
        fraster : str
            A path to raster to clip rois from.
        region : str
            The name of the CraterDatabase region geometry to plot.
        index : int, pd.Index, or iterable, optional
            Specifies which ROIs to plot. If an integer, take first n,
            otherwise plot all given indices. Default is 9.
        grid_kw : dict, optional
            Keyword args for gridlines (see matplotlib.axes.gridlines()).
        **kwargs : dict, optional
            Additional keyword arguments for customizing the plot. These can overwrite
            default settings for the raster image (`cmap`, `vmin`, `vmax`) or the ROI
            geometries (`color`, `facecolor`, `lw`, `alpha`, etc.).

        Returns
        -------
        axes : array of matplotlib.axes._subplots.AxesSubplot

        Examples
        --------
        .. code-block:: python

            >>> cdb = CraterDatabase(craters.csv, body="Moon")
            >>> cdb.add_circles("crater", 1.5)
            >>> cdb.plot_rois("moon.tif", region='crater', index=(1, 6, 9), alpha=0.2)

        """
        # Parse index an int, or pd Index otherwise pass to .iloc (e.g., range, tuple, list, array of indices should all work)
        gdf = self.data[region]
        if isinstance(index, pd.Index):
            gdf = gdf.loc[index]

        # TODO: find a way to display these
        # Filter out polar or antimeridian crossing rois
        polar = (gdf.bounds.miny < -89) | (89 < gdf.bounds.maxy)
        oob = gdf.bounds.maxx - gdf.bounds.minx >= 300
        if any(polar | oob):
            warnings.warn("Skipping ROIs that cross pole or antimeridian...")
        gdf = gdf.loc[~polar & ~oob]
        if len(gdf) == 0:
            raise ValueError(
                "No geometries are in map bounds. Perhaps check radius units or raster extent?"
            )

        if isinstance(index, int):
            index = gdf.head(index).index
        gdf = gdf.iloc[index]

        # Plot default kwargs (user kwargs will overwrite these if present)
        im_kw = {
            "cmap": kwargs.pop("cmap", "gray"),
            "vmin": kwargs.pop("vmin", None),
            "vmax": kwargs.pop("vmax", None),
        }
        shp_kw = dict(color="tab:green", facecolor="none", lw=2, alpha=0.5)
        shp_kw = {**shp_kw, **kwargs}

        # Make projection
        ellipsoid = ccrs.CRS(self._crs).ellipsoid
        globe = ccrs.Globe(
            semimajor_axis=ellipsoid.semi_major_metre,
            semiminor_axis=ellipsoid.semi_minor_metre,
            ellipse=None,
        )
        pc = ccrs.PlateCarree(globe=globe)

        # Make roi array generator
        rois = gen_zonal_stats(
            gdf.geometry,
            fraster,
            stats="count",
            raster_out=True,
            all_touched=True,
        )
        # Make subplots n x 3 grid, is hardcoded b/c labels/fontsize affect aspect ratio
        n = len(gdf)
        rows = 1 + (n - 1) // 3
        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(9 + 2, -0.5 + rows * 3.4),
            subplot_kw={"projection": pc},
            gridspec_kw={"wspace": 0.4, "hspace": 0.2},
        )
        for geom, roi, ax in zip(gdf.geometry, rois, axes.flatten()):
            img = roi["mini_raster_array"]
            extent = ch.bbox2extent(geom.bounds)
            ax.imshow(img, extent=extent, aspect="auto", **im_kw)
            ax.add_feature(ShapelyFeature(geom, crs=pc, **shp_kw))

        # Delete the unused axes
        n = len(index)
        for i in range(n, 3 * rows):
            fig.delaxes(axes.flatten()[i])

        for ax in np.atleast_1d(axes).flatten():
            # Parse gridline kws, overwrite defaults if given
            grid_kw = {} if grid_kw is None else grid_kw
            gl_kw = dict(draw_labels=True, dms=False, ls="--", alpha=0.5)
            gl_kw = {**gl_kw, **grid_kw}
            gl = ax.gridlines(**gl_kw)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        return axes

    @classmethod
    def merge(cls, cdb1, cdb2):
        """
        Return a new CraterDatabase with duplicate crater rows and no ROIs.

        Parameters
        ----------
        cdb1 : CraterDatabase
            The first CraterDatabase object to merge.
        cdb2 : CraterDatabase
            The second CraterDatabase object to merge.

        Returns
        -------
        CraterDatabase
            A new CraterDatabase instance with craters.

        Raises
        ------
        ValueError
            If CraterDatabase objects are from different bodies.
        """
        if cdb1.body != cdb2.body:
            raise ValueError(
                "Cannot merge CraterDatabases from different bodies!"
            )
        merged = ch.merge(cdb1.data, cdb2.data, rbody=cdb1._rbody)
        return cls(merged, body=cdb1.body, units=cdb1.units)

    @classmethod
    def read_shapefile(
        cls,
        filename,
        body: str = "Moon",
        units: str = "m",
    ):
        """
        Read crater data from a shapefile or GeoJSON file.

        Parameters
        ----------
        filename : str
            Path to the shapefile or GeoJSON file.
        body : str, optional
            Planetary body, e.g. Moon, Vesta (default: None).
            If None, will attempt to determine from the file's CRS.
        units : str, optional
            Length units of radius/diameter, m or km (default: m).

        Returns
        -------
        CraterDatabase

        Notes
        -----
        This method assumes the file was previously created by `CraterDatabase.to_geojson()`
        or has a compatible format with lat/lon coordinates and radius or diameter information.
        """
        # Try to read metadata from GeoJSON first (if it's a GeoJSON file)
        file_metadata = {}
        if filename.lower().endswith(".geojson"):
            try:
                with open(filename, "r") as f:
                    geojson_data = json.load(f)
                    if "metadata" in geojson_data:
                        file_metadata = geojson_data["metadata"]
            except (json.JSONDecodeError, IOError):
                # If can't read it as JSON, try geopandas anyway
                pass

        # Read the file with geopandas
        data = gpd.read_file(filename)

        # Determine body (priority: user-specified > file metadata > CRS detection)
        if body is None:
            # Check if we have body info in metadata
            if "body" in file_metadata:
                body = file_metadata["body"]
            else:
                # Try to determine from CRS
                try:
                    body = data.crs.ellipsoid.name.split()[0].lower()
                except (AttributeError, IndexError) as err:
                    raise ValueError(
                        "Could not determine planetary body from file CRS. "
                        "Please specify the body parameter explicitly."
                    ) from err

        # Check if we have units info in metadata
        file_units = file_metadata.get("units", units)

        # Restore any columns exported as "_wkt" to geometry columns
        for col in data.columns:
            if col.endswith("_wkt"):
                data[col] = data.apply(
                    lambda x: shapely.wkt.loads(x[col]), axis=1
                )
            if col.endswith("_active_wkt"):
                data.drop(columns="geometry", inplace=True)
        data.rename(
            columns={
                c: c.removesuffix("_wkt").removesuffix("_active")
                for c in data.columns
            },
            inplace=True,
        )

        # Create and return a new CraterDatabase instance
        return cls(data, body=body, units=file_units)
