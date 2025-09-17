import json
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.image as mpli
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from pyproj import CRS, Transformer
from rasterstats import gen_zonal_stats
from shapely.geometry import Point

import craterpy.helper as ch
from craterpy.crs import get_crs

# Default stats for rasterstats
STATS = ("mean", "std", "count")


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

    # CraterDatabases are stored as geopandas.GeoDataFrame with Point(lon,lat) crater center as geometry
    # The default _crs for each planetary body is the planetocentric geodetic IAU_2015 (-180,180)
    # Crater shape geometries are stored as GeoSeries and reprojected only when needed via .to_crs()

    # Private Attrs:
    # _crs (str): Internal coordinate reference system for the body, defaults to planetocentric.
    def __init__(
        self,
        dataset: str | pd.DataFrame,
        body: str,
        input_crs: str | CRS = "default",
        units: str = "m",
    ):
        """
        Initialize a CraterDatabase.

        Parameters
        ----------
        dataset : str or pandas.DataFrame
            Path to the crater file or DataFrame.
        body : str
            Planetary body (case-insensitive), e.g. 'Moon', 'mars', 'Vesta'
        input_crs : str or pyproj.CRS
            The Coordinate Reference System of the input latitude/longitude data.
            Can be a user-friendly string alias (e.g., 'planetographic', 'claudia_dp')
            or a valid pyproj.CRS object.
        units : str
            Length units of radius/diameter column, 'm' or 'km' (default: 'm')

        Raises
        ------
            ValueError
                If dataset is not a file or is not a pandas.DataFrame.
        """
        self.body = body.lower()
        self._crs = get_crs(self.body, "planetocentric")
        self._input_crs = get_crs(body, input_crs)

        if isinstance(dataset, pd.DataFrame):
            in_data = dataset
        elif Path(dataset).is_file():
            # TODO: handle craterstats .scc files here
            in_data = gpd.read_file(dataset)
        else:
            raise ValueError("Could not read crater dataset.")
        self.orig_cols = in_data.columns

        # Find lat, lon coord columns and create the point geometry
        # TODO: get centroid from geometry instead, if exists (need to handle antimeridian wrap)
        in_data = in_data.drop(columns="geometry", errors="ignore")
        lats = pd.to_numeric(in_data[ch.findcol(in_data, ["latitude", "lat"])])
        lons = pd.to_numeric(in_data[ch.findcol(in_data, ["longitude", "lon"])])

        # Create main GeoDataFrame with geometry coords in input_crs then standardize to self._crs
        transformer = Transformer.from_crs(self._input_crs, self._crs, always_xy=True)
        self.input_to_geodetic = transformer.transform
        # Cludge: Bug where transformation swaps lon, lat because of crs axis order. Needs better crs handling
        #  transformation(lon,lat,always_xy=True) only forces output to lon,lat, but dep on crs, input can
        #  be interpreted as lat, lon causing coord swap. Passing input order based on axis order didn't always work.
        #  Cludge is simple check for if in_lon 359 -> out_lat 359 (invalid) and if so, swap input order.
        if abs(self.input_to_geodetic(359, 0)[1]) > 90:
            self.input_to_geodetic = lambda x, y: transformer.transform(y, x)
        self.geodetic_to_input = partial(transformer.transform, direction="INVERSE")
        lons, lats = self.input_to_geodetic(lons, lats)
        lons = ch.lon180(lons)
        geom = gpd.points_from_xy(lons, lats)

        self.data = gpd.GeoDataFrame(in_data, geometry=geom, crs=self._crs)
        self.data["_lat"] = self.data.geometry.y
        self.data["_lon"] = self.data.geometry.x

        # Look for radius / diam column, store as _radius_m
        rcol = ch.find_rad_or_diam_col(self.data)
        div = 2 if rcol.lower().startswith("d") else 1  # diam -> rad conversion
        mul = 1000 if units == "km" else 1  # Convert km -> m
        self.data["_radius_m"] = pd.to_numeric(self.data[rcol]) * mul / div

    def __repr__(self):
        attrs = ", ".join([p for p in self._get_properties() if not p.startswith("_")])
        return f"CraterDatabase of length {len(self.data)} with attributes {attrs}."

    def __str__(self):
        body = self.body.capitalize()
        return f"{body} CraterDatabase (N={len(self.data)})"

    def __eq__(self, other):
        """
        Compare this CraterDatabase with another for equality.

        Two CraterDatabase objects are considered equal if all of the below are true:
        - They are instances of CraterDatabase.
        - Their underlying data GeoDataFrames are equal.
        - They have the same _crs

        Parameters
        ----------
        other : object
            The other database (or object) to compare against
        """
        if not isinstance(other, CraterDatabase):
            return NotImplemented

        return self.data.equals(other.data) and self._crs == other._crs

    def __len__(self):
        """Return the number of crater records in the database."""
        return len(self.data)

    def __bool__(self):
        """Return True if the database contains any crater records."""
        return not self.data.empty

    def _convert_to_geodetic(self, lons, lats):
        """Return"""

    def _gen_point(self):
        """Return point geometry (lon, lat) for each row."""
        # Note: list comprehension is faster than df.apply
        return [Point(xy) for xy in zip(self.lon, self.lat, strict=True)]

    def _gen_annulus(self, inner, outer, **kwargs):
        """Return annular geometry for each row."""
        # Don't know why it prefers to edit a copy, but _precise was giving None geometries with GeoSeries([polys_out])
        out = self.center.copy()
        out.loc[:] = ch.gen_annuli(
            self.center, self.rad, inner, outer, self._crs, **kwargs
        )
        return out

    def _make_data_property(self, col):
        """Make a column of self.data accessible directly as an attribute."""
        c = col.replace(" ", "_")  # attr can't have spaces
        setattr(self.__class__, c, property(fget=lambda self: self.data[col]))

    def _get_properties(self):
        """Return list of property names."""
        class_items = self.__class__.__dict__.items()
        return [k for k, v in class_items if isinstance(v, property)]

    @property
    def lat(self):
        return self.data["_lat"]

    @property
    def lon(self):
        return self.data["_lon"]

    @property
    def rad(self):
        return self.data["_radius_m"]

    @property
    def center(self):
        return self.data["geometry"]

    @property
    def _rbody(self):
        """Return average planetary radius from crs in meters."""
        ellipse = self._crs.ellipsoid
        return (ellipse.semi_major_metre + ellipse.semi_minor_metre) / 2

    def copy(self):
        """Return a deepcopy of a CraterDatabase."""
        return deepcopy(self)

    def add_annuli(self, name, inner, outer, **kwargs):
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

        Examples
        --------
        .. code-block:: python

            >>> cdb.add_annuli("name", 1, 2)  # generates annuli from each crater rim to 1 crater radius beyond the rim.
            >>> cdb.add_annuli("name", 1, 3)  # generates annuli from each crater rim to 1 crater diameter beyond the rim.
            >>> cdb.add_annuli("name", 0, 1)  # generates a cicle capturing the interior of the crater rim.

        """
        name = name or f"annulus_{inner}_{outer}"
        self.data[name] = self._gen_annulus(inner, outer, **kwargs)
        self._make_data_property(name)

    def add_circles(self, name="", size=1.0, **kwargs):
        """Generate circluar geometries for each crater in database.

        Parameters
        ----------
        name : str
            Name of geometry column (default: circle_{size}).
        size : int or float
            Radius of circle around each crater in crater radii (default: 1).
        """
        name = name or f"circle_{size}"
        self.add_annuli(inner=0, outer=size, name=name, **kwargs)

    def get_stats(self, rasters, regions, stats=STATS, nodata=None, n_jobs=-2):
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
        results = ch.get_stats(self.data, rasters, regions, stats, nodata, n_jobs)
        return pd.concat([self.data[self.orig_cols], results], axis=1)

    def plot(
        self,
        fraster=None,
        region="",
        ax=None,
        size=7.5,
        dpi=100,
        band=1,
        alpha=0.5,
        color="tab:blue",
        savefig=None,
        savefig_kwargs=None,
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
        savefig : str, optional
            If provided, save the figure to this path.
        savefig_kwargs : dict, optional
            Additional keyword arguments to pass to plt.savefig().
        **kwargs
            Keyword arguments supplied to GeoSeries.plot().

        Returns
        -------
        ax : matplotlib.Axes, cartopy.mpl.geoaxes.GeoAxes
        """
        ellipsoid = ccrs.CRS(self._crs).ellipsoid
        globe = ccrs.Globe(
            semimajor_axis=ellipsoid.semi_major_metre,
            semiminor_axis=ellipsoid.semi_minor_metre,
            ellipse=None,
        )
        proj = ccrs.PlateCarree(globe=globe)
        if ax is None:
            minx, miny, maxx, maxy = self.data.total_bounds
            aspect = (maxx - minx) / (maxy - miny)
            figsize = (size, size / aspect)
            _, ax = plt.subplots(
                figsize=figsize,
                dpi=dpi,
                subplot_kw={"projection": proj},
            )
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            gl = ax.gridlines(draw_labels=True)
            gl.top_labels = False
            label = " " + region if not region.startswith("_") else ""
            ax.set_title(str(self) + label)
        elif isinstance(ax, mpli.AxesImage):
            ax = ax.axes  # result of imshow is AxesImage, need the matplotlib Axes
        figsize = ax.get_figure().get_size_inches()
        dpi = ax.get_figure().get_dpi()
        if fraster:
            with rio.open(fraster) as src:
                height_npix = int(figsize[1] * dpi)
                width_npix = int(figsize[0] * dpi)
                # By default does nearest-neighbor interp to out_shape (super fast reads for low dpi)
                data = src.read(indexes=band, out_shape=(height_npix, width_npix))
                extent = ch.bbox2extent(src.bounds)
                # raster_crs = ccrs.CRS(src.crs)
                # TODO: assumes simple cylindrical, recognize and support other projections
                ax.imshow(data, cmap="gray", extent=extent, transform=proj)
        if not region:
            # Store crater rims with leading "_" to not be mistaken for user-defined ROIs
            region = "_plot_circles"
            if region not in self.data.columns:
                self.data[region] = self._gen_annulus(0, 1)
        rois = self.data.loc[:, region].boundary  # plot outline of ROI
        ax = rois.plot(ax=ax, alpha=alpha, color=color, autolim=False, **kwargs)
        if savefig is not None:
            # Set up default savefig options that can be overridden
            save_options = {"dpi": dpi, "bbox_inches": "tight"}
            if savefig_kwargs:
                save_options.update(savefig_kwargs)
            plt.savefig(savefig, **save_options)

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
            geom_col = "geometry"
        elif region in self.data.columns:
            geom_col = region
        else:
            raise ValueError(f"Geometry column '{region}' not found.")

        # Determine which columns to keep
        must_keep = set(self.orig_cols)
        if keep_cols is None:
            # Take all columns if keep cols otherwise only take cols without leading "_"
            keep_cols = [
                col
                for col in self.data.columns
                if keep_all or col in must_keep or not col.startswith("_")
            ]
        else:
            # Check if any specified cols don't exist
            missing_cols = [col for col in keep_cols if col not in self.data.columns]
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
            data[col] = data.apply(lambda x, col=col: x[col].wkt, axis=1)
            # TODO: there should be a better way to store these without all the naming hacks
            data.rename(columns={col: col + "_wkt"}, inplace=True)
        data.rename(columns={geom_col + "_wkt": geom_col + "_active_wkt"}, inplace=True)

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
            return None
        # Make a copy of the whole instance and do to_crs inplace on the copy
        new_crater_db = self.copy()
        new_crater_db.to_crs(crs, inplace=True)
        return new_crater_db

    def plot_rois(
        self,
        fraster,
        region,
        index=9,
        grid_kw=None,
        savefig=None,
        savefig_kwargs=None,
        **kwargs,
    ):
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
        savefig : str, optional
            If provided, save the figure to this path.
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

        # TODO: Display in AzEq projection instead to show antimeridian crossing and polar rois
        # Filter out polar or antimeridian crossing rois
        polar = (gdf.bounds.miny < -89) | (gdf.bounds.maxy > 89)
        oob = gdf.bounds.maxx - gdf.bounds.minx >= 300
        if any(polar | oob):
            warnings.warn(
                "Skipping ROIs that cross pole or antimeridian...", stacklevel=2
            )
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
        shp_kw = {"color": "tab:green", "facecolor": "none", "lw": 2, "alpha": 0.5}
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
        i = 0
        axs = np.atleast_1d(axes).flatten()
        while i < n:
            img = next(rois)["mini_raster_array"]
            if img.count() == 0:
                n -= 1
                continue
            geom = gdf.geometry.iloc[i]
            extent = ch.bbox2extent(geom.bounds)
            axs[i].imshow(img, extent=extent, aspect="auto", **im_kw)
            axs[i].add_feature(ShapelyFeature(geom, crs=pc, **shp_kw))
            i += 1
        # Format valid axes, delete unused axes
        for i, ax in enumerate(axs):
            if i >= n:
                fig.delaxes(ax)
            else:
                # Parse gridline kws, overwrite defaults if given
                grid_kw = {} if grid_kw is None else grid_kw
                gl_kw = {"draw_labels": True, "dms": False, "ls": "--", "alpha": 0.5}
                gl_kw = {**gl_kw, **grid_kw}
                gl = ax.gridlines(**gl_kw)
                gl.top_labels = False
                gl.right_labels = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
        fig.suptitle(f"{self.body.capitalize()} {region} ROIs", y=0.92)
        if savefig is not None:
            plt.savefig(savefig)
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
            raise ValueError("Cannot merge CraterDatabases from different bodies!")
        merged = ch.merge(cdb1.data, cdb2.data, rbody=cdb1._rbody)
        return cls(merged, body=cdb1.body, units="m")

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
                    lambda x, col=col: shapely.wkt.loads(x[col]), axis=1
                )
            if col.endswith("_active_wkt"):
                data.drop(columns="geometry", inplace=True)
        data.rename(
            columns={
                c: c.removesuffix("_wkt").removesuffix("_active") for c in data.columns
            },
            inplace=True,
        )

        # Create and return a new CraterDatabase instance
        return cls(data, body=body, units=file_units)
