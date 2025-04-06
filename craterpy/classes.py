from copy import deepcopy
from typing import Union
import warnings
import functools
from pathlib import Path
import multiprocessing
import antimeridian
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
import craterpy.helper as ch
from rasterstats import zonal_stats

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

    Attributes:
        data (GeoDataFrame): GeoDataFrame containing the crater data.
        lat (Series): Crater latitudes.
        lon (Series): Crater longitudes.
        rad (Series): Crater radii.
        center (GeoSeries): Crater center (shapely.geometry Point).
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
        body: str = "Moon",
        units: str = "m",
    ):
        """
        Initialize a CraterDatabase.

        Parameters:
            dataset (str or DataFrame):
                if str, path to the file containing crater data.
                if DataFrame, DataFrame containing crater data.
            body (str): Planetary body, e.g. Moon, Vesta (default: Moon)
            units (str): Length units of radius/diameter, m or km (default: m)

        Raises:
            ValueError: If dataset is not a file or DataFrame.
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
        try:
            r = ch.findcol(self.data, ["radius", "rad", "r_km", "r_m"])
            self.data["_radius_m"] = pd.to_numeric(self.data[r])
        except ValueError:
            d = ch.findcol(self.data, ["diameter", "diam", "d_km", "d_m"])
            self.data["_radius_m"] = pd.to_numeric(self.data[d]) / 2
        self._radcol = "_radius_m"

        # Ensure lon is in -180 to 180
        self.data[self._loncol] = self.lon - lon_offset
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

        Returns:
            bool: True if the two objects are equal, False otherwise.
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
            out.loc[:] = self._gen_annulus_simple(
                inner, outer, self._crs, self._crs180, **kwargs
            )

            # Fix large annuli that get warped, also those that cross antimeridian
            # Overly clever bit of cos(lat) math that says:
            #   - don't fix annuli with radii <= 25 deg at the equator (no warping)
            #   - we always want to use the precise method at the poles
            #   - polewards of lat=60, annuli with radius >= 1 degree should use precise method
            is_large = (
                50
                * out.minimum_bounding_radius()
                * (1 - np.cos(np.radians(self.lat)))
                > 25
            )  # [deg]
            is_anti = out.bounds["maxx"] - out.bounds["minx"] >= 300
            out.loc[is_large | is_anti] = self._gen_annulus_precise(
                self.center[is_large | is_anti],
                self.rad[is_large | is_anti],
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

        Examples:
        - cdb.add_annuli("name", 1, 2) generates annuli from each crater rim to 1 crater radius beyond the rim.
        - cdb.add_annuli("name", 1, 3) generates annuli from each crater rim to 1 crater diameter beyond the rim.
        - cdb.add_annuli("name", 0, 1) generates a cicle capturing the interior of the crater rim.
        """
        name = name or f"annulus_{inner}_{outer}"
        self.data[name] = self._gen_annulus(inner, outer, precise)
        self._make_data_property(name)

    def add_circles(self, name="", size=1, precise=True):
        """Generate circluar geometries for each crater in database.

        Parameters
        ----------
        name: str
            Name of geometry column (default: circle_{size}).
        size: int or float
            Radius of circle around each crater in crater radii (default: 1).
        precise: bool
            Precisely calculate each geometry in a local projection (default: True).
        """
        name = name or f"circle_{size}"
        self.add_annuli(inner=0, outer=size, name=name, precise=precise)

    def _get_stats(
        self, fraster, region, stats=STATS, nodata=None, suffix=None
    ):
        """Return DataFrame of zonal stats on region from fraster."""
        zstats = zonal_stats(
            self.data.set_geometry(region), fraster, stats=stats, nodata=nodata
        )
        out = pd.DataFrame(
            {stat: [z[stat] for z in zstats] for stat in stats},
            index=self.data.index,
        )
        if suffix:
            out = out.add_suffix(f"_{suffix}")
        return out

    def get_stats(self, rasters, regions, stats=STATS, nodata=None, n_jobs=1):
        """Compute stats on polygons in a GeoDataFrame in parallel."""
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

    def plot(self, name="", ax=None, alpha=0.2, **kwargs):
        """Plot crater geometries.

        Parameters
        ----------
        name (optional): str
            Defined geometries to plot (default: simple crater circles).
        ax: matplotlib.Axes
            Axes on which to plot data.
        alpha: float
            Transparancy of the plot geometries (0-1, default: 0.2).
        **kwargs
            Keyword arguments supplied to GeoSeries.plot().

        Returns
        -------
        ax: matplotlib.Axes
        """
        if not name:
            # Note: add this directly to geodataframe to not conflict with user-set properties
            if "_plot_circles" not in self.data.columns:
                self.data["_plot_circles"] = self._gen_annulus(0, 1)
            # Plot outline only
            plotdata = self.data.loc[:, "_plot_circles"].boundary
        else:
            # Plot enclosed area
            plotdata = self.data.loc[:, name]
        ax = plotdata.plot(ax=ax, alpha=alpha, **kwargs)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        label = "." + name if name else ""
        ax.set_title(f"CraterDatabase{label} (N={len(self.data)})")
        return ax

    def to_geojson(
        self,
        filename=None,
        geometry_column=None,
        crs=None,
        properties=None,
        drop_private=True,
    ):
        """
        Export the crater database to GeoJSON format.

        Parameters
        ----------
        filename : str, optional
            Path to output GeoJSON file. If None, returns a string representation.
        geometry_column : str, optional
            Name of the geometry column to use as the active geometry.
            If None, uses the current active geometry.
        crs : str or pyproj.CRS, optional
            Target coordinate reference system. If None, uses the current CRS.
        properties : list, optional
            List of column names to include as properties. If None, includes all columns
            except geometry columns.
        drop_private : bool, optional
            If True, drops columns with names starting with an underscore (private columns).
            Default is True.

        Returns
        -------
        str or None
            If filename is None, returns the GeoJSON string.
            Otherwise, writes to the file and returns None.
        """

        # Determine which geometry column to use
        if geometry_column is None:
            # Use the current active geometry
            geom_col = self.data.geometry.name
        elif geometry_column in self.data.columns:
            # Use the specified geometry column
            geom_col = geometry_column
        else:
            # Specified column doesn't exist
            raise ValueError(f"Geometry column '{geometry_column}' not found.")

        # Determine which columns to keep
        if properties is not None:
            # Check if any specified properties don't exist
            missing_cols = [
                col for col in properties if col not in self.data.columns
            ]
            if missing_cols:
                raise ValueError(f"Properties not found: {missing_cols}")

            # Make sure geometry column is included in properties
            if geom_col not in properties:
                keep_cols = properties + [geom_col]
            else:
                keep_cols = properties
        else:
            # Keep all columns in their original order
            keep_cols = list(self.data.columns)

        # Remove private columns if requested
        if drop_private:
            keep_cols = [
                col
                for col in keep_cols
                if not (col.startswith("_") and col != geom_col)
            ]

        if crs is not None:
            converted_db = CraterDatabase.to_crs(self, crs)
            # Create filtered dataframe with only needed columns and set the geometry
            data = converted_db.data[keep_cols].set_geometry(geom_col)
        else:
            # Apply the same filtering/geometry update to the converted data
            data = self.data[keep_cols].set_geometry(geom_col)

        # Export to GeoJSON
        if filename is not None:
            data.to_file(filename, driver="GeoJSON")
            return None
        else:
            # Return as string
            return data.to_json()

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
            new_crater_db = deepcopy(self)
            new_crater_db.to_crs(crs, inplace=True)
            return new_crater_db

    @classmethod
    def read_shapefile(
        cls,
        filename,
        body: str = "Moon",
        units: str = "m",
        use_point_geom: bool = True,
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
        use_point_geom : bool, optional
            If True, use point geometry for lat/lon coordinates (default: True).
            Otherwise, use lat/lon columns directly (if they exist)

        Returns
        -------
        CraterDatabase
            A new CraterDatabase instance containing the data from the file.

        Notes
        -----
        This method assumes the file was previously created by CraterDatabase.to_geojson()
        or has a compatible format with lat/lon coordinates and radius or diameter information.

        If the file contains different coordinate column names than expected, this method
        will attempt to identify them by common names (e.g., 'lat', 'latitude', 'lon', 'longitude').
        """
        # Try to read metadata from GeoJSON first (if it's a GeoJSON file)
        file_metadata = {}
        if filename.lower().endswith(".geojson"):
            import json

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

        # If geometry is Point type, extract lat/lon from geometry
        if use_point_geom and all(
            geom.geom_type == "Point" for geom in data.geometry
        ):
            data.insert(0, "_Lon", data.geometry.x)
            data.insert(0, "_Lat", data.geometry.y)

        # Create and return a new CraterDatabase instance
        return cls(data, body=body, units=file_units)
