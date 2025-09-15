"""Unittest classes.py."""

import json
import os
import unittest
import warnings
from tempfile import NamedTemporaryFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import shapely
from matplotlib.axes import Axes
from pyogrio.errors import DataSourceError
from pyproj import CRS
from shapely.geometry import Point

import craterpy
from craterpy.classes import CraterDatabase
from craterpy.crs import ALL_BODIES


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def setUp(self):
        self.moon_tif = craterpy.sample_data["moon.tif"]
        self.moon_dem = craterpy.sample_data["moon_dem.tif"]
        self.moon_craters = craterpy.sample_data["moon_craters_km.csv"]
        self.vesta_tif = craterpy.sample_data["vesta.tif"]
        self.vesta_craters = craterpy.sample_data["vesta_craters_km.csv"]
        self.moon_cdb = CraterDatabase(self.moon_craters, "moon", units="km")
        self.moon_cdb_rim = self.moon_cdb.copy()
        self.moon_cdb_rim.add_annuli("rim", 1, 1.1)
        self.vesta_cdb = CraterDatabase(self.vesta_craters, "vesta", units="km")

    def test_add_circles_annuli(self):
        """Test adding annular geometries to CraterDataBase."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("crater", 1)
        cdb.add_annuli("ejecta", 1, 2)

        # Check that ejecta appears in the string repr
        self.assertIn("ejecta", cdb.__repr__())

        # Test that crater was registered as a propety and contains a shapely geom
        self.assertIsInstance(cdb.crater[0], shapely.geometry.Polygon)

        # Test that ejecta was registered as a propety and contains a shapely geom
        self.assertIsInstance(cdb.ejecta[0], shapely.geometry.Polygon)

    def test_get_stats(self):
        """Test getting statistics on a region for a raster."""
        stats = self.moon_cdb_rim.get_stats(self.moon_tif, "rim")
        self.assertIn("count_rim", stats.columns)
        self.assertIn("Latitude", stats.columns)
        self.assertFalse(any(stats["mean_rim"].isna()))  # No null values

    def test_get_stats_list_rasters(self):
        """Test parallellization of get_stats for multiple rasters in list."""
        stats = self.moon_cdb_rim.get_stats(
            [self.moon_tif, self.moon_dem], ["rim"], nodata=0
        )
        self.assertIn("count_raster0_rim", stats.columns)
        self.assertIn("count_raster1_rim", stats.columns)
        self.assertFalse(any(stats["count_raster0_rim"].isna()))

    def test_get_stats_multi_rasters_regions(self):
        """Test parallellization of get_stats for multiple rasters/regions."""
        stats = self.moon_cdb_rim.get_stats(
            {"moon": self.moon_tif, "dem": self.moon_dem},
            ["rim"],
            ["median", "count"],
            nodata={"moon": 0, "dem": 0},
        )
        self.assertIn("count_moon_rim", stats.columns)
        self.assertIn("count_dem_rim", stats.columns)
        self.assertFalse(any(stats["median_moon_rim"].isna()))

    def test_body_all_bodies_default_crs(self):
        """Test that every defined CRS loads."""
        for body in ALL_BODIES:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Vesta*")
                cdb = CraterDatabase(self.moon_craters, body)
                for crs in [v for k, v in cdb.__dict__.items() if k.startswith("_crs")]:
                    self.assertIsInstance(crs, pyproj.CRS)

    def test_vesta_coord_correction(self):
        """Test Vesta's various coordinate systems."""
        df = pd.DataFrame({"lat": [0, -1.6], "lon": [-90, 146], "radius": [1, 2]})
        cdb = CraterDatabase(df, "vesta", "claudia_dp")  # 0 offset
        self.assertEqual(cdb.lon.iloc[0], -90)
        self.assertEqual(cdb.lon.iloc[1], 146)
        cdb = CraterDatabase(df, "vesta", "claudia_p")  # -10
        self.assertAlmostEqual(cdb.lon.iloc[0], -100)
        self.assertAlmostEqual(cdb.lon.iloc[1], 136)
        cdb = CraterDatabase(df, "vesta", "dawn_claudia")  # +210
        self.assertAlmostEqual(cdb.lon.iloc[0], 120)
        self.assertAlmostEqual(cdb.lon.iloc[1], -4)

    def test_units(self):
        """Test importing with radii in m or km."""
        df_km = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radius": [1.0]})
        cdb_km = CraterDatabase(df_km, "moon", units="km")
        # Expecting radius to be converted to meters
        self.assertAlmostEqual(cdb_km.rad.iloc[0], 1000.0)

        df_m = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radius": [1000.0]})
        cdb_m = CraterDatabase(df_m, "moon", units="m")
        self.assertAlmostEqual(cdb_m.rad.iloc[0], 1000.0)

    def test_rad_vs_diam(self):
        """Test importing with radii and diam columns."""
        # Create a DataFrame with a diameter column instead of radius.
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
        # Diameter in meters; expecting radius = 5 m.
        for d in ("diameter", "diam", "d_m", "d_km", "d"):
            test_df = df.copy()
            test_df[d] = 10.0
            cdb = CraterDatabase(test_df, "moon")
            self.assertEqual(cdb.rad.iloc[0], 5.0)

        for r in ("radius", "rad", "r (km)", "r_m", "r"):
            test_df = df.copy()
            test_df[r] = 10.0
            cdb = CraterDatabase(test_df, "moon")
            self.assertEqual(cdb.rad.iloc[0], 10.0)

    def test_bool_and_eq(self):
        """Test __bool__ and __eq__ methods."""
        # Test empty database
        empty_df = pd.DataFrame({"lat": [], "lon": [], "radius": []})
        empty_cdb = CraterDatabase(empty_df, "moon")
        self.assertFalse(bool(empty_cdb))

        # Test non-empty database
        self.assertTrue(bool(self.moon_cdb))

        # Test equality with same data
        cdb1 = self.moon_cdb.copy()
        cdb2 = self.moon_cdb.copy()
        self.assertEqual(cdb1, cdb2)

        # Test inequality with different data
        cdb2.data.loc[0, "lat"] = 0.0
        self.assertNotEqual(cdb1, cdb2)

        # Test inequality with different type
        self.assertNotEqual(cdb1, "not a CraterDatabase")

    def test_gen_point_edge_cases(self):
        """Test _gen_point with various edge cases."""
        # Test with points at poles
        df = pd.DataFrame(
            {"lat": [90.0, -90.0], "lon": [0.0, 180.0], "radius": [1.0, 1.0]}
        )
        cdb = CraterDatabase(df, "moon")
        points = cdb._gen_point()
        self.assertEqual(len(points), 2)
        self.assertTrue(all(isinstance(p, Point) for p in points))

    def test_to_geojson_basic(self):
        """Test basic export to GeoJSON string."""
        cdb = self.moon_cdb.copy()

        # Default export of center point geometry
        with NamedTemporaryFile(suffix=".geojson") as fcenter:
            cdb.to_geojson(fcenter.name)
            gdf = gpd.read_file(fcenter.name)  # inspect the written file
            self.assertEqual(len(cdb), len(gdf))

            newdb = CraterDatabase.read_shapefile(fcenter.name)
            newdb.add_circles("import_test")
            self.assertIn("import_test", newdb.data.columns)
            self.assertIn("geometry", newdb.data.columns)
            self.assertEqual(newdb.body, "moon")

    def test_to_geojson_with_custom_geometry(self):
        """Test export with a custom geometry column."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("rim")

        # Export with crater rim geometries
        with NamedTemporaryFile(suffix=".geojson") as frim:
            cdb.to_geojson(frim.name, region="rim")
            gdf = gpd.read_file(frim.name)
            self.assertEqual(len(cdb), len(gdf))
            self.assertIn("rim_active_wkt", gdf.columns)

            newdb = CraterDatabase.read_shapefile(frim.name)
            self.assertIn("rim", newdb.data.columns)
            self.assertIn("geometry", newdb.data.columns)

    def test_to_geojson_with_keep_cols(self):
        """Test export with specific columns saved."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("crater", 1)
        cdb.data["_test"] = 1

        # Test keepcols
        with NamedTemporaryFile(suffix=".geojson") as fkeep:
            cdb.to_geojson(fkeep.name, region="crater", keep_cols=["_test"])
            gdf = gpd.read_file(fkeep.name)
            self.assertEqual(len(cdb), len(gdf))
            self.assertIn("_test", gdf.columns)
            self.assertIn("crater_active_wkt", gdf.columns)

            newdb = CraterDatabase.read_shapefile(fkeep.name)
            self.assertIn("geometry", newdb.data.columns)
            self.assertIn("_test", newdb.data.columns)
            self.assertIn("Longitude", newdb.data.columns)

    def test_to_geojson_file_export(self):
        """Test export to a GeoJSON file."""
        df = pd.DataFrame(
            {
                "lat": [0.0, 10.0],
                "lon": [0.0, 20.0],
                "radius": [1.0, 2.0],
                "name": ["Crater A", "Crater B"],
            }
        )
        cdb = CraterDatabase(df, "moon")

        # Create a temporary file
        with NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Export to the temp file
            result = cdb.to_geojson(filename=tmp_path)
            self.assertIsNone(result)  # Should return None when saving to file

            # Verify the file exists and contains valid GeoJSON
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r") as f:
                geojson_data = json.load(f)
                self.assertEqual(len(geojson_data["features"]), 2)
                features = geojson_data["features"]
                properties = [feature["properties"] for feature in features]
                self.assertTrue(
                    any(prop.get("name") == "Crater A" for prop in properties)
                )
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_to_geojson_with_crs_conversion(self):
        """Test export with coordinate reference system conversion."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("crater", 1)

        with NamedTemporaryFile(suffix=".geojson") as fcrs:
            cdb.to_geojson(fcrs.name, region="crater", crs="IAU_2015:30110")  # Moon eqr
            gdf = gpd.read_file(fcrs.name)
            self.assertEqual(gdf.crs, cdb.to_crs("IAU_2015:30110").data.crs)

    def test_to_geojson_error_handling(self):
        """Test error handling for invalid inputs."""
        cdb = self.moon_cdb.copy()

        # Test for invalid geometry column
        with self.assertRaises(ValueError):
            cdb.to_geojson("tmp.geojson", region="nonexistent")

        # Test for invalid properties
        with self.assertRaises(ValueError):
            cdb.to_geojson("tmp.geojson", keep_cols=["nonexistent"])

    def test_to_crs(self):
        """Test CRS conversions with different inputs and projections."""
        # Basic CRS conversion
        cdb = self.moon_cdb.copy()
        initial_crs = cdb.data.crs
        moon_equirect_crs = CRS.from_user_input("IAU_2015:30110")

        # Test with CRS object
        converted_cdb = CraterDatabase.to_crs(cdb, moon_equirect_crs)
        self.assertNotEqual(converted_cdb.data.crs, initial_crs)
        self.assertEqual(converted_cdb.data.crs, moon_equirect_crs)
        self.assertIsNot(converted_cdb, cdb)

        # Test data preservation (only need to test once)
        np.testing.assert_array_almost_equal(converted_cdb.lat.values, cdb.lat.values)
        np.testing.assert_array_almost_equal(converted_cdb.lon.values, cdb.lon.values)
        np.testing.assert_array_almost_equal(converted_cdb.rad.values, cdb.rad.values)

        # Test with string CRS input
        str_converted = CraterDatabase.to_crs(cdb, "IAU_2015:30110")
        self.assertEqual(str_converted.data.crs, moon_equirect_crs)

        # Test converting to same CRS
        same_crs = CraterDatabase.to_crs(cdb, initial_crs)
        self.assertEqual(same_crs, cdb)

        # Test with north pole projection
        moon_north_crs = CRS.from_user_input("IAU_2015:30130")
        pole_converted = CraterDatabase.to_crs(cdb, moon_north_crs)
        self.assertEqual(pole_converted.data.crs, moon_north_crs)
        self.assertEqual(len(pole_converted.data), len(cdb.data))

    def test_merge(self):
        """Test merging two CraterDatabases."""
        # Create two databases to merge
        df1 = pd.DataFrame({"lat": [0.0, 10.0], "lon": [0.0, 10.0], "rad": [1.0, 2.0]})
        df2 = pd.DataFrame(
            {"lat": [20.0, 30.0], "lon": [20.0, 30.0], "rad": [3.0, 4.0]}
        )
        cdb1 = CraterDatabase(df1, "moon")
        cdb2 = CraterDatabase(df2, "moon")

        # Test successful merge
        merged = CraterDatabase.merge(cdb1, cdb2)
        self.assertEqual(len(merged), len(cdb1) + len(cdb2))

        # Test merge with different bodies raises error
        cdb3 = CraterDatabase(df2, "vesta")
        with self.assertRaises(ValueError):
            CraterDatabase.merge(cdb1, cdb3)

    def test_read_shapefile_errors(self):
        """Test error handling in read_shapefile."""
        # Test with invalid file
        with self.assertRaises(DataSourceError):
            CraterDatabase.read_shapefile("nonexistent.geojson")

        # Test with missing required columns
        df = pd.DataFrame({"x": [1], "y": [2]})  # Missing lat/lon
        with NamedTemporaryFile(suffix=".geojson") as tmp:
            gpd.GeoDataFrame(df, geometry=[Point(1, 2)], crs="EPSG:4326").to_file(
                tmp.name, driver="GeoJSON"
            )
            with self.assertRaises(ValueError):
                CraterDatabase.read_shapefile(tmp.name)

    def test_read_shapefile_with_metadata(self):
        """Test reading shapefile with metadata."""
        # Create a GeoJSON with metadata
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {"body": "Moon", "units": "km"},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    "properties": {"lat": 0.0, "lon": 0.0, "radius": 1.0},
                }
            ],
        }

        ftmp = NamedTemporaryFile(suffix=".geojson", delete=False)  # Noqa
        ftmp.close()
        with open(ftmp.name, "w") as tmp:
            json.dump(geojson_data, tmp)
            tmp.flush()

        # Read with explicit parameters (should override metadata)
        cdb = CraterDatabase.read_shapefile(ftmp.name, body="Mars", units="m")
        self.assertEqual(cdb.body, "mars")

        # Read without parameters (should use metadata)
        cdb = CraterDatabase.read_shapefile(tmp.name)
        self.assertEqual(cdb.body, "moon")
        self.assertEqual(cdb.rad.iloc[0], 1000.0)  # Should be converted from km to m

    def test_plot(self):
        """Test CraterDatabase plotting functionality."""
        cdb = self.moon_cdb.copy()

        # Test basic plotting
        ax = cdb.plot()
        self.assertIsInstance(ax, Axes)
        self.assertEqual(len(ax.collections), 1)  # Has ROI overlay

        # Test plotting with ROI region
        cdb.add_circles("test_region")
        ax = cdb.plot(region="test_region", color="red", alpha=0.8)
        self.assertEqual(len(ax.collections), 1)  # Has ROI overlay

        # Test plotting with raster backdrop
        ax = cdb.plot(self.moon_tif, size=4, dpi=50)
        self.assertEqual(len(ax.images), 1)  # Has raster image
        self.assertEqual(len(ax.collections), 1)  # Has ROI overlay

        # Test plotting on existing raster axes
        plt.figure()
        with rio.open(self.moon_tif) as src:
            img = src.read(1)
        ax = plt.imshow(img, extent=(-180, 180, -90, 90))
        ax = cdb.plot(region="test_region", ax=ax)
        self.assertEqual(len(ax.images), 1)  # Has raster image
        self.assertEqual(len(ax.collections), 1)  # Has ROI overlay

        # Test saving plot to file
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test basic plot saving
            ax = cdb.plot(savefig=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)  # File should not be empty

            # Test plot saving with custom kwargs
            ax = cdb.plot(
                savefig=tmp_path,
                savefig_kwargs={"dpi": 300, "bbox_inches": None},
            )
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)

            # Test plot saving with raster backdrop
            ax = cdb.plot(self.moon_tif, savefig=tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_plot_rois(self):
        """Test ROI subplot generation and filtering."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("test_region")

        # Test basic ROI plotting
        axes = cdb.plot_rois(self.moon_tif, "test_region", index=2)
        self.assertTrue(
            all(isinstance(ax, Axes) for ax in axes.flatten() if ax is not None)
        )

        # Test different index types
        axes = cdb.plot_rois(self.moon_tif, "test_region", index=[0, 2])
        self.assertTrue(
            all(isinstance(ax, Axes) for ax in axes.flatten() if ax is not None)
        )

        # Test polar/antimeridian warning
        df = pd.DataFrame(
            {
                "lat": [89.5],
                "lon": [0],
                "radius": [1],
            }
        )
        polar_cdb = CraterDatabase(df, body="moon")
        polar_cdb.add_circles("test_region")

        with self.assertRaises(ValueError):
            polar_cdb.plot_rois(self.moon_tif, "test_region")

        # Test custom styling
        axes = cdb.plot_rois(
            self.moon_tif,
            "test_region",
            index=1,
            color="red",
            alpha=0.8,
            cmap="viridis",
            grid_kw={"alpha": 0.7},
        )
        self.assertTrue(
            all(isinstance(ax, Axes) for ax in axes.flatten() if ax is not None)
        )
