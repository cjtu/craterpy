"""Unittest classes.py."""

import warnings
from pathlib import Path
import unittest
from tempfile import NamedTemporaryFile
import json
import os
import pyproj
from pyproj import CRS
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib.axes import Axes
import shapely
from shapely.testing import assert_geometries_equal
from shapely.geometry import Point
import craterpy
from craterpy.classes import CraterDatabase, CRS_DICT


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def setUp(self):
        self.moon_tif = craterpy.sample_data["moon.tif"]
        self.moon_craters = craterpy.sample_data["moon_craters.csv"]
        self.vesta_tif = craterpy.sample_data["vesta.tif"]
        self.vesta_craters = craterpy.sample_data["vesta_craters.csv"]
        self.moon_cdb = CraterDatabase(self.moon_craters, "moon", units="km")
        self.vesta_cdb = CraterDatabase(self.vesta_craters, "vesta_claudia_dp")

    def test_add_circles_annuli(self):
        """Test adding annular geometries to CraterDataBase."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("crater", 1)
        cdb.add_annuli("ejecta", 1, 2)

        # Check that ejecta appears in the string repr
        self.assertIn("ejecta", str(cdb))

        # Test that ejecta was registered as a propety and contains a shapely geom
        self.assertIsInstance(cdb.ejecta[0], shapely.geometry.Polygon)

    def test_annuli_precision(self):
        """Test that simple and precise annuli mostly agree."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("precise", 1, precise=True)
        cdb.add_annuli("simple", 0, 1, precise=False)
        assert_geometries_equal(cdb.precise, cdb.simple, tolerance=0.5)

    def test_get_stats(self):
        """Test getting statistics on a region for a raster."""
        cdb = self.moon_cdb.copy()
        cdb.add_annuli("rim", 1, 1.1)
        stats = cdb.get_stats(self.moon_tif, "rim", ["count"])
        self.assertIn("count_rim", stats.columns)

    def test_get_stats_parallel(self):
        """Test parallellization of get_stats for multiple rasters/regions."""
        pass

    def test_plot(self):
        """Test CraterDatabase summary plot."""
        cdb = self.moon_cdb.copy()
        ax = cdb.plot()
        self.assertIsInstance(ax, Axes)

    def test_body_crs_all(self):
        """Test that every defined CRS loads."""
        for body in CRS_DICT.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Vesta*")
                cdb = CraterDatabase(self.moon_craters, body)
                for crs in [
                    v for k, v in cdb.__dict__.items() if k.startswith("_crs")
                ]:
                    self.assertIsInstance(crs, pyproj.CRS)

    def test_vesta_coord_correction(self):
        """Test Vesta's various coordinate systems."""
        df = pd.DataFrame(
            {"lat": [0, 10], "lon": [-90, 120], "radius": [1, 2]}
        )
        cdb = CraterDatabase(df, "vesta_claudia_double_prime")  # 0 offset
        self.assertEqual(cdb.lon.iloc[0], -90)
        self.assertEqual(cdb.lon.iloc[1], 120)
        cdb = CraterDatabase(df, "vesta_claudia")  # +150
        self.assertEqual(cdb.lon.iloc[0], 60)
        self.assertEqual(cdb.lon.iloc[1], -90)
        cdb = CraterDatabase(df, "vesta_claudia_prime")  # +190
        self.assertEqual(cdb.lon.iloc[0], 100)
        self.assertEqual(cdb.lon.iloc[1], -50)
        with self.assertRaises(NotImplementedError):
            cdb = CraterDatabase(df, "vesta_iau_2000")

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

    def test_gen_annulus_simple(self):
        """Test generating simple annuli."""
        pass

    def test_gen_annulus_precise(self):
        """Test generating precise annuli."""
        pass

    def test_antimeridian_splitting(self):
        """Test generating annuli when split over the antimeridian."""
        pass

    def test_pole_crossing(self):
        """Test annuli that cross the North or South pole."""
        pass

    def test_to_geojson_basic(self):
        """Test basic export to GeoJSON string."""
        cdb = self.moon_cdb.copy()

        # Default export of center point geometry
        with NamedTemporaryFile(suffix=".geojson") as fcenter:
            cdb.to_geojson(fcenter.name)
            gdf = gpd.read_file(fcenter.name)  # inspect the written file
            self.assertEqual(len(cdb), len(gdf))
            self.assertIn("_center_active_wkt", gdf.columns)

            newdb = CraterDatabase.read_shapefile(fcenter.name)
            self.assertIn("_center", newdb.data.columns)
            self.assertNotIn("geometry", newdb.data.columns)
            self.assertEqual(newdb.body, "Moon")

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
            self.assertIn("_center", newdb.data.columns)
            self.assertNotIn("geometry", newdb.data.columns)

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
            self.assertIn("_center", newdb.data.columns)
            self.assertIn("_test", newdb.data.columns)
            self.assertIn("Lon", newdb.data.columns)
            self.assertNotIn("geometry", newdb.data.columns)

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
            cdb.to_geojson(fcrs.name, region="crater", crs=cdb._crs180)
            gdf = gpd.read_file(fcrs.name)
            self.assertEqual(gdf.crs, cdb.to_crs(cdb._crs180).data.crs)

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
        moon_equirect_crs = CRS.from_user_input(CRS_DICT["moon"][1])

        # Test with CRS object
        converted_cdb = CraterDatabase.to_crs(cdb, moon_equirect_crs)
        self.assertNotEqual(converted_cdb.data.crs, initial_crs)
        self.assertEqual(converted_cdb.data.crs, moon_equirect_crs)
        self.assertIsNot(converted_cdb, cdb)

        # Test data preservation (only need to test once)
        np.testing.assert_array_almost_equal(
            converted_cdb.lat.values, cdb.lat.values
        )
        np.testing.assert_array_almost_equal(
            converted_cdb.lon.values, cdb.lon.values
        )
        np.testing.assert_array_almost_equal(
            converted_cdb.rad.values, cdb.rad.values
        )

        # Test with string CRS input
        str_converted = CraterDatabase.to_crs(cdb, CRS_DICT["moon"][1])
        self.assertEqual(str_converted.data.crs, moon_equirect_crs)

        # Test converting to same CRS
        same_crs = CraterDatabase.to_crs(cdb, initial_crs)
        self.assertEqual(same_crs, cdb)

        # Test with north pole projection
        moon_north_crs = CRS.from_user_input(CRS_DICT["moon"][3])
        pole_converted = CraterDatabase.to_crs(cdb, moon_north_crs)
        self.assertEqual(pole_converted.data.crs, moon_north_crs)
        self.assertEqual(len(pole_converted.data), len(cdb.data))

    def test_to_crs_vesta(self):
        """Test Vesta-specific coordinate system handling."""
        df = pd.DataFrame(
            {
                "lat": [10.0, -20.0],
                "lon": [45.0, -60.0],
                "radius": [1000.0, 2000.0],
            }
        )
        cdb = CraterDatabase(df, body="vesta_claudia_dp")

        # Test Vesta coordinate system preservation
        self.assertTrue(hasattr(cdb, "_vesta_coord"))
        self.assertEqual(cdb._vesta_coord, "vesta_claudia_dp")

        # Test preservation through CRS conversion
        vesta_equirect_crs = CRS.from_user_input(CRS_DICT["vesta"][1])
        converted_cdb = CraterDatabase.to_crs(cdb, vesta_equirect_crs)
        self.assertEqual(converted_cdb._vesta_coord, "vesta_claudia_dp")

    def test_plot(self):
        """Test CraterDatabase plotting functionality."""
        cdb = self.moon_cdb.copy()

        # Test basic plotting
        ax = cdb.plot()
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_xlabel(), "Longitude")
        self.assertEqual(ax.get_ylabel(), "Latitude")

        # Test plotting with ROI region
        cdb.add_circles("test_region")
        ax = cdb.plot(region="test_region", color="red", alpha=0.8)
        self.assertIn("test_region", ax.get_title())

        # Test plotting with raster backdrop
        ax = cdb.plot(self.moon_tif, size=4, dpi=50)
        self.assertEqual(len(ax.images), 1)  # Has raster image
        self.assertEqual(len(ax.collections), 1)  # Has ROI overlay

    def test_plot_rois(self):
        """Test ROI subplot generation and filtering."""
        cdb = self.moon_cdb.copy()
        cdb.add_circles("test_region")

        # Test basic ROI plotting
        axes = cdb.plot_rois(self.moon_tif, "test_region", index=2)
        self.assertTrue(
            all(
                isinstance(ax, Axes) for ax in axes.flatten() if ax is not None
            )
        )

        # Test different index types
        axes = cdb.plot_rois(self.moon_tif, "test_region", index=[0, 2])
        self.assertTrue(
            all(
                isinstance(ax, Axes) for ax in axes.flatten() if ax is not None
            )
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
            with self.assertRaises(Warning):
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
            all(
                isinstance(ax, Axes) for ax in axes.flatten() if ax is not None
            )
        )
