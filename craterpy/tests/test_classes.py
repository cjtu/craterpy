"""Unittest classes.py."""

import warnings
from pathlib import Path
import unittest
import pyproj
from pyproj import CRS
import pandas as pd
import numpy as np
import shapely
from shapely.testing import assert_geometries_equal
from shapely.geometry import Point
import craterpy
from craterpy.classes import CraterDatabase, CRS_DICT


class TestCraterDatabase(unittest.TestCase):
    """TestCraterDatabase class."""

    def setUp(self):
        data_dir = Path(craterpy.__path__[0], "data")
        self.moon_tif = data_dir / "moon.tif"
        self.crater_list = data_dir / "craters.csv"
        self.vesta_crater_list = data_dir / "vesta_craters.csv"

    def test_add_annuli(self):
        """Test adding annular geometries to CraterDataBase."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli("ejecta", 1, 2)
        # Check that ejecta appears in the string repr
        self.assertIn("ejecta", str(cdb))
        # Test that ejecta was registered as a propety and contains a shapely geom
        self.assertIsInstance(cdb.ejecta[0], shapely.geometry.Polygon)

    def test_add_circles(self):
        """Test adding circular geometries to CraterDatabase"""
        pass

    def test_annuli_precision(self):
        """Test that simple and precise annuli mostly agree."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli("precise", 0, 1, precise=True)
        cdb.add_annuli("simple", 0, 1, precise=False)
        assert_geometries_equal(cdb.precise, cdb.simple, tolerance=1)

    def test_get_stats(self):
        """Test getting statistics on a region for a raster."""
        cdb = CraterDatabase(self.crater_list)
        cdb.add_annuli("rim", 1, 1.1)
        stats = cdb.get_stats(self.moon_tif, "rim", ["count"])
        self.assertIn("count_rim", stats.columns)

    def test_get_stats_parallel(self):
        """Test parallellization of get_stats for multiple rasters/regions."""
        pass

    def test_plot(self):
        """Test CraterDatabase summary plot."""
        # Create a minimal DataFrame
        df = pd.DataFrame({
            "lat": [0.0, 10.0],
            "lon": [0.0, 20.0],
            "radius": [1.0, 2.0]
        })
        cdb = CraterDatabase(df)
        cdb.add_annuli("test_annulus", 0, 1)
        # Generate the plot.
        ax = cdb.plot()
        self.assertIsNotNone(ax)
        from matplotlib.axes import Axes
        self.assertIsInstance(ax, Axes)

    def test_body_crs_all(self):
        """Test that every defined CRS loads."""
        for body in CRS_DICT.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Vesta*")
                cdb = CraterDatabase(self.crater_list, body)
                for crs in [
                    v for k, v in cdb.__dict__.items() if k.startswith("_crs")
                ]:
                    self.assertIsInstance(crs, pyproj.CRS)

    def test_vesta_coord_correction(self):
        """Test Vesta's various coordinate systems."""
        cdb = CraterDatabase(self.vesta_crater_list, "vesta_claudia_dp", "m")
        pass

    def test_import_dataframe(self):
        """Test importing from a dataframe."""
        # Create a minimal DataFrame with crater data.
        df = pd.DataFrame({
            "lat": [0.0, 10.0],
            "lon": [0.0, 20.0],
            "radius": [1.0, 2.0]
        })
        cdb = CraterDatabase(df)
        self.assertTrue(hasattr(cdb.data, "geometry"))
        self.assertEqual(len(cdb.data), 2)
        self.assertIsInstance(cdb.data.geometry.iloc[0], Point)
        self.assertAlmostEqual(cdb.data.geometry.iloc[0].x, 0.0)
        self.assertAlmostEqual(cdb.data.geometry.iloc[0].y, 0.0)

    def test_units(self):
        """Test importing with radii in m or km."""
        df_km = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radius": [1.0]})
        cdb_km = CraterDatabase(df_km, units="km")
        # Expecting radius to be converted to meters
        self.assertAlmostEqual(cdb_km.rad.iloc[0], 1000.0)

        df_m = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radius": [1000.0]})
        cdb_m = CraterDatabase(df_m, units="m")
        self.assertAlmostEqual(cdb_m.rad.iloc[0], 1000.0)

    def test_rad_vs_diam(self):
        """Test importing with radii and diam columns."""
        # Create a DataFrame with a diameter column instead of radius.
        
        # Diameter in meters; expecting radius = 5 m.
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0], "diameter": [10.0]})
        cdb = CraterDatabase(df, units="m")
        self.assertAlmostEqual(cdb.rad.iloc[0], 5.0)
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0], "diam": [10.0]})
        cdb = CraterDatabase(df, units="m")
        self.assertAlmostEqual(cdb.rad.iloc[0], 5.0)

        df = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radius": [10.0]})
        cdb = CraterDatabase(df, units="m")
        self.assertAlmostEqual(cdb.rad.iloc[0], 10.0)
        df = pd.DataFrame({"lat": [0.0], "lon": [0.0], "radii": [10.0]})
        cdb = CraterDatabase(df, units="m")
        self.assertAlmostEqual(cdb.rad.iloc[0], 10.0)

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
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"],
            "_private": [1, 2]
        })
        cdb = CraterDatabase(df)
        
        geojson_str = cdb.to_geojson()
        self.assertIsInstance(geojson_str, str)
        self.assertIn("Crater A", geojson_str)
        self.assertNotIn("_private", geojson_str)  # Private columns should be dropped by default

    def test_to_geojson_with_custom_geometry(self):
        """Test export with a custom geometry column."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0]
        })
        cdb = CraterDatabase(df)
        cdb.add_circles("rim")
        
        # Export with crater rim geometries
        geojson_with_rim = cdb.to_geojson(geometry_column="rim")
        self.assertIn("Polygon", geojson_with_rim)  # Rims should be polygons not points

    def test_to_geojson_with_specific_properties(self):
        """Test export with specific property columns."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"],
            "description": ["Description A", "Description B"]
        })
        cdb = CraterDatabase(df)
        
        # Export with only name property
        props_only = cdb.to_geojson(properties=["name"])
        self.assertIn("name", props_only)
        self.assertIn("Crater A", props_only)
        self.assertNotIn("description", props_only)
        self.assertNotIn("Description A", props_only)

    def test_to_geojson_with_private_columns(self):
        """Test export including private columns."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0],
            "_private": [1, 2],
            "_internal": ["x", "y"]
        })
        cdb = CraterDatabase(df)
        
        # By default, private columns should be dropped
        default_export = cdb.to_geojson()
        self.assertNotIn("_private", default_export)
        
        # With drop_private=False, they should be included
        with_private = cdb.to_geojson(drop_private=False)
        self.assertIn("_private", with_private)
        self.assertIn("_internal", with_private)

    def test_to_geojson_file_export(self):
        """Test export to a GeoJSON file."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"]
        })
        cdb = CraterDatabase(df)
        
        import tempfile
        import os
        import json
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Export to the temp file
            result = cdb.to_geojson(filename=tmp_path)
            self.assertIsNone(result)  # Should return None when saving to file
            
            # Verify the file exists and contains valid GeoJSON
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, 'r') as f:
                geojson_data = json.load(f)
                self.assertEqual(len(geojson_data['features']), 2)
                features = geojson_data['features']
                properties = [feature['properties'] for feature in features]
                self.assertTrue(any(prop.get('name') == "Crater A" for prop in properties))
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_to_geojson_with_crs_conversion(self):
        """Test export with coordinate reference system conversion."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0]
        })
        cdb = CraterDatabase(df, body="moon")
        
        # Get the current CRS
        original_crs = cdb.data.crs
        
        # Use a different valid CRS for the moon from CRS_DICT
        target_crs_string = CRS_DICT["moon"][1]  # Equirect (clon=0)
        
        # Export with a different CRS
        geojson_with_crs = cdb.to_geojson(crs=target_crs_string)
        
        # Verify we got a valid string that can be parsed as JSON
        self.assertIsInstance(geojson_with_crs, str)
        import json
        geojson_obj = json.loads(geojson_with_crs)
        self.assertIn("features", geojson_obj)


    def test_to_geojson_error_handling(self):
        """Test error handling for invalid inputs."""
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0]
        })
        cdb = CraterDatabase(df)
        
        # Test for invalid geometry column
        with self.assertRaises(ValueError):
            cdb.to_geojson(geometry_column="nonexistent")
        
        # Test for invalid properties
        with self.assertRaises(ValueError):
            cdb.to_geojson(properties=["nonexistent"])

    def test_read_shapefile_basic(self):
        """Test basic reading of a shapefile from a GeoJSON file."""
        # Create a simple CraterDatabase
        df = pd.DataFrame({
            "lat": [0.0, 10.0], 
            "lon": [0.0, 20.0], 
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"]
        })
        original_cdb = CraterDatabase(df)
        
        # Write to GeoJSON
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file
            original_cdb.to_geojson(filename=tmp_path)
            
            # Read back the file
            imported_cdb = CraterDatabase.read_shapefile(tmp_path)
            
            # Verify data was preserved
            self.assertEqual(len(imported_cdb.data), len(original_cdb.data))
            self.assertEqual(imported_cdb.data.shape[0], 2)
            # Check key attributes
            np.testing.assert_array_almost_equal(imported_cdb.lat.values, original_cdb.lat.values)
            np.testing.assert_array_almost_equal(imported_cdb.lon.values, original_cdb.lon.values)
            np.testing.assert_array_almost_equal(imported_cdb.rad.values, original_cdb.rad.values)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_to_geojson_with_custom_columns(self):
        """Test reading a shapefile with non-standard column names."""
        # Create a GeoDataFrame with non-standard column names
        import geopandas as gpd
        
        gdf = gpd.GeoDataFrame({
            "latitude_deg": [0.0, 10.0],
            "longitude_deg": [0.0, 20.0],
            "diameter_km": [2.0, 4.0],  # Diameters in km
            "crater_name": ["Crater A", "Crater B"],
            "geometry": [Point(0, 0), Point(20, 10)]
        }, crs="EPSG:4326")
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file with non-standard column names
            gdf.to_file(tmp_path, driver="GeoJSON")
            
            # Read the file with non-standard column names
            imported_cdb = CraterDatabase.read_shapefile(tmp_path, units="km")
            
            # Verify data was preserved and correctly interpreted
            self.assertEqual(len(imported_cdb.data), 2)
            
            # Check that our read_shapefile method correctly identified the columns
            np.testing.assert_array_almost_equal(imported_cdb.lat.values, [0.0, 10.0])
            np.testing.assert_array_almost_equal(imported_cdb.lon.values, [0.0, 20.0])
            
            # Since diameter was converted to radius in km and then to meters,
            # we expect: diameter_km / 2 * 1000 = [1000.0, 2000.0]
            np.testing.assert_array_almost_equal(imported_cdb.rad.values, [1000.0, 2000.0])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_shapefile_with_body_and_units(self):
        """Test reading a shapefile with body and units information in the file."""
        import geopandas as gpd
        import tempfile
        import os
        
        # Create a GeoDataFrame with body and units info
        gdf = gpd.GeoDataFrame({
            "lat": [0.0, 10.0],
            "lon": [0.0, 20.0],
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"],
            "body": ["Mars", "Mars"],  # Should override default body
            "units": ["km", "km"],     # Should override default units
            "geometry": [Point(0, 0), Point(20, 10)]
        }, crs="EPSG:4326")
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file
            gdf.to_file(tmp_path, driver="GeoJSON")
            
            # Read the file, specifying Moon as default (should be overridden by Mars)
            imported_cdb = CraterDatabase.read_shapefile(tmp_path, body="Moon", units="m")
            
            # Verify body and units were overridden from file
            # We can't directly compare the CRS name as it gets reformatted when the CRS object is created
            # Instead check that we're using Mars CRS by checking that it's not Moon CRS
            self.assertNotEqual(imported_cdb._crs.name, CRS.from_user_input(CRS_DICT["moon"][0]).name)
            self.assertEqual(imported_cdb._crs.name, CRS.from_user_input(CRS_DICT["mars"][0]).name)
            
            # Since units in file is km, radii should be converted to meters
            np.testing.assert_array_almost_equal(imported_cdb.rad.values, [1000.0, 2000.0])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_shapefile_with_planet_field(self):
        """Test reading a shapefile with 'planet' field instead of 'body'."""
        import geopandas as gpd
        import tempfile
        import os
        from pyproj import CRS
        
        # Create a GeoDataFrame with planet field instead of body
        gdf = gpd.GeoDataFrame({
            "lat": [0.0, 10.0],
            "lon": [0.0, 20.0],
            "radius": [1.0, 2.0],
            "planet": ["Europa", "Europa"],  # Should override default body
            "geometry": [Point(0, 0), Point(20, 10)]
        }, crs="EPSG:4326")
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file
            gdf.to_file(tmp_path, driver="GeoJSON")
            
            # Read the file, specifying Moon as default (should be overridden by Europa)
            imported_cdb = CraterDatabase.read_shapefile(tmp_path, body="Moon")
            
            # Verify body was overridden from file
            self.assertEqual(imported_cdb._crs.name, CRS.from_user_input(CRS_DICT["europa"][0]).name)  # Europa CRS
            self.assertNotEqual(imported_cdb._crs.name, CRS.from_user_input(CRS_DICT["moon"][0]).name)  # Not Moon CRS
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_shapefile_from_points_without_explicit_latlon(self):
        """Test reading a shapefile with Point geometry but no explicit lat/lon columns."""
        import geopandas as gpd
        import tempfile
        import os
        
        # Create a GeoDataFrame with only Point geometry, no explicit lat/lon columns
        gdf = gpd.GeoDataFrame({
            "radius": [1.0, 2.0],
            "name": ["Crater A", "Crater B"],
            "geometry": [Point(30, 15), Point(45, 20)]
        }, crs="EPSG:4326")
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file
            gdf.to_file(tmp_path, driver="GeoJSON")
            
            # Read the file
            imported_cdb = CraterDatabase.read_shapefile(tmp_path)
            
            # Verify lat/lon values were extracted from Point geometry
            np.testing.assert_array_almost_equal(imported_cdb.lat.values, [15.0, 20.0])
            np.testing.assert_array_almost_equal(imported_cdb.lon.values, [30.0, 45.0])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_shapefile_existing_radius_m_column(self):
        """Test reading a shapefile with _radius_m column already present."""
        import geopandas as gpd
        import tempfile
        import os
        
        # Create a GeoDataFrame with _radius_m column already present
        gdf = gpd.GeoDataFrame({
            "lat": [0.0, 10.0],
            "lon": [0.0, 20.0],
            "_radius_m": [5000.0, 7500.0],  # Already in meters
            "geometry": [Point(0, 0), Point(20, 10)]
        }, crs="EPSG:4326")
        
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save to file
            gdf.to_file(tmp_path, driver="GeoJSON")
            
            # Read the file
            imported_cdb = CraterDatabase.read_shapefile(tmp_path)
            
            # Verify radii were preserved
            np.testing.assert_array_almost_equal(imported_cdb.rad.values, [5000.0, 7500.0])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
