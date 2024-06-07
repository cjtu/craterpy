import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
import craterpy.helper as ch
from rasterstats import gen_zonal_stats

# TODO: decide how to handle geodataframe annuli
# Geopandas only allow 1 geometry per row, so we can't have annuli and craters in the rows
# Either we have to have 1 row per annulus and flatten df or 1 gdf per crater (seems excessive)
# Can it be a multiindex somehow? or multipolygon - don't think that works with rasterstats?
#  Thought: keep crater point geom and compute poly on the fly?


class CraterDatabase:
    def __init__(self, filepath, crs=''):
        self.filepath = filepath
        self.crs = CRS.from_user_input(crs)
        self.data = self._load_data()
        self.geometry = {}  # Store generated geometries here

    def __repr__(self):            
        return self.data.__repr__()
    
    def _load_data(self):
        """Load crater database from file."""
        gdf = gpd.read_file(self.filepath)
        self.latcol, self.loncol, self.radcol = ch.get_crater_cols(gdf)
        for col in [self.latcol, self.loncol, self.radcol]:
            gdf[col] = pd.to_numeric(gdf[col])
        if isinstance(gdf, pd.core.frame.DataFrame):
            geometry = [Point(xy) for xy in zip(gdf[self.loncol], gdf[self.latcol])]
            gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs=self.crs)
        return gdf

    def _generate_point(self):
        
        pass

    def _generate_rim(self):
        pass

    def _generate_floor(self):
        pass

    def _generate_peak(self):
        pass

    def _generate_ejecta(self):
        pass

    def _generate_annulus(self, inner, outer):
        """Generate annulus for each row in dataframe."""
        rims = []
        annuli = []
        for _, row in self.data.iterrows():
            center = row['geometry']
            radius = row[self.radcol] * 1000 # [km] -> [m]

            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f'AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)', 
                conversion=AzimuthalEquidistantConversion(center.y, center.x), 
                geodetic_crs=self.crs)

            # Old way with proj4 strings / dict (bad)
            # local_crs_dict = self.crs.to_dict()
            # local_crs_dict['proj'] = 'aeqd'
            # local_crs_dict['lat_0'] = center.y
            # local_crs_dict['lon_0'] = center.x
            # local_crs = CRS.from_dict(local_crs_dict)
            # local_crs = CRS.from_string(f"+proj=aeqd +lat_0={center.y} +lon_0={center.x} +x_0=0 +y_0=0 +datum={self.crs.datum} +ellps={self.crs.ellipsoid}")
            
            # Create circle in crater azimuthally centered projection (0,0)
            rim = Point(0, 0).buffer(radius)
            inner_edge = Point(0, 0).buffer(radius * inner)
            outer_edge = Point(0, 0).buffer(radius * outer)
            annulus = outer_edge.difference(inner_edge)
            
            # Project the annulus back to the original CRS
            transformer = Transformer.from_crs(local_crs, self.crs, always_xy=True)
            annulus = transform(transformer.transform, annulus)
            rim = transform(transformer.transform, rim)

            # # Create transformers for the forward and reverse transformations
            # transformer_to_local = Transformer.from_crs(self.crs, local_crs, always_xy=True)
            # transformer_from_local = Transformer.from_crs(local_crs, self.crs, always_xy=True)
            annuli.append(annulus)
            rims.append(rim)
        self.data['rim'] = gpd.GeoSeries(rims, index=self.data.index)
        self.data['annulus'] = gpd.GeoSeries(annuli, index=self.data.index)

    def zonal_stats(self, rasters, roi='ejecta', **kwargs):
        # TODO: write and test this
        if roi == 'ejecta':
            tmpdf = self.data['annulus'].rename({'annulus': 'geometry'})
            
        # Yields generator - figure out where we want to store all the data
        for raster in rasters:
            out = gen_zonal_stats(tmpdf, raster, **kwargs)
            break

        return list(out)


