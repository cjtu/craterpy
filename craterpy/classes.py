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

# CRS for units / coord transformations (convention is only Ocentric )
CRS_DICT = {
    # body: (Geographic2D [deg], Equirect [m] (clon=0), Equirect [m] (clon=180),  Npole Stereo [m], Spole Stereo [m])
    'moon': ('IAU_2015:30100', 'IAU_2015:30110', 'IAU_2015:30115', 'IAU_2015:30130', 'IAU_2015:30135'),
    'mars': ('IAU_2015:49900', 'IAU_2015:49910', 'IAU_2015:49915', 'IAU_2015:49930', 'IAU_2015:49935'),
    'mercury': ('IAU_2015:19900', 'IAU_2015:19910', 'IAU_2015:19915', 'IAU_2015:19930', 'IAU_2015:19935'),
    'venus': ('IAU_2015:29900', 'IAU_2015:29910', 'IAU_2015:29915', 'IAU_2015:29930', 'IAU_2015:29935'),
    'europa': ('IAU_2015:50200', 'IAU_2015:50210', 'IAU_2015:50215', 'IAU_2015:50230', 'IAU_2015:50235'),
    'ceres': ('IAU_2015:200000100', 'IAU_2015:200000110', 'IAU_2015:200000115', 'IAU_2015:200000130', 'IAU_2015:200000135'),
    'vesta': ('IAU_2015:200000400', 'IAU_2015:200000410', 'IAU_2015:200000415', 'IAU_2015:200000430', 'IAU_2015:200000435'),
}

class CraterDatabase:
    def __init__(self, filepath, body='Moon', units='m'):
        self.filepath = filepath
        self.crs, self.crs_180, self.crs_360, self.crs_north, self.crs_south = self._load_crs(body)
        self.data = gpd.read_file(self.filepath)

        # Store reference to lat, lon, rad columns
        cols = ch.get_crater_cols(self.data)
        for col in cols:
            # Fixes lat, lon, rad columns if loaded as a string or object
            self.data[col] = pd.to_numeric(self.data[col])
        self.latcol, self.loncol, self.radcol = cols

        # Ensure lon is in -180 to 180
        self.data[self.loncol] = ch.lon180(self.data[self.loncol])

        # Convert to meters
        if units == 'km':
            self.data[self.radcol] *= 1000  

        # Generate point geometry for each row
        self.data['_point'] = self._generate_point()

        # Set geometry and covert to GeoDataFrame if not already
        if not isinstance(self.data, gpd.GeoDataFrame):
            self.data = gpd.GeoDataFrame(self.data, geometry='_point', crs=self.crs)
        elif 'geometry' not in self.data.columns:
            self.data.set_geometry('_point', inplace=True)

        # Generate crater rim geometry for each row
        self.data['_rim'] = self._generate_rim()


    def __repr__(self):            
        return self.data.__repr__()
    
    def _load_crs(self, body):
        """Return the pyproj CRSs for the body."""
        return [CRS.from_user_input(crs) for crs in CRS_DICT[body.lower()]]
    
    def _generate_point(self):
        """Return point geometry (lon, lat) for each row."""
        # Note: list comprehension is faster than df.apply
        return [Point(xy) for xy in zip(self.data[self.loncol], self.data[self.latcol])]

    def _generate_rim(self):
        """Return rim geometry for each row."""
        # Assumes radius is in same units as projection base unit (e.g., meters)
        # TODO: make circle more precise with more vertices at cost of memory? e.g. pt.buffer(rad, 128)

        # Subset craters to wrap around 180 and compute N / S properly
        # # Add option to do local azimuthal equidistant projection for each crater for better accuracy
        # near = self.data[self.loncol]
        # far =
        # north =
        # south =

        tmp = self.data.to_crs(self.crs_180)
        return tmp._point.buffer(tmp[self.radcol]).to_crs(self.crs)
        # return [pt.buffer(rad) for pt, rad in zip(self.data['_point'], self.data[self.radcol])]

    def _generate_annulus(self, inner, outer):
        """Generate annulus for each row in dataframe."""

        return [
            pt.buffer(rad * outer).difference(
                pt.buffer(rad * inner)
            ) for pt, rad in zip(self.data['_point'], self.data[self.radcol])
        ]
    
    def _generate_annulus_proj(self, inner, outer):
        # Generate annulus centered on each crater in the dataframe
        annuli = [Point(0, 0).buffer(rad * outer).difference(
                  Point(0, 0).buffer(rad * inner)
                  ) for rad in self.data[self.radcol]]

        for i, (_, row) in enumerate(self.data.iterrows()):
            center = row['_point']

            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f'AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)', 
                conversion=AzimuthalEquidistantConversion(center.y, center.x), 
                geodetic_crs=self.crs)

            # Project the annulus from local azeq centered to main CRS
            transformer = Transformer.from_crs(local_crs, self.crs, always_xy=True)
            annuli[i] = transform(transformer.transform, annuli[i])
        return annuli

    def zonal_stats(self, rasters, roi='ejecta', **kwargs):
        # TODO: write and test this
        if roi == 'ejecta':
            tmpdf = self.data['annulus'].rename({'annulus': 'geometry'})
            
        # Yields generator - figure out where we want to store all the data
        for raster in rasters:
            out = gen_zonal_stats(tmpdf, raster, **kwargs)
            break

        return list(out)


