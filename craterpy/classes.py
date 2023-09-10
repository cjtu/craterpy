import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
import craterpy.helper as ch

# TODO: decide how to handle geodataframe annuli
# Geopandas only allow 1 geometry per row, so we can't have annuli and craters in the rows
# Either we have to have 1 row per annulus and flatten df or 1 gdf per crater (seems excessive)
# Can it be a multiindex somehow? or multipolygon - don't think that works with rasterstats?
#  Thought: keep crater point geom and compute poly on the fly?

class CraterData:
    def __init__(self, filepath, crs='', annuli=(1, 2)):
        self.filepath = filepath
        self.crs = crs
        self.annuli = annuli
        self._load_data()
        self.latcol, self.loncol, self.radcol = ch.get_crater_cols(self.data)

    def _load_data(self):
        if self.filepath.endswith('.csv'):
            df = pd.read_csv(self.filepath)
            lat, lon, rad = ch.get_crater_cols(df)
            geometry = [Point(xy).buffer(radius) for xy, radius in zip(zip(df[lon], df[lat]), df[rad])]
            self.data = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)
        else:
            self.data = gpd.read_file(self.filepath, crs=self.crs)

    def _generate_annuli(self):
        """Generate annulus for each row in dataframe."""
        radii = self.data[self.radcol]
        center = Point(self.data[self.loncol], self.data[self.latcol])
        annuli = center.buffer(outer*radii).difference(center.buffer(inner*radii))




    def __repr__(self):            
        return self.data.__repr__()
    


