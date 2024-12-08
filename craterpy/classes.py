from warnings import warn
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
    def __init__(self, filepath, body='Moon', units='m', simple=True):
        lon_offset = 0
        if 'vesta' in body.lower():
            body, lon_offset = self._vesta_check(body)
            self.vesta_coord = body
            body = 'vesta'
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
        self.data[self.loncol] = self.lon - lon_offset
        self.data[self.loncol] = ch.lon180(self.lon)

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
        self.data['_rim'] = self._generate_annulus(0, 1, simple)

    def __repr__(self):            
        return self.data.__repr__()
    
    def _load_crs(self, body):
        """Return the pyproj CRSs for the body."""
        return [CRS.from_user_input(crs) for crs in CRS_DICT[body.lower()]]
    
    def _generate_point(self):
        """Return point geometry (lon, lat) for each row."""
        # Note: list comprehension is faster than df.apply
        return [Point(xy) for xy in zip(self.lon, self.lat)]

    def _generate_annulus(self, inner, outer, simple=True):
        """Return rim geometry for each row."""
        # Assumes radius is in same units as projection base unit (e.g., meters)
        if simple:
            # Draws circles in simple cylindrical
            out = self._generate_annulus_simple(self.center, self.rad, inner, outer, self.crs, self.crs_180)

            # To account for stretch near poles, correct annuli polewards of 45
            is_n = self.lat > 45
            is_s = self.lat < -45
            out.loc[is_n] = self._generate_annulus_simple(self.center[is_n], self.rad[is_n], inner, outer, self.crs, self.crs_north)
            out.loc[is_s] = self._generate_annulus_simple(self.center[is_s], self.rad[is_s], inner, outer, self.crs, self.crs_south)

            # Finally, correct large annuli that will be inconsistently streched
            # TODO: pick a threshold and find craters with bounds > threshold to use precise method
        else:
            out = self._generate_annulus_precise(0, 1)
        return out
 
    def _get_annular_buffer(self, pt, rad, inner, outer):
        """Generate annulus for each row in dataframe pt must be projected to crs in rad units."""
        # TODO: make circle more precise with more vertices at cost of memory? e.g. pt.buffer(rad, 128)
        if inner == 0:
            return pt.buffer(rad * outer)
        else:
            return pt.buffer(rad * outer).difference(pt.buffer(rad * inner))
        
    def _generate_annulus_simple(self, pt, rad, inner, outer, src_crs, dst_crs):
        """Generate annulus naievely in one projection"""
        annuli = self._get_annular_buffer(pt.to_crs(dst_crs), rad, inner, outer)
        out = annuli.to_crs(src_crs)

        # Correct annuli that cross the antimeridian (make a huge poly stretching across the globe, i.e. lon>180)
        oob = out.bounds['maxx'] - out.bounds['minx'] >= 179
        out.loc[oob] = ch.unproject_split_meridian(annuli.loc[oob], src_crs, dst_crs)
        return out
    
    def _generate_annulus_precise(self, inner, outer):
        """
        Generate annuli using a local azimuthal equidistant projection for each crater.
        
        This method is more precise than the simple method, but slower. It is
        recommended for high-precision work or when the annuli are large and
        span many degrees of latitude.
        """
        # Generate annulus centered on each crater in the dataframe
        annuli = [] 
        for center, rad, in zip(self.center, self.rad):
            annulus = self._get_annular_buffer(Point(0, 0), rad, inner, outer)

            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f'AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)', 
                conversion=AzimuthalEquidistantConversion(center.y, center.x), 
                geodetic_crs=self.crs)

            # Project the annulus from local azeq centered to main CRS
            transformer = Transformer.from_crs(local_crs, self.crs, always_xy=True)
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
        if body in ['vesta_claudia_dp', 'vesta_claudia_double_prime']:
            return 'vesta_claudia_dp', 0
        elif body in ['vesta_claudia_p', 'vesta_claudia_prime']:
            return 'vesta_claudia_p', 190
        elif body in ['vesta_claudia', 'vesta_dawn_claudia']:
            return 'vesta_claudia', 150
        elif body in ['vesta_iau_2000', 'vesta_iau2000']:
            raise NotImplementedError('Vesta IAU 2000 coordinate system is not supported.')
        
        # Default to claudia_dp if no match
        warn('Vesta has multiple coordinate systems. Defaulting to vesta_claudia_dp... '
            'Specify one of (vesta_claudia_dp, vesta_claudia_p, vesta_claudia). to '
            'avoid this warning. Type help(CraterDatabase()._vesta_check)'
            'for more info.')
        return 'vesta_claudia_dp', 0

    @property
    def lat(self):
        return self.data[self.latcol]
    
    @property
    def lon(self):
        return self.data[self.loncol]

    @property
    def rad(self):
        return self.data[self.radcol]
    
    @property
    def center(self):
        return self.data['_point']

    def zonal_stats(self, rasters, roi='ejecta', **kwargs):
        # TODO: write and test this
        if roi == 'ejecta':
            tmpdf = self.data['annulus'].rename({'annulus': 'geometry'})
            
        # Yields generator - figure out where we want to store all the data
        for raster in rasters:
            out = gen_zonal_stats(tmpdf, raster, **kwargs)
            break

        return list(out)


