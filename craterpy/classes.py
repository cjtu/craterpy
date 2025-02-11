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
    """Database of crater locations and shapefiles.

    Attributes:
        data (GeoDataFrame): GeoDataFrame containing the crater data.
        lat (Series): Crater latitudes.
        lon (Series): Crater longitudes.
        rad (Series): Crater radii.
        center (GeoSeries): Crater center (shapely.geometry Point).
    """
    # Philosophy for database: Geopandas only allows 1 shape geometry per row
    # So, use the point geometry for each crater by default
    # Then switch to _rim / _annulus geometry on the fly for computing stats

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
    def __init__(self, filepath, body='Moon', units='m', simple=True):
        """
        Initialize a CraterDatabase.

        Parameters:
            filepath (str): Path to the file containing crater data.
            body (str): Planetary body, e.g. Moon, Vesta (default: Moon)
            units (str): Length units of radius/diameter, m or km (default: m)
            simple (bool): Constructs shapefiles using simple annuli instead of precise projected annuli (default: True).
        """
        lon_offset = 0
        if 'vesta' in body.lower():
            body, lon_offset = self._vesta_check(body)
            self._vesta_coord = body
            body = 'vesta'
        self._filepath = filepath
        self._crs, self._crs180, self._crs360, self._crsnorth, self._crssouth = self._load_crs(body)
        self.data = gpd.read_file(self._filepath)

        # Store reference to lat, lon, rad columns
        cols = ch.get_crater_cols(self.data)
        for col in cols:
            # Fixes lat, lon, rad columns if loaded as a string or object
            self.data[col] = pd.to_numeric(self.data[col])
        self._latcol, self._loncol, self.radcol = cols

        # Ensure lon is in -180 to 180
        self.data[self._loncol] = self.lon - lon_offset
        self.data[self._loncol] = ch.lon180(self.lon)

        # Convert to meters
        if units == 'km':
            self.data[self.radcol] *= 1000

        # Generate point geometry for each row
        self.data['_point'] = self._generate_point()

        # Set geometry and covert to GeoDataFrame if not already
        if not isinstance(self.data, gpd.GeoDataFrame):
            self.data = gpd.GeoDataFrame(self.data, geometry='_point', crs=self._crs)
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
            out = self._generate_annulus_simple(self.center, self.rad, inner, outer, self._crs, self._crs180)

            # To account for stretch near poles, re-process annuli polewards of 50 degrees
            is_n = self.lat > 50
            is_s = self.lat < -50
            out.loc[is_n] = self._generate_annulus_simple(self.center[is_n], self.rad[is_n], inner, outer, self._crs, self._crsnorth)
            out.loc[is_s] = self._generate_annulus_simple(self.center[is_s], self.rad[is_s], inner, outer, self._crs, self._crssouth)

            # Finally, correct large annuli that get inconsistently streched
            is_large = out.area > 25  # TODO: Is 25 degrees^2 a good threshold?
            out.loc[is_large] =  self._generate_annulus_precise(self.center[is_large], self.rad[is_large], inner, outer)
        else:
            out = self._generate_annulus_precise(self.center, self.rad, 0, 1)
        return out
 
    def _get_annular_buffer(self, pt, rad, inner, outer):
        """Generate annulus for each row in dataframe pt must be projected to crs in rad units."""
        # TODO: option to make circle more precise with more vertices at cost of memory? e.g. pt.buffer(rad, 128)
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
    
    def _generate_annulus_precise(self, centers, rads, inner, outer):
        """
        Generate annuli using a local azimuthal equidistant projection for each crater.
        
        This method is more precise than the simple method, but slower. It is
        recommended for high-precision work or when the annuli are large and
        span many degrees of latitude.
        """
        # Generate annulus centered on each crater in the dataframe
        annuli = []
        for center, rad, in zip(centers, rads):
            annulus = self._get_annular_buffer(Point(0, 0), rad, inner, outer)

            # Create a local azimuthal equidistant projection for the crater
            local_crs = ProjectedCRS(
                name=f'AzimuthalEquidistant({center.y:.2f}N, {center.x:.2f}E)', 
                conversion=AzimuthalEquidistantConversion(center.y, center.x), 
                geodetic_crs=self._crs)

            # Project the annulus from local azeq centered to main CRS
            transformer = Transformer.from_crs(local_crs, self._crs, always_xy=True)
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
        """Crater latitudes."""
        return self.data[self._latcol]
    
    @property
    def lon(self):
        """Crater longitudes."""
        return self.data[self._loncol]

    @property
    def rad(self):
        """Crater radii."""
        return self.data[self.radcol]
    
    @property
    def center(self):
        """Crater center point geometry."""
        return self.data['_point']

    # def zonal_stats(self, rasters, roi='ejecta', **kwargs):
    #     """Compute zonal statistics on all craters."""
    #     # TODO: finish writing this and test
    #     tmpdf = self.data
    #     if roi == 'ejecta':
    #         tmpdf = tmpdf['annulus'].rename({'annulus': 'geometry'}).set_geometry('geometry')
    #     elif roi == 'crater':
    #         tmpdf = tmpdf['_rim'].rename({'_rim': 'geometry'}).set_geometry('geometry')

    #     # Yields generator - figure out where we want to store all the data
    #     for raster in rasters:
    #         out = gen_zonal_stats(tmpdf, raster, **kwargs)
    #         break

    #     return list(out)
