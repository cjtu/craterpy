"""This file contains helper functions for plotting."""

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs


def plot_CraterRoi(croi, figsize=((4, 4)), title=None, cmap="gray", **kwargs):
    """
    Plot 2D CraterRoi.

    The plot offers limited arguments for basic customization. It is further
    customizable by supplying valid matplotlib.imshow() keyword-arguments. See
    matplotlib.imshow for full documentation.

    Parameters
    ----------
    roi : CraterRoi object
        2D CraterRoi to plot.
    figsize : tuple
        Length and width of plot in inches (default 4in x 4in).
    title : str
        Plot title.
    cmap : str
        Color map to plot (default 'gray'). See matplotlib.cm for full list.

    Other parameters
    ----------------
    **kwargs : object
        Keyword arguments to pass to imshow. See matplotlib.pyplot.imshow
    """
    if not title:
        title = "CraterRoi at ({}, {})".format(croi.lat, croi.lon)
    plt.figure(title, figsize=figsize)
    plt.imshow(croi.roi, extent=croi.extent, cmap=cmap, **kwargs)
    plt.title(title)
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.show()

    def plot_region(fraster, region, row):
        """Plot a region of a crater in CraterDatabase."""

    #     def b2e(bounds):
    #         """Reorder bounds from [w, s, e, n] to extent [x0, x1, y0, y1]."""
    #         return bounds[0], bounds[2], bounds[1], bounds[3]

    #     # TODO: Switch to only plotting what is clipped by zonal stats
    #     img = plt.imread(fraster)  # Will crash for very large rasters
    #     # rois = zonal_stats(data.set_geometry(region), fraster, stats='count',
    #     #                    raster_out=True)

    #     for i, crater in self.data.loc[subset].iterrows():
    #         lat = self.lat.loc[i]
    #         lon = self.lon.loc[i]
    #         crs = ProjectedCRS(name=f'AzimuthalEquidistant({lat:.2f}N, {lon:.2f}E)',
    #                                   conversion=AzimuthalEquidistantConversion(lat, lon),
    #                                   geodetic_crs=self._crs)
    #         globe = ccrs.Globe(semimajor_axis=crs.ellipsoid.semi_major_metre,
    #                            semiminor_axis=crs.ellipsoid.semi_minor_metre,
    #                            ellipse=None)
    #         ae = ccrs.AzimuthalEquidistant(lon, lat, globe=globe)
    #         pc = ccrs.PlateCarree(globe=globe)

    #         # Plot
    #         fig = plt.figure(figsize=(4, 4))
    #         ax = plt.subplot(111, projection=ae)
    #         ax.imshow(img, extent=(-180,180,-90,90), transform=pc, cmap='gray')

    #         # Crop to region
    #         ax.set_extent(b2e(crater[region].bounds), crs=pc)
    #         ax.set_boundary(mpath.Path(crater[region].exterior.coords), transform=pc)

    #         # Draw region
    #         ax.add_geometries(crater[region], crs=pc, facecolor='none', edgecolor='r', linewidth=2, alpha=0.8)
    #         ax.gridlines()
