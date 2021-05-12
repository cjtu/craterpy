"""This file contains helper functions for plotting."""
import matplotlib.pyplot as plt


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
