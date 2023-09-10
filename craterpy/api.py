from typing import List, Optional, Union
import pandas as pd
import rasterio

def compute_stats(crater_df: Union[str, pd.DataFrame], rasters: List[str], ejrad: Optional[float] = 2, floormask: Optional[bool] = True, stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute statistics on each raster for each row in the dataframes.

    Parameters:
    crater_df (pd.DataFrame): DataFrame containing locations of craters.
    rasters (List[str]): List of file paths to raster files.
    stats (Optional[List[str]]): List of statistics to compute. If None, compute all statistics.

    Returns:
    pd.DataFrame: DataFrame containing statistics for each raster for each row in the dataframes.
    """
    # Implementation goes here
