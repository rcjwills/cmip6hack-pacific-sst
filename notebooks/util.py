"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or 
(b) that are complicated and would thus otherwise clutter notebook design.
"""

import re
import socket
import numpy as np


def is_ncar_host():
    """Determine if host is an NCAR machine."""
    hostname = socket.getfqdn()

    return any([re.compile(ncar_host).search(hostname)
                for ncar_host in ['cheyenne', 'casper', 'hobart']])


def simple_spatial_average(dsvar, lat_bounds=[-90, 90], lon_bounds=[0, 360]):
    '''
    simple_spatial_average(dsvar)

    weighted average for DataArrays

    Function does not yet handle masked data.

    Parameters
    ----------
    dsvar : data array variable (with lat / lon axes)

    Optional Arguments
    ----------
    lat_bounds : list of latitude bounds to average over (e.g., [-20., 20.])
    lon_bounds : list of longitude bounds to average over (e.g., [0., 360.])

    Returns
    -------
    NewArray : DataArray
        New DataArray with proper spatial weighting.

    '''
    # Make sure lat and lon ranges are in correct order
    if lat_bounds[0] > lat_bounds[1]:
        lat_bounds = np.flipud(lat_bounds)
    if lon_bounds[0] > lon_bounds[1]:
        lon_bounds = np.flipud(lon_bounds)
    if float(dsvar.lon.min().values) < 0.:
        raise ValueError('Not expecting longitude values less than 0.')
    # Subset data into a box
    dsvar_subset = dsvar.sel(lat=slice(lat_bounds[0], lat_bounds[1]),
                             lon=slice(lon_bounds[0], lon_bounds[1]))
    # Get weights (cosine(latitude))
    w = np.cos(np.deg2rad(dsvar_subset.lat))
    # Ensure weights are the same shape as the data array
    w = w.broadcast_like(dsvar_subset)
    # Convolve weights with data array
    x = (dsvar_subset*w).mean(dim=['lat', 'lon']) / w.mean(dim=['lat', 'lon'])

    return x
