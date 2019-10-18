"""This is a general purpose module containing routines
(a) that are used in multiple notebooks; or
(b) that are complicated and would thus otherwise clutter notebook design.
"""

import re
import socket
import numpy as np
from datetime import datetime
import intake
import xarray as xr
import cftime


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
    if 'lon' not in dsvar.dims:
        lonDim = 'longitude'
    else:
        lonDim = 'lon'
    if 'lat' not in dsvar.dims:
        latDim = 'latitude'
    else:
        latDim = 'lat'
    if float(dsvar[lonDim].min().values) < 0.:
        raise ValueError('Not expecting longitude values less than 0.')
    # Subset data into a box
    dsvar_subset = dsvar.sel(lat=slice(lat_bounds[0], lat_bounds[1]),
                             lon=slice(lon_bounds[0], lon_bounds[1]))
    # Get weights (cosine(latitude))
    w = np.cos(np.deg2rad(dsvar_subset[latDim]))
    # Ensure weights are the same shape as the data array
    w = w.broadcast_like(dsvar_subset)
    # Mask out NaN values in weighting matrix
    w = w.where(~np.isnan(dsvar_subset))
    # Convolve weights with data array
    x = (dsvar_subset*w).sum(dim=['lat', 'lon']) / w.sum(dim=['lat', 'lon'])

    return x


def get_anomaly(dsvar):
    '''
    get_anomaly(dsvar)

    remove climatological annual cycle to compute anomalies


    Parameters
    ----------
    dsvar : data array variable


    Returns
    -------
    NewArray : DataArray
        New DataArray with seasonal cycle removed.

    '''
    climatology = dsvar.groupby('time.month').mean('time')
    anomalies = dsvar.groupby('time.month') - climatology
    return anomalies


def get_decimal_time(dsvar):
    '''
    get_decimal_time(dsvar)

    get the time in decimal units (e.g., 1979.0438...)


    Parameters
    ----------
    dsvar : data array variable


    Returns
    -------
    dtime : ndarray
        Array of time axis in decimal units

    '''
    dtime = []
    for t in dsvar.time:
        # differentiate between datetime64 and datetime objects
        if isinstance(t.item(), int):
            t = datetime.utcfromtimestamp(t.item()/1e9)
        else:
            t = t.item()
        doy = t.timetuple().tm_yday
        dtime.append(t.year + doy / 365.)
    dtime = np.array(dtime)
    return dtime


def spatial_trends(dsvar, anom=True):
    '''
    spatial_trends(dsvar)

    get a map of spatial trends for a data array


    Parameters
    ----------
    dsvar : data array variable

    Optional Arguments
    ----------
    anom : boolean to specify whether anomalies should
           be computed (default is True)

    Returns
    -------
    NewArray : DataArray
        Array of trends (units / year)

    '''
    # get decimal time for trend calculations
    time_decimal = get_decimal_time(dsvar)
    # deal with weird lat / lon
    if 'lon' not in dsvar.dims:
        lonDim = 'longitude'
    else:
        lonDim = 'lon'
    if 'lat' not in dsvar.dims:
        latDim = 'latitude'
    else:
        latDim = 'lat'
    # compute anomalies
    if anom:
        dsvar_anom = np.array(get_anomaly(dsvar))
    else:
        dsvar_anom = np.array(dsvar)
    # compute trends
    m, b = np.polyfit(time_decimal, np.reshape(dsvar_anom, (len(time_decimal), -1)), 1)
    # transform into correct coordinates
    m = np.reshape(m, (len(dsvar[latDim]), len(dsvar[lonDim])))
    # get coordinates for new array (minus time)
    coords = []
    coords_str = []
    for dim in dsvar.dims:
        if dim == 'time':
            continue
        coords.append(dsvar[dim])
        coords_str.append(dim)
    # create a data array and add coordinates
    trend = xr.DataArray(m, coords=coords, dims=coords_str)
    # return data array
    return trend


def upscale(x,y,field,f):
    'Reduces resolution of field by factor f'
    
    xdel=(x[1]-x[0])/2
    xbin=np.zeros(int(len(x)/2)+1)
    xbin[0:-1]=x[::f].values-xdel.values
    xbin[-1]=x[-1].values+xdel.values
    
    ydel=(y[1]-y[0])/2
    ybin=np.zeros(int(len(y)/2)+1)
    ybin[0:-1]=y[::f].values-ydel.values
    ybin[-1]=y[-1].values+ydel.values
    
    xn=x.groupby_bins(group=x.name,bins=xbin).mean(x.name)
    yn=y.groupby_bins(group=y.name,bins=ybin).mean(y.name)
    fieldn=field.groupby_bins(group=x.name,bins=xbin).mean(x.name,skipna=True)\
        .groupby_bins(group=y.name,bins=ybin).mean(y.name,skipna=True)
    fieldn=fieldn.rename({'lat_bins':'lat','lon_bins':'lon'})
    fieldn[x.name].values=xn
    fieldn[y.name].values=yn

    return fieldn

def reindex_time(startingtimes):
    newtimes = startingtimes.values
    for i in range(0,len(startingtimes)):
        yr = int(str(startingtimes.values[i])[0:4])
        mon = int(str(startingtimes.values[i])[5:7])
        day = int(str(startingtimes.values[i])[8:10])
        hr = int(str(startingtimes.values[i])[11:13])
        newdate = cftime.DatetimeProlepticGregorian(yr,mon,15)
        newtimes[i]=newdate
    return newtimes


def _compute_slope(y):
    """
    Private function to compute slopes at each grid cell using
    polyfit.
    """
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0] # return only the slope


def compute_slope(da):
    """
    Computes linear slope (m) at each grid cell.
   
    Args:
      da: xarray DataArray to compute slopes for
     
    Returns:
      xarray DataArray with slopes computed at each grid cell.
    """
    # apply_ufunc can apply a raw numpy function to a grid.
    #
    # vectorize is only needed for functions that aren't already
    # vectorized. You don't need it for polyfit in theory, but it's
    # good to use when using things like np.cov.
    #
    # dask='parallelized' parallelizes this across dask chunks. It requires
    # an output_dtypes of the numpy array datatype coming out.
    #
    # input_core_dims should pass the dimension that is being *reduced* by this operation,
    # if one is being reduced.
    slopes = xr.apply_ufunc(_compute_slope,
                            da,
                            vectorize=True,
                            dask='parallelized',
                            input_core_dims=[['time']],
                            output_dtypes=[float],
                            )
    return slopes
