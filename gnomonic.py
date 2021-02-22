import numpy as np
from numpy import pi, sin, cos
from scipy.ndimage import map_coordinates


def lon_lat2indices(lon, lat, rows, cols, rettype=None):
    """ 
    This function converts lat and lon i.e.[-pi/2, pi/] x [-pi,pi]
    to image indices  (r, c) in [0, rows] x [0, cols]
    --------------
    Returns r and c either as integers or as floats to be interpolated
    """
    r = ((rows - 1)*(0.5-lat/pi)) % rows
    c = ((cols - 1)* (lon+pi)/(2*pi)) % cols
    return (int(r), int(c)) if rettype == int else (r,c)
    
def gnomonic_proj(lon, lat, lon0=0, lat0=0):
    """
    lon, lat : arrays of the same shape; longitude and latitude
               of points to be projected
    lon0, lat0: floats, longitude and latitude in radians for
                the tangency  point
    ---------------------------
    Returns the gnomonic projection, x, y
    https://mathworld.wolfram.com/GnomonicProjection.html
    """
    
    cosc = sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0)
    x = cos(lat)*sin(lon-lon0)/cosc
    y = (cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0))/cosc
    return x, y    
    
def inv_gnomonic_proj(x, y, lon0=0, lat0=0):
    """
    x, y - arrays of the same shape; coordinate of points
           in the projection plane
    ---------       
    Returns the lon and lat corresponding to x and y 
    """
    
    rho = np.sqrt(x**2 + y**2)
    c = np.arctan(rho)
    cosc = cos(c)
    sinc = sin(c)
    glat = np.arcsin(cosc * sin(lat0) +\
                     (y * sinc * cos(lat0)) / rho)
    glon = lon0 + np.arctan2(x * sinc,
           (rho * cos(lat0) * cosc - y * sin(lat0) * sinc))
    return glon, glat  
    
def equi2gnomonic(img, look_at=[0,0], FOV = [60, 30]): 
    """
    img - array of ndim at least 2
    look_at - tuple; lon and lat in degrees, of the tangency point
    FOV - tuple; lon, and lat in degrees for the field of view
    -----------------------
    Returns the projected sub-image defined by look_at and FOV
    """
    
    if 2*FOV[0] >=180 or 2*FOV[1] >= 90:
        raise ValueError("The FOV exceeds an open semisphere")
    if img.ndim < 2:
        raise ValueError("img must be an image")
    nrows, ncols = img.shape[:2] if img.ndim >2 else img.shape
    
    lon0, lat0 = np.radians(look_at)
    lon_fov, lat_fov = np.radians(FOV)
    
    # upper-right corner coords of the FOV projection 
    #        onto the tangent  plane:
    
    xu, yu = gnomonic_proj(lon_fov, lat_fov)
    X, Y = np.meshgrid(np.linspace(-xu, xu, ncols), 
                       np.linspace(-yu, yu, nrows))
    glon, glat = inv_gnomonic_proj(X, Y, lon0, lat0)
    glon, glat = glon.reshape(-1), glat.reshape(-1)
    #map glon, glat to column, respectively row indices
    grows, gcols = lon_lat2indices(glon, glat, nrows, ncols) 
    coords = np.vstack((grows, gcols)) 
    proj_idx = np.zeros((3, coords.shape[1]))
    for k in range(3):
        proj_idx[k, :] = map_coordinates(img[:, :, k], 
                                         coords, 
                                         order=3, 
                                         mode="nearest")
    return np.stack([np.flipud(proj_idx[k].reshape(nrows, ncols)) 
                     for k in range(3)], axis=-1)    
        
