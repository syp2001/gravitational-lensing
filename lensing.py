import matplotlib.pylab as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import map_coordinates, rotate, shift
from astropy.io import fits
from matplotlib import path
from numba import jit
import numba as nb

def open_fits(file_name):
    data = fits.open(file_name)[0].data
    return np.flip(data,axis=0)

def interpolate(xlim,ylim,values,points):
    """
    Interpolate a 2D array of values at a given set of points.

    :param xlim: x limits of the 2D array :math:`[x_{min},x_{max}]`
    :type xlim: tuple
    :param ylim: y limits of the 2D array :math:`[y_{min},y_{max}]`
    :type ylim: tuple
    :param values: 2D array of values
    :type values: numpy.ndarray
    :param points: points at which to interpolate given as :math:`[[x_1,\dots,x_n],[y_1,\dots,y_n]]]`
    :type points: tuple

    :return: interpolated values
    :rtype: numpy.ndarray
    """
    n = values.shape[0]
    y_min, y_max = ylim
    x_min, x_max = xlim
    i = (y_max-points[1])/(y_max-y_min)*(n-1)
    j = (points[0]-x_min)/(x_max-x_min)*(n-1)
    return map_coordinates(values, np.array([i, j]), order=1)


def fast_griddata(points,values,xi):
    """
    A faster version of scipy.interpolate.griddata implemented using scipy.ndimage.map_coordinates.

    :param points: list of points at which the values are defined
    :type points: numpy.ndarray
    :param values: values to interpolate
    :type values: numpy.ndarray
    :param xi: list of points at which to interpolate
    :type xi: numpy.ndarray

    :return: interpolated values
    :rtype: numpy.ndarray
    """
    n = np.sqrt(points.shape[0]).astype(int)
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    i = (y_max-xi[:,1])/(y_max-y_min)*(n-1)
    j = (xi[:,0]-x_min)/(x_max-x_min)*(n-1)

    values_2d = np.reshape(values, (n,n))
    return map_coordinates(values_2d, np.array([i, j]), order=1)

def transform_image(image,angle,center,pixel_scale):
    """
    Rotates and shifts an image.

    :param image: image to transform
    :type image: numpy.ndarray
    :param angle: angle by which to rotate the image given in degrees
    :type angle: float
    :param center: new center of the image
    :type center: tuple
    :param pixel_scale: pixel scale of the image
    :type pixel_scale: float

    :return: transformed image
    :rtype: numpy.ndarray
    """
    pixel_shift = -center[1]/pixel_scale, center[0]/pixel_scale
    transformed_image = image
    transformed_image = rotate(transformed_image,angle,order=0,reshape=False)
    transformed_image = shift(transformed_image,pixel_shift,order=0)    
    # return al.Array2D.no_mask(transformed_image,pixel_scales=scale)
    return transformed_image

def list_of_points_from_grid(grid):
    """
    Flattens a grid into a list of points.

    :param grid: grid to flatten
    :type grid: numpy.ndarray

    :return: flattened grid
    :rtype: numpy.ndarray
    """
    grid_points = np.transpose(grid,(1,2,0))
    grid_points = np.reshape(grid_points,(-1,2))

    return grid_points

def deflection_angle_scale_factor(z1,z2,H0=70,Om0=0.3):
    """
    Compute the deflection angle scale factor :math:`D_{ds}/D_s` for a lens at redshift z1 and a source at redshift z2.

    :param z1: redshift of the lens
    :type z1: float
    :param z2: redshift of the source
    :type z2: float

    :return: deflection angle scale factor :math:`D_{ds}/D_s`
    :rtype: float
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    D_ds = cosmo.angular_diameter_distance_z1z2(z1,z2) # distance from source to lens
    D_s = cosmo.angular_diameter_distance_z1z2(0,z2) # distance to source

    return float(D_ds/D_s)

def grid(center,num_pix,pixel_scale):
    """
    Create a 2D grid of coordinates.

    :param center: center of the grid
    :type center: tuple
    :param fov: field of view of the grid
    :type fov: float
    :param scale: pixel scale of the grid
    :type scale: float

    :return: 2D grid of coordinates
    :rtype: numpy.ndarray
    """
    x_center = center[0]
    y_center = center[1]
    fov = num_pix*pixel_scale
    x = np.linspace(x_center - fov / 2 + pixel_scale / 2, x_center + fov / 2 - pixel_scale / 2, int(fov / pixel_scale))
    y = np.linspace(y_center + fov / 2 - pixel_scale / 2 , y_center - fov / 2 + pixel_scale / 2, int(fov / pixel_scale))
    return np.meshgrid(x, y)

def ray_trace_points(deflections_grid, x_deflections, y_deflections, points, z1=None, z2=None):
    """
    Trace a list of points from the image plane to the source plane

    :param deflections_grid: coordinate grid on which the deflection angles are defined
    :type deflections_grid: numpy.ndarray
    :param x_deflections: x component of the deflection angles
    :type x_deflections: numpy.ndarray
    :param y_deflections: y component of the deflection angles
    :type y_deflections: numpy.ndarray
    :param points: list of points to trace
    :type points: numpy.ndarray
    :param z1: redshift of the lens
    :type z1: float

    :return: list of traced points
    :rtype: numpy.ndarray
    """
    
    # compute deflection angle scale factor
    scale_factor = deflection_angle_scale_factor(z1,z2) if z1 is not None and z2 is not None else 1

    deflections_grid_points = list_of_points_from_grid(deflections_grid)

    # perform ray tracing by interpolating deflection angles
    traced_points_x = points[:,0] - scale_factor*fast_griddata(points=deflections_grid_points, values=x_deflections.flatten(), xi=points)
    traced_points_y = points[:,1] - scale_factor*fast_griddata(points=deflections_grid_points, values=y_deflections.flatten(), xi=points)

    return np.transpose(np.array([traced_points_x,traced_points_y]))



def ray_trace_grid(deflections_grid, x_deflections, y_deflections, image_plane, z1=None, z2=None):
    """
    Trace a grid of coordinates from the image plane to the source plane

    :param deflections_grid: coordinate grid on which the deflection angles are defined
    :type deflections_grid: numpy.ndarray
    :param x_deflections: x component of the deflection angles
    :type x_deflections: numpy.ndarray
    :param y_deflections: y component of the deflection angles
    :type y_deflections: numpy.ndarray
    :param image_plane: grid of image plane coordinates
    :type image_plane: numpy.ndarray
    :param z1: redshift of the lens
    :type z1: float
    :param z2: redshift of the source
    :type z2: float

    :return: grid of source plane coordinates
    :rtype: numpy.ndarray
    """

    # compute deflection angle scale factor
    scale_factor = deflection_angle_scale_factor(z1,z2) if z1 is not None and z2 is not None else 1

    # convert coordinate grids into list of points
    image_plane_points = list_of_points_from_grid(image_plane)
    deflections_grid_points = list_of_points_from_grid(deflections_grid)

    # perform ray tracing by interpolating deflection angles
    traced_points_x = image_plane_points[:,0] - scale_factor*fast_griddata(points=deflections_grid_points, values=x_deflections.flatten(), xi=image_plane_points)
    traced_points_y = image_plane_points[:,1] - scale_factor*fast_griddata(points=deflections_grid_points, values=y_deflections.flatten(), xi=image_plane_points)

    n = image_plane[0].shape[0]
    traced_points_x = np.reshape(traced_points_x, (n,n))
    traced_points_y = np.reshape(traced_points_y, (n,n))

    return traced_points_x, traced_points_y

@jit(nopython=True)
def get_nonempty_pixels(traced_corners,source_x_range,source_y_range):     
    image_pix = traced_corners.shape[0]-1
    nonempty_pixels = []
    for i in range(image_pix):
        for j in range(image_pix):
            top_left = traced_corners[i,j]
            if (top_left[1] < source_y_range[1]) and (top_left[1] > source_y_range[0]) and (top_left[0] < source_x_range[1]) and (top_left[0] > source_x_range[0]):
                nonempty_pixels.append((i,j))

    return nonempty_pixels

def get_arc_pixels(source_grid,traced_corners_grid,image_grid):
    """
    Returns the list of pixels on the image plane that fall on the source grid when traced back to the source plane.

    :param source_grid: grid of source plane coordinates
    :type source_grid: numpy.ndarray
    :param traced_corners_grid: grid of source plane coordinates of the corners of each pixel on the image plane
    :type traced_corners_grid: numpy.ndarray
    :param image_grid: grid of image plane coordinates
    :type image_grid: numpy.ndarray

    :return: list of image plane pixels
    """
    source_plane = np.transpose(a=source_grid,axes=(1,2,0))
    traced_corners = np.transpose(a=traced_corners_grid,axes=(1,2,0))
    image_plane = np.transpose(image_grid,axes=(1,2,0))
    
    y = source_plane[:,0,1]
    x = source_plane[0,:,0]
    source_x_range = [np.min(x),np.max(x)]
    source_y_range = [np.min(y),np.max(y)]
    
    nonempty_pixels = get_nonempty_pixels(traced_corners,nb.typed.List(source_x_range),nb.typed.List(source_y_range))

    return np.array([image_plane[index] for index in nonempty_pixels])


def get_traced_pixels(source_grid,traced_corners_grid):
    source_plane = np.transpose(a=source_grid,axes=(1,2,0))
    traced_corners = np.transpose(a=traced_corners_grid,axes=(1,2,0))

    y = source_plane[:,0,1]
    x = source_plane[0,:,0]
    source_x_range = [np.min(x),np.max(x)]
    source_y_range = [np.min(y),np.max(y)]
    nonempty_pixels = get_nonempty_pixels(traced_corners,nb.typed.List(source_x_range),nb.typed.List(source_y_range))
    polygons = []

    for r in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[r]
        top_left = traced_corners[i,j]
        top_right = traced_corners[i+1,j]
        bottom_right = traced_corners[i+1,j+1]
        bottom_left = traced_corners[i,j+1]
        polygons.append([top_left,top_right,bottom_right,bottom_left])

    return np.array(polygons)

def get_traced_luminosities(source_image,source_grid,traced_corners_grid):
    """
    Compute the lensed image of a source image.

    :param source_image: image of the source
    :type source_image: numpy.ndarray
    :param source_grid: grid of source plane coordinates
    :type source_grid: numpy.ndarray
    :param traced_corners_grid: grid of source plane coordinates of the corners of each pixel on the image plane
    :type traced_corners_grid: numpy.ndarray
    :param func: function used to compute the luminosity of a pixel
    :type func: function

    :return: lensed image
    :rtype: numpy.ndarray
    """
    source_plane = np.transpose(a=source_grid,axes=(1,2,0))
    traced_corners = np.transpose(a=traced_corners_grid,axes=(1,2,0))
    
    y = source_plane[:,0,1]
    x = source_plane[0,:,0]
    source_x_range = [np.min(x),np.max(x)]
    source_y_range = [np.min(y),np.max(y)]
    
    nonempty_pixels = get_nonempty_pixels(traced_corners,nb.typed.List(source_x_range),nb.typed.List(source_y_range))

    lum = np.zeros(len(nonempty_pixels))

    for r in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[r]
        top_left = traced_corners[i,j]
        top_right = traced_corners[i+1,j]
        bottom_right = traced_corners[i+1,j+1]
        bottom_left = traced_corners[i,j+1]

        vertices = [top_left,top_right,bottom_right,bottom_left]

        traced_pixel = path.Path(vertices)

        # compute bounding box
        x_min = min([v[0] for v in vertices])
        y_min = min([v[1] for v in vertices])
        x_max = max([v[0] for v in vertices])
        y_max = max([v[1] for v in vertices])

        y_index_range = [np.where(y-y_max > 0, y-y_max, np.inf).argmin(),np.where(y_min-y > 0, y_min-y, np.inf).argmin()]
        x_index_range = [np.where(x_min-x > 0, x_min-x, np.inf).argmin(),np.where(x-x_max > 0, x-x_max, np.inf).argmin()]

        image_slice = source_image[y_index_range[0]:y_index_range[1]+1,x_index_range[0]:x_index_range[1]+1]
        if image_slice.size == 0:
            lum[r] = 0
        else:
            source_plane_slice = source_plane[y_index_range[0]:y_index_range[1]+1,x_index_range[0]:x_index_range[1]+1]
            source_points = np.reshape(source_plane_slice,(-1,2))

            index = traced_pixel.contains_points(source_points)
            luminosity_values = image_slice.flatten()[index]
            if len(luminosity_values) == 0:
                lum[r] = 0
            else:
                lum[r] = np.mean(luminosity_values)
    
    return lum

def lens_image(source_image,source_grid,traced_corners_grid,func=np.mean):
    """
    Compute the lensed image of a source image.

    :param source_image: image of the source
    :type source_image: numpy.ndarray
    :param source_grid: grid of source plane coordinates
    :type source_grid: numpy.ndarray
    :param traced_corners_grid: grid of source plane coordinates of the corners of each pixel on the image plane
    :type traced_corners_grid: numpy.ndarray
    :param func: function used to compute the luminosity of a pixel
    :type func: function

    :return: lensed image
    :rtype: numpy.ndarray
    """
    source_plane = np.transpose(a=source_grid,axes=(1,2,0))
    traced_corners = np.transpose(a=traced_corners_grid,axes=(1,2,0))
    
    y = source_plane[:,0,1]
    x = source_plane[0,:,0]
    source_x_range = [np.min(x),np.max(x)]
    source_y_range = [np.min(y),np.max(y)]
    
    nonempty_pixels = get_nonempty_pixels(traced_corners,nb.typed.List(source_x_range),nb.typed.List(source_y_range))
    image_pix = traced_corners.shape[0]-1

    image = np.zeros((image_pix,image_pix))

    for r in range(len(nonempty_pixels)):
        i, j = nonempty_pixels[r]
        top_left = traced_corners[i,j]
        top_right = traced_corners[i+1,j]
        bottom_right = traced_corners[i+1,j+1]
        bottom_left = traced_corners[i,j+1]

        vertices = [top_left,top_right,bottom_right,bottom_left]

        traced_pixel = path.Path(vertices)

        # compute bounding box
        x_min = min([v[0] for v in vertices])
        y_min = min([v[1] for v in vertices])
        x_max = max([v[0] for v in vertices])
        y_max = max([v[1] for v in vertices])

        y_index_range = [np.where(y-y_max > 0, y-y_max, np.inf).argmin(),np.where(y_min-y > 0, y_min-y, np.inf).argmin()]
        x_index_range = [np.where(x_min-x > 0, x_min-x, np.inf).argmin(),np.where(x-x_max > 0, x-x_max, np.inf).argmin()]

        image_slice = source_image[y_index_range[0]:y_index_range[1]+1,x_index_range[0]:x_index_range[1]+1]
        if image_slice.size == 0:
            image[i,j] = 0
        else:
            source_plane_slice = source_plane[y_index_range[0]:y_index_range[1]+1,x_index_range[0]:x_index_range[1]+1]
            source_points = np.reshape(source_plane_slice,(-1,2))

            index = traced_pixel.contains_points(source_points)
            luminosity_values = image_slice.flatten()[index]
            if len(luminosity_values) == 0:
                image[i,j] = 0
            else:
                image[i,j] = np.mean(luminosity_values)
    
    return image


def plot_image(image,center,fov,ax=None,title=None,**kwargs):
    x_center, y_center = center
    if ax is None: fig, ax = plt.subplots(1, 1, dpi=150)
    im = ax.imshow(
    np.log10(
        image,
        where=(image != 0),
        out=np.full_like(image, -15),
    )+6,
    extent=[
         x_center - fov / 2,
         x_center + fov / 2,
         y_center - fov / 2,
         y_center + fov / 2,
     ],
    cmap="inferno",
    vmin=-5,
    vmax=-1,
    **kwargs)
    ax.set(
    xlabel="arcsec", ylabel="arcsec"
    )
    ax.set_facecolor("black")

    if title is not None: ax.set_title(title)
    return im

def magnification(magnification_grid, magnification_values, image_plane):
    """
    Interpolate the magnification on a grid of coordinates.

    :param magnification_grid: grid of magnifications
    :type magnification_grid: numpy.ndarray
    :param magnification: magnification value
    :type magnification: float
    :param image_plane: grid of image plane coordinates
    :type image_plane: numpy.ndarray

    :return: magnification line
    :rtype: numpy.ndarray
    """
    image_plane_points = list_of_points_from_grid(image_plane)
    magnification_grid_points = list_of_points_from_grid(magnification_grid)
    
    image_plane_magnification = fast_griddata(points=magnification_grid_points, values=magnification_values.flatten(), xi=image_plane_points)

    return np.reshape(image_plane_magnification, (image_plane[0].shape[0],image_plane[0].shape[1]))

def magnification_line(magnification_grid, magnification_values, image_plane,threshold=500):
    image_plane_points = list_of_points_from_grid(image_plane)
    image_plane_magnification = magnification(magnification_grid, magnification_values, image_plane)
    return image_plane_points[image_plane_magnification.flatten() > threshold]

def caustic(deflections_grid,x_deflections,y_deflections,magnification_grid, magnification_values, image_plane,z1=None, z2=None,threshold=500):
    """
    Compute the caustic curve of a lens.

    :param magnification_grid: grid of magnifications
    :type magnification_grid: numpy.ndarray
    :param magnification: magnification value
    :type magnification: float
    :param image_plane: grid of image plane coordinates
    :type image_plane: numpy.ndarray

    :return: caustic curve
    :rtype: numpy.ndarray
    """
    magnification_line_points = magnification_line(magnification_grid, magnification_values, image_plane,threshold=threshold)

    # compute deflection angle scale factor
    if z1 is not None and z2 is not None:
        scale_factor = deflection_angle_scale_factor(z1,z2)
    else:
        scale_factor = 1

    # convert coordinate grids into list of points
    deflections_grid_points = list_of_points_from_grid(deflections_grid)

    # perform ray tracing by interpolating deflection angles
    traced_points_x = magnification_line_points[:,0] - scale_factor*fast_griddata(points=deflections_grid_points, values=x_deflections.flatten(), xi=magnification_line_points)
    traced_points_y = magnification_line_points[:,1] - scale_factor*fast_griddata(points=deflections_grid_points, values=y_deflections.flatten(), xi=magnification_line_points)

    return np.transpose(np.array([traced_points_x,traced_points_y]))

def magnification_from_deflections(deflections_grid, x_deflections, y_deflections, image_plane, z1=None, z2=None):
    """
    Compute the magnification from the deflection angles.

    :param x_deflections: x component of the deflection angles
    :type x_deflections: numpy.ndarray
    :param y_deflections: y component of the deflection angles
    :type y_deflections: numpy.ndarray

    :return: magnification
    :rtype: numpy.ndarray
    """
    image_grid = np.transpose(a=image_plane,axes=(1,2,0))
    y = image_grid[:,0,1]
    x = image_grid[0,:,0]
    traced_points_x, traced_points_y = ray_trace_grid(deflections_grid, x_deflections, y_deflections, image_plane, z1, z2)

    # compute jacobian
    dxdx = np.gradient(traced_points_x, x, axis=1)
    dxdy = np.gradient(traced_points_x, y, axis=0)
    dydx = np.gradient(traced_points_y, x, axis=1)
    dydy = np.gradient(traced_points_y, y, axis=0)

    return abs(1/(dxdx*dydy-dxdy*dydx))

def convergence_from_deflections(deflections_grid, x_deflections, y_deflections, image_plane, z1=None, z2=None):

    image_grid = np.transpose(a=image_plane,axes=(1,2,0))
    y = image_grid[:,0,1]
    x = image_grid[0,:,0]
    traced_points_x, traced_points_y = ray_trace_grid(deflections_grid, x_deflections, y_deflections, image_plane, z1, z2)

    # compute jacobian
    dxdx = np.gradient(traced_points_x, x, axis=1)
    dydy = np.gradient(traced_points_y, y, axis=0)

    return 1-0.5*(dxdx+dydy)