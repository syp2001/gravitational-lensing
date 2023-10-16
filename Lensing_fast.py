from os import path
#matplotlib inline
from pyprojroot import here
workspace_path = '~/GitHub/gravitational-lensing'
#cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt
import matplotlib.pylab as plt
#plt.rcParams.update({'font.size': 6})
#plt.rcParams.update(plt.rcParamsDefault)

import numpy as np
from scipy.interpolate import griddata
from astropy.cosmology import FlatLambdaCDM
from scipy import ndimage

from astropy.convolution import convolve, convolve_fft, AiryDisk2DKernel

# JWST NIRCAM 2 micrometer pixel size = 0.031 as
# https://www.stsci.edu/jwst/instrumentation/imaging-modes
# Airy function radius = 0.05 as
# https://www.stsci.edu/files/live/sites/www/files/home/jwst/documentation/technical-documents/_documents/JWST-STScI-001157.pdf

# --------------------------------------------------------------------------------------------

def create_grid(size,spacing,thickness,gradient=False):
    grid = np.zeros((size,size))
    num_lines = int(size/spacing)
    
    for i in range(num_lines):
        grid[:,spacing*i:(spacing*i+thickness)] = 10**(-7-4*i/num_lines) if gradient else 1
    for j in range(num_lines):
        grid[spacing*j:(spacing*j+thickness),:] = 10**(-7-4*j/num_lines) if gradient else 1
    return grid

def transform_image(image,angle,center,scale):
    pixel_shift = np.array([-center[0],center[1]])/scale
    transformed_image = image.native
    transformed_image = ndimage.rotate(transformed_image,angle,order=0,reshape=False)
    transformed_image = ndimage.shift(transformed_image,pixel_shift,order=0)    
    # return al.Array2D.no_mask(transformed_image,pixel_scales=scale)
    return transformed_image

def change_coordinates(source_plane,center,fov,scale):
    traced_grid = np.copy(source_plane)
    traced_grid[:,0] = (-traced_grid[:,0]+center[0]+fov/2)/scale
    traced_grid[:,1] = (traced_grid[:,1]+center[1]+fov/2)/scale
    return np.transpose(traced_grid)

def plot_image(image,center,fov,ax,title,**kwargs):
    #ax.contourf(X+0.02187,Y+7.52,magnif.native,levels=[70,100000],zorder=10)
    x_center = center[1]
    y_center = center[0]
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
    vmin=-3.5,
    vmax=-2,
    **kwargs)
    ax.set(
    xlabel="arcsec", ylabel="arcsec"
    )
    ax.set(xlabel='arcsec ({:.2f} " / pixel)'.format(detector_arcsec_per_pxl), ylabel="arcsec")
    ax.set_facecolor("black")
    #ax.set_facecolor("white")
    ax.set_title(title)
    return im

# --------------------------------------------------------------------------------------------
# SOURCE IMAGE

#image_center = (-30,20)
#reco_image_fov = 20  # arcsec on one side

zoom_factor=4.0

#image_center = (-30,25)
#reco_image_fov = 10  # arcsec on one side
#detector_arcsec_per_pxl = 0.0025
#dim="0025"

image_center = (-30,25)
reco_image_fov = 10  # arcsec on one side
detector_arcsec_per_pxl = 0.005
dim="005c"

#image_center = (-30,25)
#reco_image_fov = 10  # arcsec on one side
#detector_arcsec_per_pxl = 0.001
#dim="05b"

#detector_arcsec_per_pxl = 0.005
#dim="005"

#detector_arcsec_per_pxl = 0.05
#dim="05"

air_pix=0.05/detector_arcsec_per_pxl
print("Air_pix=",air_pix)

print("load mapping files")
# read in source and image plane from files computed by Ray Tracing.inpyb
traced_image_plane = al.Grid2D.no_mask(np.load("image_plane_"+dim+".npy"), pixel_scales=0.1)
traced_source_plane = al.Grid2D.no_mask(np.load("source_plane_"+dim+".npy"), pixel_scales=0.1)
print(traced_source_plane.shape)
print("load mapping files done")

"""
# find index of location of earendel on image plane
index = np.where((abs(traced_image_plane[:,0]+29.2)<0.0025)\
         & (abs(traced_image_plane[:,1]-25)<0.0025))
print('index=',index)
# trace back to source plane
source_center = traced_source_plane[index][0]
"""

shift_x=0.0
shift_y=0.0
source_center=(-7.521+shift_x, 0.022+shift_y)
print("Source Plane Center: ({:.3f}, {:.3f})".format(source_center[0],source_center[1]))


#image_path = path.join("new2.fits")
image_path = path.join("galaxy_image_lowsfe.fits")
# read in source image data
galaxy_image = al.Array2D.from_fits(
    file_path=image_path,
    pixel_scales=0.0001*zoom_factor,
)

source_pix = galaxy_image.shape_native[0]
source_fov = source_pix*galaxy_image.pixel_scale


image_pix = int(reco_image_fov / detector_arcsec_per_pxl)
print('Image size',image_pix)

print('Define source plane')
# define source plane
source_plane_grid_2d = al.Grid2D.uniform(
    shape_native=galaxy_image.shape_native,
    pixel_scales=galaxy_image.pixel_scale,
    origin=source_center
)

print('Transform coordinates')
# transform traced grid coordinates to use with map_coordinates
traced_grid = change_coordinates(traced_source_plane, \
                                 center = source_center, \
                                 fov = source_fov, \
                                 scale = galaxy_image.pixel_scale)

print('Transform coordinates done')
# --------------------------------------------------------------------------------------------
# READ CONVERGENCE and MAGIFICATION

print('Read magnification')
# read data
magnif_path = path.join("glafic-model",\
                        "hlsp_relics_model_model_whl0137-08_glafic_v1_z06p2-magnif.fits")
magnif = al.Array2D.from_fits(file_path=magnif_path,pixel_scales=0.1)
kappa_path = path.join("glafic-model","hlsp_relics_model_model_whl0137-08_glafic_v1_kappa.fits")
kappa = al.Array2D.from_fits(file_path=kappa_path,pixel_scales=0.1)

# compute grid values
N = 1200
x = np.linspace(-60, 60, N)
y = np.linspace(-60, 60, N)
X, Y = np.meshgrid(x, y)
Y = -Y # pyautolens and numpy use different conventions

#CAUSTIC and CRITICAL LINES

magnif_line_x = X.flatten()[magnif>200]
magnif_line_y = Y.flatten()[magnif>200]
magnif_line_points = np.column_stack((magnif_line_y,magnif_line_x))
magnif_grid = change_coordinates(magnif_line_points, \
                                 center = (-30,-20), \
                                 fov = reco_image_fov, \
                                 scale = detector_arcsec_per_pxl)

caustic_y = ndimage.map_coordinates(traced_source_plane.native[:,:,0], magnif_grid, order=1, cval=float('nan'))
caustic_x = ndimage.map_coordinates(traced_source_plane.native[:,:,1], magnif_grid, order=1, cval=float('nan'))

# --------------------------------------------------------------------------------------------
# PLOT IMAGES

#galaxy_center = (0,0.15) # relative to center of source plane
galaxy_center = (0.0,0.0) # relative to center of source plane
galaxy_angle = -15

print("create lensed image")
# rotate and shift image
transform = transform_image(galaxy_image, galaxy_angle, galaxy_center, galaxy_image.pixel_scale)

# lens image
#lensed_image = ndimage.map_coordinates(transform, traced_grid, order=1, cval=float('nan'))
lensed_image = ndimage.map_coordinates(transform, traced_grid, order=1, cval=1e-19)
lensed_image = np.reshape(lensed_image, (-1, int(reco_image_fov / detector_arcsec_per_pxl)))
print("create lensed image done")

print(lensed_image.shape,np.min(lensed_image), np.max(lensed_image))
print("start filtering")
#lensed_image=ndimage.gaussian_filter(lensed_image, sigma=air_pix)
#lensed_image = ndimage.rotate(lensed_image, -45.0,reshape=True,mode='wrap',cval=0.0)

# convolve image
lensed_image = convolve(lensed_image, AiryDisk2DKernel(air_pix,x_size=1*air_pix+1,y_size=1*air_pix+1),fill_value=0,nan_treatment='fill')

print("end filtering")
#print(lensed_image_r.shape,np.min(lensed_image_r), np.max(lensed_image_r))

# plot source image
fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=150)

#ax[1].contourf(X,Y,np.log(kappa.native),levels=40,cmap="bone",vmin=0,vmax=3,alpha=0.5)
#ax[1].contourf(X,Y,magnif.native,levels=[1000,100000],zorder=2, \
#                           cmap="Reds",alpha=0.2,vmax=80000)

# plot center of source plane
ax[0].scatter(source_center[1],source_center[0],color="yellow",s=1)
#ax[0].scatter(caustic_x,caustic_y,s=1.)

im1 = plot_image(transform,source_center,source_fov,ax[0],"Source")
im2 = plot_image(lensed_image,image_center,reco_image_fov,ax[1],"Image")
fig.colorbar(im2, ax=ax, label="log Flux [$\mu$Jy]")

ax[0].set_xlim([-0.2,0.2])
ax[0].set_ylim([-7.8,-7.4])
ax[1].set_xlim([22.5,27.5])
ax[1].set_ylim([-32.5,-27.5])


import matplotlib.animation as animation
def init():
    return [im1,im2]

def animate(i):
    #galaxy_center = (-0.2+0.6*i/300,-0.2+0.6*i/300)
    #galaxy_center = (-0.01+0.4*i/200,0.015-0.4*i/200)
    #galaxy_center = (0.5*i/200,0.15-0.15*i/200)
    galaxy_angle = i*4
    print(i)
    transform = transform_image(galaxy_image, galaxy_angle, galaxy_center, galaxy_image.pixel_scale)
    lensed_image = ndimage.map_coordinates(transform, traced_grid, order=1, cval=float('nan'))
    lensed_image = np.reshape(lensed_image, (-1, int(reco_image_fov / detector_arcsec_per_pxl))
    )
    # convolve image
    lensed_image = convolve(lensed_image, AiryDisk2DKernel(air_pix,x_size=1*air_pix+1,y_size=1*air_pix+1),fill_value=0,nan_treatment='fill')
    array1 = np.log10(transform, where=(transform != 0), out=np.full_like(transform, -15))+6
    array2 = np.log10(lensed_image, where=(lensed_image != 0), out=np.full_like(lensed_image, -15))+6
    im1.set_array(array1)
    im2.set_array(array2)
    return [im1,im2]

print('Start animation')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=90, interval=100, blit=True)
print('Start writing animation')
FFwriter = animation.FFMpegWriter(fps=10)
anim.save('lens_rot_caustic_lowsfe4.mp4', writer = FFwriter)

plt.show()
