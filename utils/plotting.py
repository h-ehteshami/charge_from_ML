import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def iso_plot(input_array, mn = 0.0, mx = 0.3, title = '', lx = 1, ly = 1, lz = 1):
    """
    Input must be a 3D array: X[n1,n2,n3]. The min and max will be showen to help decide isocounters.
    """
    dnx, dny, dnz = input_array.shape
    array_plt   = input_array.flatten()
    print(f"Min : {np.min(array_plt):5.7f}, and Max : {np.max(array_plt):5.7f}")
    X, Y, Z = np.mgrid[0:lx:dnx*1j, 0:ly:dny*1j, 0:lz:dnz*1j]
    fig = go.Figure(

        data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=array_plt,
        opacity=0.6,
        isomin=mn,
        isomax=mx,
        surface_count=1,
        caps=dict(x_show=False, y_show=False)),

        layout=go.Layout(
        title=dict( text = title ),
        font=dict( family="Courier New, monospace", 
                    size=13, color="RebeccaPurple", 
                    variant="small-caps", ) )
        )
    fig.show()

def intg_plot(input_array, axis = 2, title = ''):
    integrated = np.trapezoid(input_array, axis = axis )
    vmin = integrated.min()
    vmax = integrated.max()
    plt.imshow(integrated, cmap='plasma',
               interpolation ='spline36', 
               vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show()
