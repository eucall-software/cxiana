""" Credits: Matthias Geier """

from matplotlib import pyplot
from mpl_toolkits import axes_grid1

def addColorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1/aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = pyplot.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    pyplot.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def setupHistogram( histogram, bins, **kwargs ):
    """ Takes the return from numpy.histogram and returns a mpl axis object to plot. """
    #fig, ax = plt.subplots()
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    pyplot.bar(center, histogram, align='center', width=width)
