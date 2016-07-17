import skimage
from skimage import io
from skimage import feature,measure
from scipy.interpolate import InterpolatedUnivariateSpline

import matplotlib.pyplot as plt
from matplotlib import colors
import os
#import glob
import pandas as pd
import numpy as np

datafolder = "/Users/gus/CDIPS/nerve-project/"

if os.environ['USER'] == 'chrisv':
    datafolder = '../'

trainfolder = os.path.join(datafolder, 'train')
testfolder = os.path.join(datafolder, 'test')

class image():
    def __init__(self, row, figsize=(6,4)):
        if row is not None:
            self.info = row
        self._image = None  # io.imread(os.path.join(trainfolder, imagefile))
        self.title = '{subject}_{img}'.format(subject=row['subject'],
                                              img=row['img'])
        self.filename = self.title + '.tif'
        self.figsize = figsize
    
    def __str__(self):
        return self.info.__str__()
    
    def __repr__(self):
        return self.info.__repr__()
    
    def load(self):
        """ Load image file """
        return io.imread(os.path.join(trainfolder, self.filename))

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(self.image, cmap=plt.cm.gray)
        ax.set_title(self.title)
        ax.axis('equal')
        ax.tick_params(which='both', axis='both', 
                          bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.autoscale(tight=True)

        return ax
    
    @property
    def image(self):
        if self._image is None:
            self._image = self.load()
        return self._image
    
    def get_patch(image,pixel,F):
            hor_range = (pixel[0]-F,pixel[0]+F+1)
            ver_range= (pixel[1]-F,pixel[1]+F+1)
            return image[hor_range[0]:hor_range[1],ver_range[0]:ver_range[1]]


class mask(image):
    def __init__(self, info):
        if type(info) is np.ndarray:
            image.__init__(self, row=None)
            image._image = info

        self._contour = None
        self.contourlength = 40

    @property
    def contour(self):
        if self._contour is not None: 
            contours = measure.find_contours(self._image, 254.5)
            # downsample contour
            contour = np.vstack(contours)
            T_orig = np.linspace(0, 1, contour.shape[0])
            ius0 = InterpolatedUnivariateSpline(T_orig, contour[:,0])
            ius1 = InterpolatedUnivariateSpline(T_orig, contour[:,1])
            T_new = np.linspace(0, 1, self.contourlength)
            self._contour = np.vstack((ius0(T_new), ius1(T_new)))
        return self._contour
    
    @contour.setter
    def contour(self, contour):
        self._contour = contour
    
    @property
    def RLE(self):
        pass
    
    def plot_contour(self, *args, ax=None, **kwargs):
        x = self.contour[:,1]
        y = self.contour[:,0]
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x,y, *args, **kwargs)
        ax.axis('equal')
        ax.tick_params(which='both', axis='both', 
                          bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.autoscale(tight=True)

        return ax
        
    
class image_pair(object):
    def __init__(self, row, figsize=(6,4)):
        self.image = image(row, figsize)
        self.mask = mask(row, figsize)      
        self.pred = mask(None, figsize)
        self._score = None

    @property        
    def score(self):
        if self._score is None:
            X = self.mask.image 
            Y = self.pred.image                               
        return 2 * np.count_nonzero(X == Y) / (np.prod(X.shape) + np.prod(Y.shape))


def plot_pca_comps(P, ncomp, *args, **kwargs):
    fig = plt.figure()
    for i in np.arange(ncomp):        
        for j in np.arange(i, ncomp):
            ax = fig.subplot(ncomp-1, ncomp-1, j+i*(ncomp-1))
            ax.scatter(P[:,i], P[:,j], *args, **kwargs)
            
class batch(object):
    def __init__(self, rows):
        self.batch = [image_pair(row) for row in rows]
    
    def array(self):
        """ Load a series of images and return as a 3-D numpy array.
        imageset consists of rows from training.bin"""
        return np.array([im.image for im in self.batch])

        
    def plot_grid(self, ncols=5, plotimage=True, plotcontour=True, plotpred=False, figwidth=16):
        """Plot a grid of images, optionally overlaying contours and predicted contours
            Assumes the input is a Pandas DataFrame as in training.bin    
        """    
        nrows = nrows=int(np.ceil(len(self.batch)/ncols))
        figheight = figwidth/ncols*nrows
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                               figsize=(figwidth,figheight))
        ax = ax.flatten()
        for idx, imgpair in enumerate(self.batch):
            
            if plotimage:
                imgpair.image.plot(ax=ax[idx])
            if plotcontour:
                imgpair.mask.plot_contour('-b', ax=ax[idx])
            if plotpred:
                imgpair.pred.plot_contour('-r', ax=ax[idx])

        return fig
    
         
    def plot_hist(self, ax=None):
        """Plot histograms of a set of images
            Assumes the input is a Pandas DataFrame as in training.bin    
        """    
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,8))
        for imgpair in self.batch:
            ax.hist(imgpair.image.flatten(), cumulative=True, normed=True,
                    bins=100, histtype='step', label=imgpair.image.filename, alpha=0.1, color='k')
        return ax
    
    def scores(self):
        scores = [imgpair.score for imgpair in self.batch]
        return scores

