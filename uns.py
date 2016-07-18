from skimage import io
from skimage import measure
from skimage import morphology

from scipy.interpolate import InterpolatedUnivariateSpline

import matplotlib.pyplot as plt

import os
#import glob
import pandas as pd
import numpy as np

datafolder = "/Users/gus/CDIPS/nerve-project/"

if os.environ['USER'] == 'chrisv':
    datafolder = '../'

trainfolder = os.path.join(datafolder, 'train')
testfolder = os.path.join(datafolder, 'test')

training = pd.read_msgpack('training.bin')

class image():
    def __init__(self, row):
        if type(row) is np.ndarray:
            # given an array, assume this is the image 
            self._image = row
            self.title = ''
            self.filename = ''
        else:
            self.info = row
            self._image = None  # io.imread(os.path.join(trainfolder, imagefile))
            self.title = '{subject}_{img}'.format(subject=row['subject'],
                                                  img=row['img'])
            self.filename = self.title + '.tif'
    
    def __str__(self):
        return self.info.__str__()
    
    def __repr__(self):
        return self.info.__repr__()
    
    
    def load(self):
        """ Load image file """
        return io.imread(os.path.join(trainfolder, self.filename))

    def plot(self, ax=None, **plotargs):
        if ax is None:
            fig, ax = plt.subplots()
        plotargs['cmap'] = plotargs.get('cmap',plt.cm.gray)
        ax.imshow(self.image, **plotargs)
        ax.set_title(self.title)
        ax.axis('equal')
        ax.axis('off')     
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
        image.__init__(self, info)

        self._contour = None
        self.contourlength = 40
        self.filename = self.title + '_mask.tif'
        self._properties = None
        self.hasmask = False
        
    @property
    def contour(self):
        if self._contour is None: 
            contours = measure.find_contours(self.image, 254.5)
            # downsample contour
            if len(contours)>0:
                contour = contours[np.argmax([c.shape[0] for c in contours])]
                T_orig = np.linspace(0, 1, contour.shape[0])
                ius0 = InterpolatedUnivariateSpline(T_orig, contour[:,0])
                ius1 = InterpolatedUnivariateSpline(T_orig, contour[:,1])
                T_new = np.linspace(0, 1, self.contourlength)
                self._contour = np.vstack((ius0(T_new), ius1(T_new)))
                self.hasmask = True
            else:
                self.hasmask = False
        return self._contour
    
    @contour.setter
    def contour(self, contour):
        self._contour = contour
        

    @property
    def properties(self):
        if self._properties is None:
            C = self.contour
            if self.hasmask:
                x, y = C
                m = measure.moments(self.image)
                c = np.mean(self.contour, axis=1)
                D = {}
                D['npixels'] = np.count_nonzero(self.image) 
                # Contour-derived values
                D['xmin'] = np.min(x)
                D['xmax'] = np.max(x)
                D['ymin'] = np.min(y)
                D['ymax'] = np.max(y)
    
                D['W'] = D['xmax'] - D['xmin']     
                D['H'] = D['ymax'] - D['ymin']
                
                # Image moments
                D['moments'] = m
                D['Cr'] = m[0, 1]/m[0, 0] 
                D['Cc'] = m[1, 0]/m[0, 0]
    
                # Contour SVD
                _, s, v = np.linalg.svd(self.contour.T - c)
                D['S0'] = s[0]
                D['S1'] = s[1]
                D['V0'] = v[:,0]        
                D['V1'] = v[:,1]
                
                # Width by medial axis
                skel, distance = morphology.medial_axis(self.image,
                                                        self.image,
                                                        return_distance=True)
                self.skel = skel
                self.distance = distance
                D['lenskel'] = np.sum(skel)
                distances = distance[self.image>0] # pick distance within mask
                D['meandist'] = np.mean(distances)
                D['maxdist'] = np.max(distances)
                D['medidist'] = np.median(distances)
                self._properties = D
        return self._properties

    @property
    def pandas(self):                
        df = self.info[['subject','img','pixels']]
        props = self.properties        
        if props is not None:
            for k, v in props.items():
                df.loc[k] = v
        return df
        
    @property
    def RLE(self):
        """Convert mask to run length encoded format"""
        dm = np.diff(self.image.T.flatten().astype('int16'))
        start = np.nonzero(dm>0)[0]
        stop = np.nonzero(dm<0)[0]
        RLE = np.vstack((start+2, stop-start))
        return ' '.join(str(n) for n in RLE.flatten(order='F'))
        
    
    def plot_contour(self, *args, **kwargs):
        C = self.contour      
        if self.hasmask:
            x = C[1,:]
            y = C[0,:]
            ax = kwargs.pop('ax', None)
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(x,y, *args, **kwargs)
            ax.axis('equal')
            ax.tick_params(which='both', axis='both', 
                              bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.autoscale(tight=True)
            return ax
        else:
            return None
        
    
class image_pair(object):
    def __init__(self, row):
        self.image = image(row)
        self.mask = mask(row)
        self._score = None
        
    def __add__(self, imgpair):
        return self.image + imgpair.image
    
    def __sub__(self, imgpair):
        return self.image - imgpair.image
        
    def plot(self):
        ax = self.image.plot()
        self.mask.plot_contour(ax=ax)
        return ax
        
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
            
class batch(list):
    def __init__(self, rows):
        list.__init__(self, [])
        for row in rows.iterrows():
            self.append(image_pair(row[1]))
             
    @property 
    def array(self):
        """ Load a series of images and return as a 3-D numpy array.
        imageset consists of rows from training.bin"""
        return np.array([im.image.image for im in self])

   
    def plot_grid(self, ncols=5, plotimage=True, plotcontour=True, plotpred=False, figwidth=16):
        """Plot a grid of images, optionally overlaying contours and predicted contours
            Assumes the input is a Pandas DataFrame as in training.bin    
        """    
        nrows=int(np.ceil(len(self)/ncols))
        figheight = figwidth/ncols*nrows
        fig = plt.figure(figsize=(figwidth,figheight))
        for idx, imgpair in enumerate(self,1):
            ax = fig.add_subplot(nrows, ncols, idx)
            if plotimage:
                imgpair.image.plot(ax=ax)
            if plotcontour:
                imgpair.mask.plot_contour('-b', ax=ax)
            if plotpred:
                imgpair.pred.plot_contour('-r', ax=ax)

        return ax
    
         
    def plot_hist(self, ax=None):
        """Plot histograms of a set of images
            Assumes the input is a Pandas DataFrame as in training.bin    
        """    
        if ax is None:
            fig, ax = plt.subplots()
        for imgpair in self:
            ax.hist(imgpair.image.image.flatten(), cumulative=True, normed=True,
                    bins=100, histtype='step', label=imgpair.image.filename, alpha=0.1, color='k')
        return ax
    
    def scores(self):
        scores = [imgpair.score for imgpair in self]
        return scores

if __name__ == '__main__':
        
    # Load image pair from training table
    img = image_pair(training.iloc[0])

    #plot individual image/maks
    fig, ax = plt.subplots(1,2)
    img.image.plot(ax=ax[0])
    img.mask.plot(ax=ax[1])
    
    #plot image pair to overlay contour
    img.plot()
    
    #create/plot batch of images
    imgbatch = batch(training.iloc[0:6])
    imgbatch.plot_grid()
    
    #use batch.array to get a NumPy array of all images in a batch
    #Call image with a 2-D NumPy array to get access to plotting etc.
    imgsum = image(np.sum(imgbatch.array,axis=0))
    imgsum.plot(cmap=plt.cm.viridis)    
    
    # histograms of batch of images?
    imgbatch.plot_hist()
    plt.show()
    
    print(img.mask.properties)
    print(img.mask.pandas)
    # Use batch.pop() to process images sequentially
    
#    import psutil
#    process = psutil.Process(os.getpid())
#    # Memory usage
#    newbatch=[]
#    mem = process.memory_info().rss
#    newbatch = batch(training.iloc[10:16])
#    print('Batch of 6: {:d}'.format(process.memory_info().rss))
#    newbatch.array
#    print('Batch of 6 with images: {:d}'.format(process.memory_info().rss))
#    mem = process.memory_info().rss
#    newbatch = batch(training.iloc[0:50])
#    A = newbatch.pop().image.image
#    for i in range(len(newbatch)):
#        A = A + newbatch.pop().image.image
#        if i%10 == 0:
#            print('{:d}, images: {:d}'.format(i,process.memory_info().rss))
#    savebatch = batch(training.iloc[0:50])
#    for i, img in enumerate(savebatch):
#        data = img.image.image        
#        if i%10 == 0:
#            print('{:d}, images: {:d}'.format(i,process.memory_info().rss))
#    
#    print(len(newbatch), len(savebatch))