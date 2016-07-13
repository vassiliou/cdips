from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import skimage
from skimage import io
from skimage import feature,measure
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import glob
import pandas as pd
import numpy as np

from builtins import (
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip)


#trainfolder = "/Users/gus/CDIPS/nerve-project/train/"
datafolder = "/Users/gus/CDIPS/nerve-project/"

if os.environ['USER'] == 'chrisv':
    datafolder = '../'

trainfolder = os.path.join(datafolder, 'train')
testfolder = os.path.join(datafolder, 'test')

trainimages = glob.glob(os.path.join(trainfolder, '*.tif'))

training = pd.read_csv(os.path.join(datafolder,'train_masks.csv'))
all_training = training[['subject','img']]
non_empty = training[ ~pd.isnull(training.pixels)][['subject','img']]
empty = training[ pd.isnull(training.pixels)][['subject','img']]

def load_image(idx, training):
    nameformat = '{subject}_{img}.tif'
    maskformat = '{subject}_{img}_mask.tif'
    #skimage.io.imread(os.path.join(imagefolder,''))
    imagefile = nameformat.format(subject=training['subject'][idx],
                              img=training['img'][idx])
    maskfile = maskformat.format(subject=training['subject'][idx],
                              img=training['img'][idx])
    image = io.imread(os.path.join(trainfolder, imagefile))
    mask = io.imread(os.path.join(trainfolder, maskfile))   
    return (image, mask, imagefile)


class image_pair(object):
    
    def __init__(self,subject,img_number):
        nameformat = '{subject}_{img}.tif'
        maskformat = '{subject}_{img}_mask.tif'
        imagefile = nameformat.format(subject=subject,
                              img=img_number)
        maskfile = maskformat.format(subject=subject,
                              img=img_number)
        self.img_number = img_number
        self.subject = subject
        self.image = io.imread(os.path.join(trainfolder, imagefile))
        self.mask = io.imread(os.path.join(trainfolder, maskfile))   
        self.dims = self.image.shape
        try: 
            contours = measure.find_contours(self.mask, 254.5)
            if len(contours)>1:
                ic = np.argmax([contour.shape[0] for contour in contours])
                contour = contours[ic]
            else:
                contour = contours[0]
            self.contour = contour
            self.contours = contours
        except:
            self.contour = np.empty((1,2))
            self.contours = [] 
        
    def plot(self, ax=None, figure_size=(6,4)):
        title = '{subject}_{img}.tif'
        title=title.format(subject=self.subject,img=self.img_number)
        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)
        ax.imshow(self.image, cmap=plt.cm.gray)
        maskcontour = [self.contour]
        if not maskcontour==[]:
            ax.plot(maskcontour[0][:,1], maskcontour[0][:,0], linewidth=2)
            ax.set_title(title)
        else:
                ax.set_title(title + ' (no region)', fontsize=18)
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        return ax
    
    def plotmask(self, ax=None, figure_size=(6,4)):
        title = '{subject}_{img}.tif'.format(subject=self.subject,
                                             img=self.img_number)
        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)
        ax.imshow(self.mask, cmap=plt.cm.gray)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        return ax
        
    def sample_contour_negative(self,P,F=17) :
        ## this might return < P samples..
        contour= self.contour.astype(int)
        l = contour.shape[0]
        contour_pixels = set([ tuple(contour[i]) for i in range(l)])
        x_coords=np.random.randint(F,self.dims[0]-F,size=P)
        y_coords=np.random.randint(F,self.dims[1]-F,size=P)
        coords=zip(x_coords,y_coords)
        #print type(coords[0][0])
        #print type(tuple(contour[0])[0])
        negative = [c for c in coords if c not in contour_pixels ]
        return np.array(negative)
        


def build_data(num_images,P,F):
    data ={}
    sample_images=non_empty.sample(num_images)
    ind = np.array(sample_images)
    for i in range(num_images):
        im_pair = image_pair(ind[i,0],ind[i,1])
        sample_from_image(P,F,im_pair,data)
    all_sample_images = all_training.sample(num_images)
    ind = np.array(all_sample_images)
    for i in range(num_images):
        im_pair = image_pair(ind[i,0],ind[i,1])
        sample_negatives_from_image(P,F,im_pair,data)    
    return data

def get_patch(image,pixel,F):
        hor_range = (pixel[0]-F,pixel[0]+F+1)
        ver_range= (pixel[1]-F,pixel[1]+F+1)
        return image[hor_range[0]:hor_range[1],ver_range[0]:ver_range[1]]

def write_patch(patch,pixel,data,subject,img,sample_number,negative=False):
    data[(subject,img,tuple(pixel))] = patch

def sample_from_image(P,F,image_pair,data):
    mask = image_pair.mask
    width,height = image_pair.dims
    contour=np.rint(image_pair.contour)
    total_pixels = contour.shape[0]
    sample_pixels = np.random.randint(0,total_pixels, size=P)
    for sample_num,p in enumerate(sample_pixels):
        patch = get_patch(mask,contour[p],F)
        write_patch(patch,contour[p],data,image_pair.subject,image_pair.img_number,sample_num)

def sample_negatives_from_image(P,F,image_pair,data):
    mask = image_pair.mask
    width,height = image_pair.dims
    sample_pixels=image_pair.sample_contour_negative(P,F)
    for sample_num, pixel in enumerate(sample_pixels):
        patch = get_patch(mask,pixel,F)
        write_patch(patch,pixel,data,image_pair.subject,image_pair.img_number,sample_num,negative=True)
