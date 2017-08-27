import numpy as np
import scipy.misc
from scipy.ndimage import filters

class cRegion():
    """ Generate an 8-connected region """
    
    def __init__(self,density=.15,g_size=[15,15]):        
        # window limits
        self.g_size = g_size
        self.half_win = [g_size[0]//2,g_size[1]//2]
        self.lims = [(-self.half_win[0],self.half_win[0]),
                     (-self.half_win[1],self.half_win[1])]

        # store the total number of blob points
        self.blob_pts = np.round(density*np.prod(self.half_win)*4).astype(int)
        self.generate()

    def generate(self):        
        # generate sets to hold blob and edge points
        self.edges = set()
        self.blobs = {(0,0)} # initialize at window center
        [self.edges.add(ni) for ni in self.neighbours((0,0))]
        
        while len(self.blobs) < self.blob_pts:
            # choose a random edge to convert to a blob
            r_i = np.random.randint(len(self.edges))
            e = list(self.edges)[r_i]            
            self.edges.remove(e) # remove from edges
            # add its unique neighbors
            [self.edges.add(ni) for ni in self.neighbours(e)
             if not ni in self.blobs] 
            self.blobs.add(e) # append to blobs
        
    def neighbours(self,pt):
        # compute the 8-neighbors within window bounds
        x,y = pt[0],pt[1]
        n = [(x-1,y),(x+1,y),(x,y-1),(x,y+1),(x-1,y-1),(x-1,y+1),
             (x+1,y+1),(x+1,y-1)]
        return [c for c in n if c[0] >= -self.half_win[0]
                and c[0] < self.half_win[0] and c[1] >= -self.half_win[1]
                and c[1] < self.half_win[1]]

    def image(self,size=None,interp='nearest',blur=None):
        # return a mask
        m = np.zeros([self.g_size[0],self.g_size[1]])
        for bi in self.blobs: # returns a mask if not resized
            m[bi[0]+self.half_win[0],bi[1]+self.half_win[1]] = 1
        if size is not None:
            m = scipy.misc.imresize(m,size,interp=interp)
        if blur is not None:
            m = filters.gaussian_filter(m, blur, mode='nearest')
        return m
