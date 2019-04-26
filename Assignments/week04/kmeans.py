import numpy as np
import random

class KMeans:
    
    def __init__(self, k, max_iter=1000, min_variation_threshold=0.001):
        self.k = k
        self.max_iter = max_iter
        self.min_variation_threshold = min_variation_threshold
        self.variation_count = k


    def fit(self, X):
        '''Each sample should be a row in X matrix'''
        if not isinstance(X, np.ndarray):
            raise TypeError('Input X should be type of numpy.array.')
        if self.k > X.shape[0]:
            raise ValueError("Need more input data")
        
        self.data = X.reshape((X.shape[0], 1))\
             if len(X.shape) == 1 else X

        if self.k == X.shape[0]:
            self.clusters = [self.data[i] for i in range(self.k)]
        
        # init k centers in random
        center_inds = np.random.randint(0, self.data.shape[0], self.k)
        centers = self.data[center_inds].copy()
        
        # update centers
        for iter in range(self.max_iter):
            self.variation_count = self.k
            # overall distances between each sample and center
            cluster_inds = [[] for i in range(self.k)]
            for i in range(self.data.shape[0]):
                dists = np.linalg.norm(centers - self.data[i], axis=1)
                # put nearest sample to each cluster
                j = dists.argmin()
                cluster_inds[j].append(i)

            # new centers
            for i, inds in enumerate(cluster_inds):
                if len(inds) == 0:
                    # select a random sample as center in empty cluster
                    inds = random.randint(0, self.data.shape[0] - 1)
                new_center = self.data[inds].mean(axis=0)
                if np.linalg.norm(new_center - centers[i]) < self.min_variation_threshold:
                    self.variation_count -= 1 # no change
                else:
                    centers[i] = new_center
            # print(centers)
            # print(cluster_inds)
            if self.variation_count == 0: 
                print(iter)
                break
        
        self.clusters = [self.data[inds] for inds in cluster_inds]
        self.centers = centers

    def predict(self, sample):
        '''
        input sample should be 1-D array
        return: which cluster the sample belongs to
        '''
        return np.linalg.norm(self.centers - sample, axis=1).argmin()
            

        
if __name__ == '__main__':
    line = np.random.randn(10)
    p = np.hstack((np.arange(i, i+5,0.2, dtype=np.float) for i in [10, 15, 20, 30, 40]))
    X = np.outer(p, np.ones((10,))) + line
    
    km = KMeans(10)
    km.fit(X)
    print(km.clusters)
    print(km.predict(np.random.randn(10) + 17))
        