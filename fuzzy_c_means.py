import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import cv2


class C_Means(object):
    def __init__(self):
        pass

    def c_means(self, image):
        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

        img = cv2.imread(image)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        r = r.reshape(-1)
        g = g.reshape(-1)
        b = b.reshape(-1)

        # # Set up the loop and plot
        fig1, ax = plt.subplots()
        alldata = np.vstack((r, g, b))
        ncenters = 4

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        tmp = img.copy()
        tmp = tmp.reshape(-1, 3)
        for i in range(ncenters):
            new_rgb = self.get_colors_membership(cluster_membership, r, g, b, i)
            tmp[cluster_membership == i] = new_rgb

        tmp = tmp.reshape(img.shape)
        cv2.imwrite("A.jpg", tmp)
        return tmp

    def get_colors_membership(self, cluster_membership, r, g, b, cluster):
        tmp_r = sum(r[cluster_membership == cluster])
        tmp_g = sum(g[cluster_membership == cluster])
        tmp_b = sum(b[cluster_membership == cluster])

        size = len(cluster_membership[cluster_membership == cluster])

        tmp_r /= size
        tmp_g /= size
        tmp_b /= size

        tmp_r = int(tmp_r)
        tmp_g = int(tmp_g)
        tmp_b = int(tmp_b)

        return [tmp_r, tmp_g, tmp_b]
