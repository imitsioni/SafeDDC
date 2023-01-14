import shapely.geometry as sh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

class SafeSetApproximation:
    """
    Class that represents to polygon approximation of the safe sets
    """

    def __init__(self, segments, multi_dist=False):
        """
        Initializes the safe sets with given segments of polygons
        """

        # compute approximations
        self.polygons = list()
        for s in segments:
            p = sh.Polygon(np.array(s))
            self.polygons.append(p)

        # return distances to all safe sets?
        self._multi_dist = multi_dist

        # compute acceleration structure
        self.acc = self.compute_safe_set_acceleration_structure(self.polygons)

    @property
    def polygons(self):
        return self._polygons

    @polygons.setter
    def polygons(self, polygons):
        self._polygons = polygons

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, acc):
        self._acc = acc

    def plot(self, data_points= None, plot_acc = False, color = 'k', alpha = 0.1):
        """
        Plots the safe sets polygons or acceleration structure
        """
        ax = plt.gca()

        if data_points is not None:
            ax.plot(data_points[:,0],data_points[:,1],'-.k')

        if not plot_acc:
            for p in self.polygons:
                ax.add_patch(mp.Polygon(np.array(p.exterior.coords), color=color, alpha=alpha))
        else:
            for p in self.acc:
                ax.add_patch(mp.Polygon(np.array(p.exterior.coords), color=color, alpha=alpha))

    def compute_safe_set_acceleration_structure(self, sets: list) -> list:
        """
        Computes the safe set acceleration structure
        """
        acceleration_polygons = list()
        for p in sets:
            acceleration_polygons.append(p.envelope)
        return acceleration_polygons

    def query_safety(self, sample) -> (bool, float):
        """
        Performs the safety check for a given sample.

        Returns: tuple (bool safe yes or no, distance to safe set where >= 0 is in the set)
        """

        point = sh.Point(np.array(sample))

        # identify candidate sets through acceleration structure
        candidates = self.polygons

        dist = -np.inf
        d = 0
        self._multi_dist = False
        if not self._multi_dist:
            d = np.max([signed_distance(point, c) for c in candidates])
            return d>=0., d
            # for c in candidates:
            #     d = signed_distance(point, c)
            #     if d > dist:
            #         dist = d
            # return dist >= 0., dist
        else:
            d = list()
            for c in candidates:
                d.append(signed_distance(point, c))
            return np.greater_equal(d,0.).any(), d


def signed_distance(p: sh.Point, poly: sh.Polygon) -> float:
    """
    Computes the signed distance between a point and a polygon. However, here the sign is flipped! Negative values are
    outside and >= 0 is within the set
    """
    enclosed = poly.contains(p)
    return poly.exterior.distance(p) if enclosed else -poly.distance(p)


