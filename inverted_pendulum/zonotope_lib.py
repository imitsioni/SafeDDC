import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from copy import copy
import gurobipy as gp



class Zonotope:
    ndim: int

    def __init__(self, generators: np.ndarray, center: np.ndarray):
        if len(generators.shape) == 1:
            generators = generators.reshape(generators.shape[0], 1)
        if len(center.shape) == 1:
            center = center.reshape(center.shape[0], 1)
        assert generators.shape[0] == center.shape[0], "zonotope generators should have same number of columns as " \
                                                       "the zonotope center."
        self.ndim = generators.shape[0]
        self.ngen = generators.shape[1]
        self.generators = generators
        self.center = center
        self.generators_pinv = np.linalg.pinv(self.generators)

    def update_pinv(self):
        self.generators_pinv = np.linalg.pinv(self.generators)

    def sample(self, nsamples=1) -> np.ndarray:
        rand_coefficients = np.random.rand(self.ngen, nsamples) * 2 - 1
        samples = np.matmul(self.generators, rand_coefficients) + self.center
        return samples

    def minkowski_sum(self, other_zonotope: 'Zonotope'):
        SELF = copy(self)
        SELF.ngen += other_zonotope.ngen
        SELF.generators = np.concatenate((self.generators, other_zonotope.generators), axis=1)
        SELF.center += other_zonotope.center
        SELF.update_pinv()
        return SELF

    def __rmul__(self, other):
        SELF = copy(self)
        if isinstance(other, np.ndarray):
            SELF.generators = np.matmul(other, SELF.generators)
            SELF.center = np.matmul(other, SELF.center)
            SELF.ndim = SELF.generators.shape[0]
        else:
            SELF.generators = other * SELF.generators
        SELF.update_pinv()
        return SELF

    def __add__(self, other):
        SELF = copy(self)
        if isinstance(other, Zonotope):
            return SELF.minkowski_sum(other)
        elif isinstance(other, np.ndarray):
            SELF.center = self.center + other
            return SELF
        elif isinstance(other, float):
            SELF.center = self.center + other
            return SELF

    def get_bounding_box_size(self)-> np.ndarray:
        return np.sum(np.abs(self.generators), axis=1).reshape(self.ndim, 1) * 2

    def get_range(self):
        range = np.concatenate(
            (self.center - self.get_bounding_box_size() / 2, self.center + self.get_bounding_box_size() / 2),
            axis=1)
        return range

    def get_bounding_box(self):
        size_vect = self.get_bounding_box_size()
        bounding_box = size_to_box(size_vect)
        bounding_box.center = self.center
        return bounding_box

    def get_inner_box(self):
        size_vect = self.get_bounding_box_size()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        alpha = m.addMVar(1, lb=0, ub=1)
        beta = m.addMVar((self.ndim, self.ngen), lb=-np.inf, ub=np.inf)
        beta_abs = m.addMVar((self.ndim, self.ngen), lb=0, ub=np.inf)
        for i in range(self.ndim):
            m.addConstr(beta[i, :] <= beta_abs[i, :])
            m.addConstr(-beta[i, :] <= beta_abs[i, :])

        for i in range(self.ngen):
            vect = np.ones((self.ndim,))
            m.addConstr(vect @ beta_abs[:, i] <= 2)

        for i in range(self.ndim):
            g = self.generators[i, :]
            for j in range(self.ndim):
                if i == j:
                    m.addConstr(g @ beta[j, :] >= size_vect[i] * alpha)
                else:
                    m.addConstr(g @ beta[j, :] == 0)

        m.setObjective(-1 * alpha)
        m.optimize()
        box = size_to_box(alpha.X * size_vect)
        box.center = self.center
        return box

    def pseudo_convert(self, points: np.ndarray) -> np.ndarray:
        """
        Convert a set of points to the basis of the zonotope (i.e. as a convex combination of the zonotope vectors
        from the zonotope center) using the pseudo-inverse matrix
        """
        return np.matmul(self.generators_pinv, points-self.center)



    __array_priority__ = 10000

    def get_corners(self):
        corners = np.zeros((self.ndim, 2 ** self.ngen))
        corners += self.center
        for i in range(2 ** self.ngen):
            i_bin = bin(i)
            i_bin = i_bin[2:]
            for j in range(self.ngen - len(i_bin)):  # add trailing zeros
                i_bin = '0' + i_bin
            for j in range(self.ngen):
                if i_bin[j] == '0':
                    mul = -1
                else:
                    mul = 1
                corners[:, i] += mul * self.generators[:, j]
        return corners


class Box(Zonotope):
    def __init__(self, interval_ranges: np.ndarray):
        interval_lengths = np.diff(interval_ranges, axis=1).reshape(interval_ranges.shape[0], )
        generators = np.diag(interval_lengths / 2)
        center = np.mean(interval_ranges, axis=1)
        Zonotope.__init__(self, generators=generators, center=center)

    def contains(self, z: [Zonotope, np.ndarray]):
        s = self.get_bounding_box()
        ndim = s.ndim
        if isinstance(z, Zonotope):
            b = z.get_bounding_box()
            b_lb = b.center - np.sum(b.generators, axis=1).reshape(ndim, 1)
            b_ub = b.center + np.sum(b.generators, axis=1).reshape(ndim, 1)
            s_lb = s.center - np.sum(s.generators, axis=1).reshape(ndim, 1)
            s_ub = s.center + np.sum(s.generators, axis=1).reshape(ndim, 1)
            if np.all(s_lb <= b_lb + 1e-12) and np.all(b_ub <= s_ub + 1e-12):
                return True
            else:
                return False
        elif isinstance(z, np.ndarray):
            s_lb = s.center - np.sum(s.generators, axis=1).reshape(ndim, 1)
            s_ub = s.center + np.sum(s.generators, axis=1).reshape(ndim, 1)
            if np.all(s_lb <= z + 1e-12) and np.all(z <= s_ub + 1e-12):
                return True
            else:
                return False

    def intersects(self, b: "Box"):
        ndim = b.ndim
        b_lb = b.center - np.sum(b.generators, axis=1).reshape(ndim, 1)
        b_ub = b.center + np.sum(b.generators, axis=1).reshape(ndim, 1)
        lb = self.center - np.sum(self.generators, axis=1).reshape(ndim, 1)
        ub = self.center + np.sum(self.generators, axis=1).reshape(ndim, 1)
        if np.all(lb <= b_ub) and np.all(b_lb <= ub):
            return True
        else:
            return False


def size_to_box(size_vector: np.ndarray) -> Box:
    ndim = size_vector.shape[0]
    size_vector = size_vector.reshape((ndim, 1))
    box = Box(np.concatenate((-size_vector / 2, size_vector / 2), axis=1))
    return box


def plot_zonotope(zonotope: Zonotope, color='r--', fill=True, linewidth=1):
    corneres = zonotope.get_corners().T
    hull = ConvexHull(corneres)
    if fill:
        fig = plt.fill(corneres[hull.vertices, 0], corneres[hull.vertices, 1], color, linewidth=linewidth)
    else:
        vertices = np.concatenate((hull.vertices, hull.vertices[0:1]))
        fig = plt.plot(corneres[vertices, 0], corneres[vertices, 1], color, linewidth=linewidth)
    return fig


class StateImage():
    def __init__(self, range: np.ndarray, resolution: np.ndarray):
        self.range = copy(range)
        self.resolution = copy(resolution)
        map_size = tuple((np.diff(range) / self.resolution + 1).reshape(2,).astype(int).tolist())
        self.map = np.zeros(map_size)
        self.map_size = np.array(map_size).reshape(2, 1)

    def state2image(self, range):
        relative_range = range - self.range[:, 0:1]
        return relative_range / self.resolution

    def get_patch(self, state_range):
        image_range = self.state2image(state_range)
        image_range[:, 0] = np.floor(image_range[:, 0])
        image_range[:, 1] = np.ceil(image_range[:, 1])
        image_range = np.maximum(image_range, np.zeros((2, 1)))
        image_range = np.minimum(image_range, self.map_size - 1)
        image_range = image_range.astype(int)
        if np.any(image_range[:, 0] == image_range[:, 1]):
            return None
        row_id = list(range(image_range[0, 0], image_range[0, 1] + 1))
        col_id = list(range(image_range[1, 0], image_range[1, 1] + 1))

        return self.map[np.ix_(row_id, col_id)]

    def set_patch(self, state_range, value):
        image_range = self.state2image(state_range)
        image_range[:, 0] = np.ceil(image_range[:, 0])
        image_range[:, 1] = np.floor(image_range[:, 1])
        image_range = np.maximum(image_range, np.zeros((2, 1)))
        image_range = np.minimum(image_range, self.map_size - 1)
        image_range = image_range.astype(int)
        if np.any(image_range[:, 0] == image_range[:, 1]):
            return None
        row_id = list(range(image_range[0, 0], image_range[0, 1] + 1))
        col_id = list(range(image_range[1, 0], image_range[1, 1] + 1))

        self.map[np.ix_(row_id, col_id)] = value

class WinSetCheck():
    def __init__(self, state_range, resolution):
        self.winset = StateImage(state_range, resolution)
        self.map_range = state_range

    def is_included(self, box: Box):
        state_range = box.get_range()
        if np.any(state_range[:,0]<self.map_range[:,0]) or np.any(state_range[:,1]>self.map_range[:,1]):
            return False
        state_patch = self.winset.get_patch(state_range)
        if state_patch is not None and np.all(state_patch > 0):
            return True
        return False

    def intersects(self, box: Box):
        state_range = box.get_range()
        state_patch = self.winset.get_patch(state_range)
        if state_patch is not None and np.any(state_patch > 0):
            return True
        return False
