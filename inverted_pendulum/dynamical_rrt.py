import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import compress
from zonotope_lib import Zonotope, Box, plot_zonotope
from pwa_lib import compute_multistep_affine_dynamics
from dynamics_library import SampleAndHold
from rtree import index

show_animation = True


def get_distance(state1: np.ndarray, state2: np.ndarray) -> float:
    return np.linalg.norm(state1 - state2)


class TreeNode:
    """
        tree node
        """

    def __init__(self, state: np.ndarray, ctrl_input: np.ndarray = None, parent: int = -1, path: list = [np.array([])]):
        self.state = state  # numpy array
        self.input = ctrl_input
        self.path = path  # numpy matrix (sequence of states)
        self.parent = parent  # the parent node
        self.id = None  # the parent node
        # self.cost = 0  # cost to get from/to the root
        self.path_length = 0  # cost to get to the parent


class DynamicalRRT:
    """
    RRT algorithm that returns a forward (backward) tree from (to) a starting (target) state under a defined continuous time dynamics.
    """

    def __init__(self, dynamics, state_space: Zonotope, input_space: Zonotope, root_state: np.ndarray,
                 max_iter=2000, backward=False, target_region=Box, nsteps=5, dynamics_type='continuous'):
        """
        Setting Parameters
        """
        self.continuous_dynamics = dynamics
        self.state_space = state_space
        self.input_space = input_space
        self.root_state = root_state
        self.max_iter = max_iter
        self.backward = backward
        self.root = TreeNode(root_state)
        self.node_list = [self.root]
        self.nnodes = 1
        self.ndim = state_space.ndim
        # Rtree definition for finding nearest nodes quickly
        rtree_properties = index.Property()
        rtree_properties.dimension = self.ndim
        self.rtree = index.Rtree(interleaved=True, properties=rtree_properties)
        root_tuple = tuple(root_state[:,-1].reshape((self.ndim,)))
        self.rtree.insert(0, root_tuple + root_tuple, root_tuple)
        self.tree_dict = {root_tuple: 0}  # To find the node index from the node state
        if dynamics_type == 'continuous':
            self.discretization_step_s = 0.001
            self.sample_time_s = 0.01
            self.discrete_dynamics = SampleAndHold(dynamics, self.sample_time_s, self.discretization_step_s,
                                                   backward=backward)
        elif dynamics_type == 'discrete':
            self.discrete_dynamics = dynamics

        # Defining the termination criteria (i.e. reaching the starting/target region)
        self.target_region = target_region
        self.success = target_region.contains(
            root_state)  # It will be trivially true if the root is in the target region
        self.counter = 0
        self.nsteps = nsteps
        input_range = self.input_space.get_range()
        self.input_space_multistep = Box(np.tile(input_range, (nsteps, 1)))
        self.node_mat = root_state

    def get_reachable_set(self, state: np.ndarray):
        state = Box(np.concatenate((state, state), 1))
        affine_dynamics = compute_multistep_affine_dynamics(self.discrete_dynamics, self.nsteps, state_box=state, input_box=self.input_space)
        reach_set = affine_dynamics.get_ru(self.input_space_multistep) + affine_dynamics.get_rx(state).center
        return reach_set

    def planning(self, animation=False, max_iter=0):
        if max_iter == 0:
            max_iter = self.max_iter
        select_from_target_rate = 0.0
        for i in range(max_iter):
            if self.success:
                break
            if random.random() > select_from_target_rate:
                rnd_state = self.state_space.sample()
            else:
                rnd_state = self.target_region.sample()

            nearest_node_indx = self.get_nearest_node_indx(rnd_state)
            nearest_node = self.node_list[nearest_node_indx]
            new_node = self.rand_steer(nearest_node)
            if self.state_space.get_bounding_box().contains(new_node.state):
                new_node.parent = nearest_node_indx
                self.add_node(new_node)
                # self.rewire(new_node.id, tolerance=0.02)
                if self.target_region.contains(new_node.state):
                    self.success = True
            if animation and i % 200 == 0:
                self.draw_graph()

    def add_node(self, new_node):
        new_node.id = self.nnodes
        self.node_list.append(copy.deepcopy(new_node))
        node_tuple = tuple(new_node.state.reshape((self.ndim,)))
        self.rtree.insert(0, node_tuple + node_tuple, node_tuple)
        self.tree_dict[node_tuple] = self.nnodes
        self.nnodes += 1
        self.node_mat = np.concatenate((self.node_mat, new_node.state), 1)

    def get_solution(self):
        if not self.success:
            return None
        else:
            solution_list = []
            waypoint = self.node_list[-1]
            solution_list.append(copy.deepcopy(waypoint))
            seen = []
            while waypoint.parent >= 0:
                assert waypoint.parent not in seen
                seen.append(waypoint.parent)
                waypoint = self.node_list[waypoint.parent]
                solution_list.append(copy.deepcopy(waypoint))
            return solution_list

    def steer(self, sample_node: TreeNode, ctrl_input: np.ndarray) -> TreeNode:
        new_state = copy.copy(sample_node.state)
        tot_path = []
        nsteps = int(ctrl_input.shape[0] / self.input_space.ndim)
        for s in range(nsteps):
            crnt_input = ctrl_input[s*self.input_space.ndim:(s+1)*self.input_space.ndim]
            new_state, state_path = self.discrete_dynamics(new_state, crnt_input, return_path=True)
            tot_path.extend(copy.copy(state_path))
        new_node = TreeNode(state=new_state, ctrl_input=ctrl_input, path=tot_path)
        new_node.path_length = 1  # TODO add a path length function to be able to change the length metric
        return new_node

    def rand_steer(self, sample_node: TreeNode) -> TreeNode:
        # random_input = self.input_space_multistep.sample()
        random_input = np.tile(self.input_space.sample(), (self.nsteps, 1))
        new_node = self.steer(sample_node, ctrl_input=random_input)
        return new_node

    def get_nearest_node_indx(self, state: np.ndarray) -> int:
        state_tuple = tuple(state)
        near_point = self.rtree.nearest(state_tuple, 1, objects='raw')
        near_point = next(near_point)
        min_index = self.tree_dict[near_point]
        return min_index

    def get_near_nodes(self, state: np.ndarray, num=50) -> list:
        state_tuple = tuple(state)
        near_points = self.rtree.nearest(state_tuple, num, objects='raw')
        near_node_list = []
        for point in near_points:
            near_node_list.append(self.tree_dict[point])
        return near_node_list

    def draw_graph(self, node_select=-1, draw_path=True, new_fig = True, fig1=None, ax1=None):
        if new_fig and fig1 is None:
            fig1, ax1 = plt.subplots()
        elif fig1 is None:
            fig1 = plt.gcf()
            ax1 = plt.gca()

        if node_select == -1:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            for node in self.node_list:
                if node.parent >= 0:
                    if draw_path:
                        path_x = [self.node_list[node.parent].state[0]]
                        path_y = [self.node_list[node.parent].state[1]]
                        path_x.extend([s[0] for s in node.path])
                        path_y.extend([s[1] for s in node.path])

                        ax1.plot(path_x, path_y, "-g")
                    ax1.plot(node.state[0], node.state[1], "xk")
            ax1.plot(self.root.state[0], self.root.state[1], "xr")
            ax1.grid(True)
            plt.pause(0.01)
        else:
            node = node_select
            if node.parent >= 0:
                ax1.plot(node.path[0, :], node.path[1, :], "-g")
            ax1.plot(node.state[0], node.state[1], "xk")
        return fig1, ax1

    def draw_rand_trajectories(self, states:np.ndarray, draw_path=True, new_fig = True, fig1=None, ax1=None):
        if new_fig and fig1 is None:
            fig1, ax1 = plt.subplots()
        elif fig1 is None:
            fig1 = gcf()
            ax1 = gca()
        n_states = states.shape[1]
        for i in range(n_states):
            state = states[:, i]
            node_idx = self.get_boxnearest(state)
            node = self.node_list[node_idx]
            while node.parent >= 0:
                if draw_path:
                    path_x = [self.node_list[node.parent].state[0]]
                    path_y = [self.node_list[node.parent].state[1]]
                    path_x.extend([s[0] for s in node.path])
                    path_y.extend([s[1] for s in node.path])

                    ax1.plot(path_x, path_y, "-c", linewidth=0.8)
                ax1.plot(node.state[0], node.state[1], "xk")
                node = self.node_list[node.parent]
        ax1.plot(self.root.state[0], self.root.state[1], "xr")
        ax1.grid(True)
        return fig1, ax1

    def rewire(self, node_id: int, tolerance=0.05):
        node = self.node_list[node_id]
        node_cost = self.get_cost(node_id)
        near_nodes = self.get_near_nodes(node.state)
        for near_node_id in near_nodes:
            if near_node_id == node_id:
                continue
            near_node = self.node_list[near_node_id]
            near_cost = self.get_cost(near_node_id)
            if get_distance(node.state, near_node.state) < tolerance:
                if node_cost < near_cost:
                    self.counter += 1
                    near_node.parent = node.parent
                    near_node.path = node.path
                    near_node.input = node.input
                    near_node.path_length = node.path_length
            

    def get_reachable_nodes(self, node_id: int):
        reach_set = self.get_reachable_set(np.array(self.node_list[node_id].state))
        reach_mat = reach_set.pseudo_convert(self.node_mat)
        reach_nodes = np.where(np.max(np.abs(reach_mat), 0) <= 1.1)
        ctrl_input = self.input_space_multistep.center + self.input_space_multistep.generators @ np.squeeze(reach_mat[:, reach_nodes])
        return reach_nodes[0], ctrl_input

    def get_boxnear_nodes(self, state: np.ndarray, box_size: np.ndarray):
        box_near_nodes = np.where(np.all(np.abs(self.node_mat - state) <= box_size, axis=0))
        return box_near_nodes

    def get_boxnear_nodes_sorted(self, state: np.ndarray, box_size: np.ndarray):
        if state.ndim == 1:
            state = state.reshape((state.shape[0], 1))
        if box_size.ndim == 1:
            box_size = box_size.reshape((box_size.shape[0], 1))
        box_near_nodes = np.where(np.all(np.abs(self.node_mat - state) <= box_size, axis=0))
        box_near_nodes = box_near_nodes[0]
        dists = np.max((self.node_mat[:, box_near_nodes] - state) / box_size, axis=0)
        order = np.argsort(dists)
        return box_near_nodes[order]





    def rewire_reachable(self, node_id: int):
        near_nodes, ctrl_input = self.get_reachable_nodes(node_id)
        node = self.node_list[node_id]
        node_cost = self.get_cost(node_id)
        for i in range(len(near_nodes)):
            near_node_id = near_nodes[i]
            parent_cost = self.get_cost(self.node_list[near_node_id].parent)
            if 0 < node_cost < parent_cost:
                assert near_node_id != 0
                self.counter += 1
                near_node = self.node_list[near_node_id]
                near_node.parent = node_id
                new_node = self.steer(node, ctrl_input[:, i])
                near_node.path = new_node.path
                near_node.input = ctrl_input[:, i]
                near_node.path_length = new_node.path_length



    def get_cost(self, node_id: int):
        if node_id == 0 or node_id == -1:
            return 0
        else:
            this_node = self.node_list[node_id]
            return np.linalg.norm(this_node.input) + self.get_cost(this_node.parent)
            # return 1 + self.get_cost(this_node.parent)

    def get_avg_cost(self):
        total_cost = 0
        for n in range(self.nnodes):
            total_cost += self.get_cost(n)
        return total_cost/self.nnodes

    def get_traces(self, trace_time_length: int, overlapping=True) -> np.ndarray:
        """

        @param trace_time_length:
        @param overlapping: Whether it's ok if traces have common time instances (see Hankel vs Page matrices)
        @return:
        """
        n_states = self.state_space.ndim
        n_inputs = self.input_space_multistep.ndim
        instance_length = n_states + n_inputs
        trace_length = trace_time_length * instance_length
        trace = np.zeros((trace_length, 0))
        visited_nodes_list = []
        for starting_node_id in range(len(self.node_list)):
            crnt_node_id = starting_node_id
            valid_trace = True
            crnt_trace = np.zeros((trace_length, 1))
            crnt_trace_ids = []
            for i in range(trace_time_length):
                if crnt_node_id == -1:
                    valid_trace = False
                    break
                crnt_node = copy.deepcopy(self.node_list[crnt_node_id])
                if not overlapping and crnt_node_id in visited_nodes_list:
                    valid_trace = False
                    break
                crnt_trace_ids.append(copy.copy(crnt_node_id))
                state_idx = i * instance_length
                input_idx = state_idx + n_states
                crnt_trace[state_idx:state_idx + n_states] = crnt_node.state
                crnt_trace[input_idx:input_idx + n_inputs] = crnt_node.input
                crnt_node_id = crnt_node.parent

            if valid_trace:
                visited_nodes_list.extend(crnt_trace_ids)
                trace = np.concatenate((trace, crnt_trace), axis=1)

        return trace

    def add_custom_node(self, node_id: int, ctrl_input: np.ndarray) -> int:
        """
        Add a new node by extending an existing node with the specified input.
        """
        new_node = self.steer(self.node_list[node_id], ctrl_input=ctrl_input)
        new_node.parent = node_id
        self.add_node(new_node)
        return self.nnodes - 1

    def get_leaf_nodes_least(self):
        leaf_list = list(range(self.nnodes))
        for i in range(self.nnodes):
            if self.node_list[i].parent in leaf_list:
                leaf_list.remove(self.node_list[i].parent)
        return leaf_list

    def get_state_cost(self, state):
        size = np.array([[0.1], [1.0]])
        near_nodes = [[]]
        while len(near_nodes[0]) == 0:
            near_nodes = self.get_boxnear_nodes(state, size)
            size = 2*size
        state_cost = self.get_cost(near_nodes[0][0])
        return state_cost

    def get_boxnearest(self, state):
        size = np.array([[0.1], [1.0]])
        near_nodes = []
        while len(near_nodes) == 0:
            near_nodes = self.get_boxnear_nodes_sorted(state, size)
            size = 2 * size
        return near_nodes[0]



    def get_control(self, state):
        size = np.array([[0.1], [1.0]])
        near_nodes = self.get_boxnear_nodes(state, size)
        if len(near_nodes[0]) > 0:
            ctrl = self.node_list[near_nodes[0][0]].input
        else:
            ctrl = np.array([[0]])

        if ctrl is None:
            ctrl = np.array([[0]])
        return ctrl.reshape((ctrl.size,1))

