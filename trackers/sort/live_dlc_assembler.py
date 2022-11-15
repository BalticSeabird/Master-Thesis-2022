"""
Live assembler adapted from DeepLabCut2.2 toolbox inferenceutils

--------- reference -----------

DeepLabCut2.2 Toolbox (deeplabcut.org)
https://github.com/AlexEMG/DeepLabCut

"""
import heapq
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt, erf
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import softmax
from typing import Tuple


Position = Tuple[float, float]

@dataclass(frozen=True)
class Joint:
    pos: Position
    confidence: float = 1.0
    label: int = None
    idx: int = None
    group: int = -1


class Link:
    def __init__(self, j1, j2, affinity=1):
        self.j1 = j1
        self.j2 = j2
        self.affinity = affinity
        self._length = sqrt((j1.pos[0] - j2.pos[0]) ** 2 + (j1.pos[1] - j2.pos[1]) ** 2)

    def __repr__(self):
        return (
            f"Link {self.idx}, affinity={self.affinity:.2f}, length={self.length:.2f}"
        )

    @property
    def confidence(self):
        return self.j1.confidence * self.j2.confidence

    @property
    def idx(self):
        return self.j1.idx, self.j2.idx

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    def to_vector(self):
        return [*self.j1.pos, *self.j2.pos]


class Assembly:
    def __init__(self, size):
        self.data = np.full((size, 4), np.nan)
        self.confidence = 0  # 0 by defaut, overwritten otherwise with `add_joint`
        self._affinity = 0
        self._links = []
        self._visible = set()
        self._idx = set()
        self._dict = dict()

    def __len__(self):
        return len(self._visible)

    def __contains__(self, assembly):
        return bool(self._visible.intersection(assembly._visible))

    def __add__(self, other):
        if other in self:
            raise ValueError("Assemblies contain shared joints.")

        assembly = Assembly(self.data.shape[0])
        for link in self._links + other._links:
            assembly.add_link(link)
        return assembly

    @classmethod
    def from_array(cls, array):
        n_bpts, n_cols = array.shape
        ass = cls(size=n_bpts)
        ass.data[:, :n_cols] = array
        visible = np.flatnonzero(~np.isnan(array).any(axis=1))
        if n_cols < 3:  # Only xy coordinates are being set
            ass.data[visible, 2] = 1  # Set detection confidence to 1
        ass._visible.update(visible)
        return ass

    @property
    def xy(self):
        return self.data[:, :2]

    @property
    def extent(self):
        bbox = np.empty(4)
        bbox[:2] = np.nanmin(self.xy, axis=0)
        bbox[2:] = np.nanmax(self.xy, axis=0)
        return bbox

    @property
    def area(self):
        x1, y1, x2, y2 = self.extent
        return (x2 - x1) * (y2 - y1)

    @property
    def confidence(self):
        return np.nanmean(self.data[:, 2])

    @confidence.setter
    def confidence(self, confidence):
        self.data[:, 2] = confidence

    @property
    def soft_identity(self):
        data = self.data[~np.isnan(self.data).any(axis=1)]
        unq, idx, cnt = np.unique(data[:, 3], return_inverse=True, return_counts=True)
        avg = np.bincount(idx, weights=data[:, 2]) / cnt
        soft = softmax(avg)
        return dict(zip(unq.astype(int), soft))

    @property
    def affinity(self):
        n_links = self.n_links
        if not n_links:
            return 0
        return self._affinity / n_links

    @property
    def n_links(self):
        return len(self._links)

    def intersection_with(self, other):
        x11, y11, x21, y21 = self.extent
        x12, y12, x22, y22 = other.extent
        x1 = max(x11, x12)
        y1 = max(y11, y12)
        x2 = min(x21, x22)
        y2 = min(y21, y22)
        if x2 < x1 or y2 < y1:
            return 0
        ll = np.array([x1, y1])
        ur = np.array([x2, y2])
        xy1 = self.xy[~np.isnan(self.xy).any(axis=1)]
        xy2 = other.xy[~np.isnan(other.xy).any(axis=1)]
        in1 = np.all((xy1 >= ll) & (xy1 <= ur), axis=1).sum()
        in2 = np.all((xy2 >= ll) & (xy2 <= ur), axis=1).sum()
        return min(in1 / len(self), in2 / len(other))

    def add_joint(self, joint):
        if joint.label in self._visible or joint.label is None:
            return False
        self.data[joint.label] = *joint.pos, joint.confidence, joint.group
        self._visible.add(joint.label)
        self._idx.add(joint.idx)
        return True

    def remove_joint(self, joint):
        if joint.label not in self._visible:
            return False
        self.data[joint.label] = np.nan
        self._visible.remove(joint.label)
        self._idx.remove(joint.idx)
        return True

    def add_link(self, link, store_dict=False):
        if store_dict:
            # Selective copy; deepcopy is >5x slower
            self._dict = {
                "data": self.data.copy(),
                "_affinity": self._affinity,
                "_links": self._links.copy(),
                "_visible": self._visible.copy(),
                "_idx": self._idx.copy(),
            }
        i1, i2 = link.idx
        if i1 in self._idx and i2 in self._idx:
            self._affinity += link.affinity
            self._links.append(link)
            return False
        if link.j1.label in self._visible and link.j2.label in self._visible:
            return False
        self.add_joint(link.j1)
        self.add_joint(link.j2)
        self._affinity += link.affinity
        self._links.append(link)
        return True

    def calc_pairwise_distances(self):
        return pdist(self.xy, metric="sqeuclidean")


class Assembler:
    def __init__(
        self,
        max_n_individuals,
        n_multibodyparts,
        graph=None,
        paf_inds=None,
        greedy=False,
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=2,
        max_overlap=0.8,
        nan_policy="little",
        window_size=0,
        method="m1",
    ):

        self.max_n_individuals = max_n_individuals
        self.n_multibodyparts = n_multibodyparts
        self.greedy = greedy
        self.pcutoff = pcutoff
        self.min_affinity = min_affinity
        self.min_n_links = min_n_links
        self.max_overlap = max_overlap

        self.nan_policy = nan_policy
        self.window_size = window_size
        self.method = method

        # TODO: fix inputs
        self.graph = graph
        self.paf_inds = paf_inds

        self._gamma = 0.01
        self._trees = dict()
        self.safe_edge = False
        self._kde = None
        self.assemblies = dict()
        self.unique = dict()

    @staticmethod
    def _flatten_detections(data_dict):
        ind = 0
        coordinates = data_dict["coordinates"][0]
        confidence = data_dict["confidence"]
        ids = data_dict.get("identity", None)
        if ids is None:
            ids = [np.ones(len(arr), dtype=int) * -1 for arr in confidence]
        else:
            ids = [arr.argmax(axis=1) for arr in ids]
        for i, (coords, conf, id_) in enumerate(zip(coordinates, confidence, ids)):
            if not np.any(coords):
                continue
            for xy, p, g in zip(coords, conf, id_):
                joint = Joint(tuple(xy), p.item(), i, ind, g)
                ind += 1
                yield joint


    def set_max_individuals(self, n_max_ind):
        self.max_n_individuals = n_max_ind

    # bbox_points: is Dictionary with points in bbox {0: [], 1: [], 2:[], ...}
    def extract_best_links(self, joints_dict, costs, bbox_points = None, trees=None):
        links = []
        for ind in self.paf_inds:
            s, t = self.graph[ind]
            dets_s = joints_dict.get(s, None)
            dets_t = joints_dict.get(t, None)
            if dets_s is None or dets_t is None:
                continue
            if ind not in costs:
                continue
            lengths = costs[ind]["distance"]
            if np.isinf(lengths).all():
                continue

            aff = costs[ind][self.method].copy()
            aff[np.isnan(aff)] = 0

            #print(f'aff original: {aff}')
            if bbox_points is not None:
                row_idx = set(range(aff.shape[0]))
                col_idx = set(range(aff.shape[1]))

                # Index for s and t points in current bounding box
                bb_s_idx = set(bbox_points[s])
                bb_t_idx = set(bbox_points[t])

                #print(f's_idx: {bb_s_idx}')
                #print(f't_idx: {bb_t_idx}')
                # Index not to be considered
                rm_s_idx = list(row_idx.difference(bb_s_idx))
                rm_t_idx = list(col_idx.difference(bb_t_idx))

                # Mask connections with key points outside of box
                aff[rm_s_idx,:] = 0
                aff[:,rm_t_idx] = 0

            #print(f'Masked aff: {aff}')
            if trees:
                vecs = np.vstack(
                    [[*det_s.pos, *det_t.pos] for det_s in dets_s for det_t in dets_t]
                )
                dists = []
                for n, tree in enumerate(trees, start=1):
                    d, _ = tree.query(vecs)
                    dists.append(np.exp(-self._gamma * n * d))
                w = np.mean(dists, axis=0)
                aff *= w.reshape(aff.shape)

            if self.greedy:
                conf = np.asarray(
                    [
                        [det_s.confidence * det_t.confidence for det_t in dets_t]
                        for det_s in dets_s
                    ]
                )
                rows, cols = np.where(
                    (conf >= self.pcutoff * self.pcutoff) & (aff >= self.min_affinity)
                )
                candidates = sorted(
                    zip(rows, cols, aff[rows, cols], lengths[rows, cols]),
                    key=lambda x: x[2],
                    reverse=True,
                )
                i_seen = set()
                j_seen = set()
                for i, j, w, l in candidates:
                    if i not in i_seen and j not in j_seen:
                        i_seen.add(i)
                        j_seen.add(j)
                        links.append(Link(dets_s[i], dets_t[j], w))
                        if len(i_seen) == self.max_n_individuals:
                            break
            # else:  # Optimal keypoint pairing
            #     inds_s = sorted(
            #         range(len(dets_s)), key=lambda x: dets_s[x].confidence, reverse=True
            #     )[: self.max_n_individuals]
            #     inds_t = sorted(
            #         range(len(dets_t)), key=lambda x: dets_t[x].confidence, reverse=True
            #     )[: self.max_n_individuals]
            #     keep_s = [
            #         ind for ind in inds_s if dets_s[ind].confidence >= self.pcutoff
            #     ]
            #     keep_t = [
            #         ind for ind in inds_t if dets_t[ind].confidence >= self.pcutoff
            #     ]
            #     aff = aff[np.ix_(keep_s, keep_t)]
            #     rows, cols = linear_sum_assignment(aff, maximize=True)
            #     for row, col in zip(rows, cols):
            #         w = aff[row, col]
            #         if w >= self.min_affinity:
            #             links.append(Link(dets_s[keep_s[row]], dets_t[keep_t[col]], w))
        return links

    def _fill_assembly(self, assembly, lookup, assembled, safe_edge, nan_policy):
        stack = []
        visited = set()
        tabu = []
        counter = itertools.count()

        def push_to_stack(i):
            for j, link in lookup[i].items():
                if j in assembly._idx:
                    continue
                if link.idx in visited:
                    continue
                heapq.heappush(stack, (-link.affinity, next(counter), link))
                visited.add(link.idx)

        for idx in assembly._idx:
            push_to_stack(idx)

        while stack and len(assembly) < self.n_multibodyparts:
            _, _, best = heapq.heappop(stack)
            i, j = best.idx
            if i in assembly._idx:
                new_ind = j
            elif j in assembly._idx:
                new_ind = i
            else:
                continue
            if new_ind in assembled:
                continue
            assembly.add_link(best)
            push_to_stack(new_ind)

    def build_assemblies(self, links):
        lookup = defaultdict(dict)
        for link in links:
            i, j = link.idx
            lookup[i][j] = link
            lookup[j][i] = link

        assemblies = []
        assembled = set()

        # Fill the subsets with unambiguous, complete individuals
        G = nx.Graph([link.idx for link in links])
        #print(f'nx Graph: {G}')
        for chain in nx.connected_components(G):
            #print(len(chain))
            if len(chain) == self.n_multibodyparts:
                edges = [tuple(sorted(edge)) for edge in G.edges(chain)]
                assembly = Assembly(self.n_multibodyparts)
                for link in links:
                    i, j = link.idx
                    if (i, j) in edges:
                        success = assembly.add_link(link)
                        if success:
                            lookup[i].pop(j)
                            lookup[j].pop(i)
                assembled.update(assembly._idx)
                assemblies.append(assembly)

        if len(assemblies) == self.max_n_individuals:
            return assemblies, assembled

        for link in sorted(links, key=lambda x: x.affinity, reverse=True):
            if any(i in assembled for i in link.idx):
                continue
            assembly = Assembly(self.n_multibodyparts)
            assembly.add_link(link)
            self._fill_assembly(
                assembly, lookup, assembled, self.safe_edge, self.nan_policy
            )
            for link in assembly._links:
                i, j = link.idx
                lookup[i].pop(j)
                lookup[j].pop(i)
            assembled.update(assembly._idx)
            assemblies.append(assembly)

        # Fuse superfluous assemblies
        n_extra = len(assemblies) - self.max_n_individuals
        if n_extra > 0:
            store = dict()
            for assembly in assemblies:
                if len(assembly) != self.n_multibodyparts:
                    for i in assembly._idx:
                        store[i] = assembly
            used = [link for assembly in assemblies for link in assembly._links]
            unconnected = [link for link in links if link not in used]
            for link in unconnected:
                i, j = link.idx
                try:
                    if store[j] not in store[i]:
                        temp = store[i] + store[j]
                        store[i].__dict__.update(temp.__dict__)
                        assemblies.remove(store[j])
                        for idx in store[j]._idx:
                            store[idx] = store[i]
                except KeyError:
                    pass

        # Second pass without edge safety
        for assembly in assemblies:
            if len(assembly) != self.n_multibodyparts:
                self._fill_assembly(assembly, lookup, assembled, False, "")
                assembled.update(assembly._idx)

        return assemblies, assembled

    def assemble(self, data_dict, ind_frame, bbox_points = None):
        '''
        data_dict - coordinates, confidence, and costs for keypoints
        ind_frame - index of current frame
        '''

        # Iterable containing joint information
        joints = list(self._flatten_detections(data_dict))
        if not joints:
            return None, None

        # Dictionary with keypoint information
        bag = defaultdict(list)
        for joint in joints:
            bag[joint.label].append(joint)

        assembled = set()

        # TODO: remove unique
        unique = None

        if not any(i in bag for i in range(self.n_multibodyparts)):
            return None, unique

        trees = []
        for j in range(1, self.window_size + 1):
            tree = self._trees.get(ind_frame - j, None)
            if tree is not None:
                trees.append(tree)

        links = self.extract_best_links(bag, data_dict["costs"], bbox_points, trees = trees)

        if self.window_size >= 1 and links:
            # Store selected edges for subsequent frames
            vecs = np.vstack([link.to_vector() for link in links])
            self._trees[ind_frame] = cKDTree(vecs)

        assemblies, assembled_ = self.build_assemblies(links)
        assembled.update(assembled_)

        # Remove invalid assemblies
        discarded = set(
            joint
            for joint in joints
            if joint.idx not in assembled and np.isfinite(joint.confidence)
        )
        for assembly in assemblies[::-1]:
            if 0 < assembly.n_links < self.min_n_links or not len(assembly):
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
                assemblies.remove(assembly)
        if 0 < self.max_overlap < 1:  # Non-maximum pose suppression
            scores = [ass._affinity for ass in assemblies]
            lst = list(zip(scores, assemblies))
            assemblies = []
            while lst:
                temp = max(lst, key=lambda x: x[0])
                lst.remove(temp)
                assemblies.append(temp[1])
                for pair in lst[::-1]:
                    if temp[1].intersection_with(pair[1]) >= self.max_overlap:
                        lst.remove(pair)
        if len(assemblies) > self.max_n_individuals:
            assemblies = sorted(assemblies, key=len, reverse=True)
            for assembly in assemblies[self.max_n_individuals :]:
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
            assemblies = assemblies[: self.max_n_individuals]

        return assemblies, unique
