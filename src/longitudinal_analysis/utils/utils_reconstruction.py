import os
import glob
import torch
import numpy as np
import SimpleITK as sitk

from typing import Dict, List
from scipy.spatial import cKDTree

from scipy.spatial.distance import cdist
from torch_cluster import knn
import pyvista as pv

import vtk
import trimesh
import src.longitudinal_analysis.utils.utils_torch as utils_torch
import tqdm

from sklearn.decomposition import PCA
from shapely import Polygon
import matplotlib.pyplot as plt


class VascularModel:
    """Vascular model object for centerline correction and pruning"""

    def __init__(self, contours: Dict[str, np.array], npoints: int = 128):
        self.contours = {name: contour.reshape(-1, npoints, 3) for name, contour in contours.items()}
        self.centerlines = {name: contour.mean(axis=1) for name, contour in self.contours.items()}
        self.correct_dict = {
            name: np.ones(len(centerline), dtype=bool) for name, centerline in self.centerlines.items()
        }

    def get_corrected_contour(self, branch: str, connect_to: np.array = None):
        contours = self.contours[branch][self.correct_dict[branch]]
        centerline = contours.mean(axis=1)

        if connect_to is not None:
            if np.linalg.norm(vec1 := (connect_to - centerline[-1])) > np.linalg.norm(
                vec0 := (connect_to - centerline[0])
            ):
                contours = np.concatenate([(contours[0] + vec0).reshape(1, -1, 3), contours])
            else:
                contours = np.concatenate([contours, (contours[-1] + vec1).reshape(1, -1, 3)])

        return contours

    def _get_merge_point(self, 
                         main_centerline: np.array, 
                         segment_centerline: np.array, tol: float):
        
        main_points = torch.tensor(main_centerline)
        segment_points = torch.tensor(segment_centerline)

        _, row = knn(main_points, segment_points, 1)
        distance = torch.linalg.norm(segment_points - main_points[row], dim=1)
        merge_mask = (distance < tol).int()

        if distance[0] > distance[-1]:
            if merge_mask.sum() == 0:
                threshold = len(merge_mask) - 1
            else:
                threshold = torch.argwhere(merge_mask).min()

            merge_mask = np.zeros_like(merge_mask)
            merge_mask[threshold:] = 1
        else:
            if merge_mask.sum() == 0:
                threshold = 0
            else:
                threshold = torch.argwhere(merge_mask).max()

            merge_mask = np.zeros_like(merge_mask)
            merge_mask[:threshold] = 1

        return segment_points[threshold].numpy(), ~merge_mask.astype(bool)

    def correct_bifurcation(self, 
                            main_branch: str, 
                            side_branch_1: str, 
                            side_branch_2: str, 
                            merge_tol: float):
        
        def get_bifurcation_point(branch: np.array, anchor1: np.array, anchor2: np.array):
            top_point = branch[np.argmax(branch[:, -1])]

            # Specify higher point to be bifurcation point
            if np.linalg.norm(top_point - anchor1) > np.linalg.norm(top_point - anchor2):
                anchor = anchor2
            else:
                anchor = anchor1

            return anchor

        def trim_by_bifurcation(branch: np.array, bifurcation_point: np.array, z_mode: str):
            index = np.argmin(np.linalg.norm(branch - bifurcation_point, axis=1))
            point_mask = np.zeros(len(branch))

            if index > 0 and index < len(branch) - 1:
                if branch[0, -1] > branch[-1, -1]:
                    if z_mode == "min":
                        point_mask[index:] = 1
                    else:
                        point_mask[: index + 1] = 1
                else:
                    if z_mode == "min":
                        point_mask[: index + 1] = 1
                    else:
                        point_mask[index:] = 1

                return point_mask.astype(bool)

            else:
                return ~(point_mask.astype(bool))

        anchor_1, _ = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch_1], merge_tol)
        anchor_2, _ = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch_2], merge_tol)
        bifurcation_point = get_bifurcation_point(self.centerlines[main_branch], anchor_1, anchor_2)

        self.correct_dict[side_branch_1] = trim_by_bifurcation(
            self.centerlines[side_branch_1], bifurcation_point, "min"
        )
        self.correct_dict[side_branch_2] = trim_by_bifurcation(
            self.centerlines[side_branch_2], bifurcation_point, "min"
        )
        self.correct_dict[main_branch] = trim_by_bifurcation(self.centerlines[main_branch], bifurcation_point, "max")

        endings = self.centerlines[main_branch][self.correct_dict[main_branch]][[0, -1]]
        main_anchor = np.argmin(np.linalg.norm(endings - bifurcation_point, axis=1))

        return endings[main_anchor]

    def correct_overlap(self, main_branch: str, side_branch: str, merge_tol: float):
        _, point_mask = self._get_merge_point(self.centerlines[main_branch], self.centerlines[side_branch], merge_tol)
        self.correct_dict[side_branch] = point_mask.astype(bool)

    def get_pruned_contour(
        self,
        branch: str,
        ref_branch: str,
        distance: float,
        connect_centerline: bool = True,
        keep_overlap: bool = True,
    ):
        branch_centerline = self.centerlines[branch][self.correct_dict[branch]]
        ref_branch_centerline = self.centerlines[ref_branch][self.correct_dict[ref_branch]]

        # Determine direction of pruning
        all_distances = cdist(branch_centerline[[0, -1]], ref_branch_centerline)
        indices = np.argmin(all_distances, axis=1)
        distances = np.min(all_distances, axis=1)

        # Prune on determined side to given distance
        if distances[0] > distances[1]:
            cumdist = np.cumsum(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1))
            ncontours = np.argwhere(cumdist < 10 * distance).max()

            # Prune and include overlap
            if keep_overlap:
                pruned_contour = np.concatenate(
                    [
                        self.contours[branch][self.correct_dict[branch]][-ncontours:],
                        self.contours[branch][~self.correct_dict[branch]],
                    ]
                )
            else:
                pruned_contour = self.contours[branch][self.correct_dict[branch]][-ncontours:]

            # Connect to ref branch if necessary
            pruned_centerline = pruned_contour.mean(axis=1)
            if connect_centerline:
                root = pruned_centerline[-1]

                tangent = ref_branch_centerline[indices[1]] - root
                unit_tangent = tangent / np.linalg.norm(tangent)

                spacing = np.mean(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1))
                nrings = int(np.linalg.norm(tangent) / spacing)

                for _ in range(nrings - 1):
                    vec = spacing * unit_tangent
                    pruned_centerline = np.concatenate([pruned_centerline, (pruned_centerline[-1] + vec)[None]])

        else:
            cumdist = np.cumsum(np.linalg.norm(np.diff(branch_centerline[::-1], axis=0), axis=1))
            ncontours = np.argwhere(cumdist < 10 * distance).max()

            # Prune and include overlap
            if keep_overlap:
                pruned_contour = np.concatenate(
                    [
                        self.contours[branch][~self.correct_dict[branch]],
                        self.contours[branch][self.correct_dict[branch]][:ncontours],
                    ]
                )
            else:
                pruned_contour = self.contours[branch][self.correct_dict[branch]][:ncontours]

            # Connect to ref branch if necessary
            pruned_centerline = pruned_contour.mean(axis=1)
            if connect_centerline:
                root = pruned_centerline[0]

                tangent = ref_branch_centerline[indices[0]] - root
                unit_tangent = tangent / np.linalg.norm(tangent)

                spacing = np.mean(np.linalg.norm(np.diff(branch_centerline, axis=0), axis=1)) / 2
                nrings = int(np.linalg.norm(tangent) / spacing)

                for _ in range(nrings - 1):
                    vec = spacing * unit_tangent
                    pruned_centerline = np.concatenate([(pruned_centerline[0] + vec)[None], pruned_centerline])

        return pruned_contour, pruned_centerline
    
    @classmethod
    def load_from_directory(
        cls,
        root_dir: str,
        filenames: List[str],
        extension: str = ".vtp",
        npoints: int = 128,
    ):  
        
        contours = {
            filename: pv.read(glob.glob(os.path.join(root_dir, f"*{filename}*{extension}"))[0]).points
            for filename in filenames
        }
        return cls(contours, npoints)

class IliacVascularModel(VascularModel):

    def __init__(self,
                 centerlines : Dict[str, np.array],
                 contours : Dict[str, np.array],
                 npoints : int):
        
        self.centerlines = {name: centerline for name, centerline in centerlines.items()}
        self.contours = {name : contour.reshape(-1, npoints, 3) for name, contour in contours.items()}
        self.ref_centerline = {name : contour.mean(axis = 1) for name, contour in self.contours.items()}

class CarotidVascularModel(VascularModel):

    def __init__(self,
                 centerlines : Dict[str, np.array],
                 contours : Dict[str, np.array],
                 npoints : int,
                 mask_image : sitk.Image | None):
        """
        Initialize not with the average of the contours, since if PTF was used if gets rid of the
        correction done by it. If PTF was used, the centerlines are no longer the average of the 
        contours.
        """
        self.centerlines = {name: centerline for name, centerline in centerlines.items()}

        self.contours = {name: contour.reshape(-1, npoints, 3) for name, contour in contours.items()}
        self.ref_centerline = {name: contour.mean(axis=1) for name, contour in self.contours.items()}

        self.mask_image = mask_image

    def detect_bifurcation(self,
                           vessel_1 : str,
                           vessel_2 : str, 
                           merge_tol : float):
        
        """
        Returns an averaged bifurcation point, and the detected bifurcation point for both
        initial centerlines.
        """
        point_1, _ = self._get_merge_point(self.centerlines[vessel_1],
                                           self.centerlines[vessel_2],
                                           tol = merge_tol)

        point_2, _ = self._get_merge_point(self.centerlines[vessel_2],
                                           self.centerlines[vessel_1],
                                           tol = merge_tol)
        bifurcation = 0.5 * (point_1 + point_2)

        return bifurcation, point_1, point_2
    
    def build_common_cca(self, cca1, cca2, tol=1.5, average=True):
        """
        Computes an average of the common carotid, or returns the largest one.
        """
        D = cdist(cca1, cca2)
        min_dist_1 = D.min(axis=1)

        valid_1 = min_dist_1 < tol
        if valid_1.sum() == 0:
            return None

        last_idx_1 = np.where(valid_1)[0].max()
        cca1_common = cca1[:last_idx_1 + 1]

        last_point = cca1_common[-1]
        idx2 = np.argmin(np.linalg.norm(cca2 - last_point, axis=1))
        cca2_common = cca2[:idx2 + 1]

        n = min(len(cca1_common), len(cca2_common))
        cca1_common = cca1_common[:n]
        cca2_common = cca2_common[:n]

        if average:
            return 0.5 * (cca1_common + cca2_common)
        else:
            return cca1_common if len(cca1_common) <= len(cca2_common) else cca2_common

    def identify_common_trunk(self,
                              part_a: np.ndarray, 
                              part_b: np.ndarray, 
                              other_centerline: np.ndarray, 
                              tol: float = 1.5):
        """
        Decide which split part corresponds to the common carotid trunk (CCA)
        by checking which part overlaps more with the other centerline.
        """
        def overlap_score(part, other, tol):
            D = cdist(part, other)
            min_dist = D.min(axis=1)
            return np.sum(min_dist < tol)

        score_a = overlap_score(part_a, other_centerline, tol)
        score_b = overlap_score(part_b, other_centerline, tol)

        if score_a >= score_b:
            return part_a, part_b, score_a, score_b  # cca, branch
        else:
            return part_b, part_a, score_a, score_b  # cca, branch
    
    def split_carotid_branches(self,
                               ica_cca : str, 
                               eca_cca : str, 
                               bif_point):
        """
        Dividing the centerline in ICA, ECA, CCA.
        To do: Figure out how to merge a distal ECA/ICA.
        """
        def split_by_bifurcation(centerline, bif_point):
            # Looking for the closest point to the bifurcation 
            # point of the provided centerline.
            distances = np.linalg.norm(centerline - bif_point, axis=1)
            idx = np.argmin(distances)

            part1 = centerline[:idx+1]
            part2 = centerline[idx:]
            return part1, part2, idx
        
        # Initialization of the dict.
        results = {}
        # Splitting the centerlines and then identifying which ones is which.
        # Which ones is CCA an which one is ICA/ECA.
        part1_ica, part2_ica, _ = split_by_bifurcation(self.centerlines[ica_cca], bif_point)
        part1_eca, part2_eca, _ = split_by_bifurcation(self.centerlines[eca_cca], bif_point)

        cca_from_ica, ica_part, s1, s2 = self.identify_common_trunk(
            part1_ica, part2_ica, self.centerlines[eca_cca], tol=1.5
        )

        cca_from_eca, eca_part, s3, s4 = self.identify_common_trunk(
            part1_eca, part2_eca, self.centerlines[ica_cca], tol=1.5
        )
        
        # Splitting the contours using the reference centerlines (average of the contours).
        # Then we have to do the same identification.
        _, _, ctr_idx_ica = split_by_bifurcation(self.ref_centerline[ica_cca], bif_point)
        
        _, _, ctr_idx_eca = split_by_bifurcation(self.ref_centerline[eca_cca], bif_point)
        
        if s1 >= s2:
            results["ICA_contours"] = self.contours[ica_cca][ctr_idx_ica:]
            results["CCA_from_ICA_contours"] = self.contours[ica_cca][:ctr_idx_ica+1]
        else:
            results["ICA_contours"] = self.contours[ica_cca][:ctr_idx_ica+1]
            results["CCA_from_ICA_contours"] = self.contours[ica_cca][ctr_idx_ica:]

        if s3 >= s4:
            results["ECA_contours"] = self.contours[eca_cca][ctr_idx_eca:]
            results["CCA_from_ECA_contours"] = self.contours[eca_cca][:ctr_idx_eca+1]
        else:
            results["ECA_contours"] = self.contours[eca_cca][:ctr_idx_eca+1]
            results["CCA_from_ECA_contours"] = self.contours[eca_cca][ctr_idx_eca:]

        # Probably for now we can keep every part of the carotid trunk.
        results.update({"ICA": ica_part,
                        "CCA_from_ICA" : cca_from_ica,
                        "ECA": eca_part,
                        "CCA_from_ECA": cca_from_eca,
                        # "CCA": self.build_common_cca(cca_from_eca, cca_from_ica, 1, True),                            
                        "bifurcation_point" : bif_point})
        
        return results

    @staticmethod
    def _centerline_to_polydata(centerline: np.ndarray) -> pv.PolyData:
        centerline = np.asarray(centerline, dtype=float)

        if centerline.ndim != 2 or centerline.shape[1] != 3:
            raise ValueError(f"Centerline must have shape (N, 3), got {centerline.shape}")

        if len(centerline) < 2:
            raise ValueError("A centerline needs at least 2 points to be saved as a polyline.")

        # Casting all the data from the centerlines to a PolyData array so we can save them.
        poly = pv.PolyData(centerline)
        lines = np.hstack(([len(centerline)], np.arange(len(centerline)))).astype(np.int64)
        poly.lines = lines

        return poly
    
    @staticmethod
    def _contours_to_polydata(contours: np.ndarray, close_loop: bool = True) -> pv.PolyData:
        contours = np.asarray(contours, dtype=float)

        if contours.ndim != 3 or contours.shape[2] != 3:
            raise ValueError(f"Contours must have shape (N, npoints, 3), got {contours.shape}")

        n_contours, npoints, _ = contours.shape

        if n_contours == 0:
            raise ValueError("Contours array is empty.")

        # Flatten all points
        all_points = contours.reshape(-1, 3)

        lines = []
        for i in range(n_contours):
            start = i * npoints
            ids = np.arange(start, start + npoints, dtype=np.int64)

            if close_loop: # closed polyline: repeat first point at the end
                line = np.hstack(([npoints + 1], ids, ids[0]))
            else:
                line = np.hstack(([npoints], ids))

            lines.append(line)
        lines = np.hstack(lines).astype(np.int64)

        poly = pv.PolyData()
        poly.points = all_points
        poly.lines = lines

        return poly
    
    def save_split_carotid_branches_vtp(split_dict: dict,
                                        output_dir: str,
                                        prefix: str = "",
                                        sufix : str = "",
                                        branch_names : str = None,
                                        save_bifurcation: bool = False):
    
        output_dir = os.path.join(output_dir, "final_centerlines")
        os.makedirs(output_dir, exist_ok=True)

        if branch_names == None:
            branch_names = ["ICA", "ECA", "CCA", "CCA_from_ICA", "CCA_from_ECA"]

        for name in branch_names:
            if name not in split_dict:
                print(f"Skipping {name}: not found in split_dict")
                continue

            centerline = split_dict[name]
            poly = CarotidVascularModel._centerline_to_polydata(centerline)

            out_path = os.path.join(output_dir, f"{prefix}{name}{sufix}.vtp")
            poly.save(out_path)
            # print(f"Saved {name} to: {out_path}")

        if save_bifurcation and "bifurcation_point" in split_dict:

            # This is only a point, is the detected bifurcation point of the carotids.
            bif = np.asarray(split_dict["bifurcation_point"], dtype=float).reshape(1, 3)
            bif_poly = pv.PolyData(bif)
            out_path = os.path.join(output_dir, f"{prefix}bifurcation_point{sufix}.vtp")
            bif_poly.save(out_path)
            # print(f"Saved bifurcation point to: {out_path}")

    def save_split_carotid_contours_vtp(split_dict, 
                                        output_dir, 
                                        prefix="",
                                        sufix="",
                                        contour_names : str = None):
        
        output_dir = os.path.join(output_dir, "final_contours")
        os.makedirs(output_dir, exist_ok=True)

        if contour_names == None:
            contour_names = [
                "ICA_contours",
                "ECA_contours",
                "CCA_from_ICA_contours",
                "CCA_from_ECA_contours",
            ]

        for name in contour_names:
            if name not in split_dict:
                print(f"Skipping {name}: not found in split_dict")
                continue

            poly = CarotidVascularModel._contours_to_polydata(split_dict[name], close_loop=True)
            out_path = os.path.join(output_dir, f"{prefix}{name}{sufix}.vtp")
            poly.save(out_path)
            # print(f"Saved {name} to: {out_path}")
    
    def save_data(split_dict : Dict,
                  output_dir : str,
                  prefix : str = "",
                  sufix : str = "",
                  save_branches : bool = True,
                  save_contours : bool = True,
                  save_mask : bool = True,
                  branch_names : list = None,
                  contour_names : list = None):
        
        if save_branches:
            CarotidVascularModel.save_split_carotid_branches_vtp(split_dict,
                                                                 output_dir,
                                                                 prefix,
                                                                 sufix,
                                                                 branch_names = branch_names,
                                                                 save_bifurcation = True)
        
        if save_contours:
            CarotidVascularModel.save_split_carotid_contours_vtp(split_dict,
                                                                 output_dir,
                                                                 prefix,
                                                                 sufix,
                                                                 contour_names = contour_names)

    @classmethod
    def load_from_directory(
        cls,
        root_dir_centerlines: str,
        root_dir_contours : str,
        filenames: List[str],
        dir_mask : str = None,
        npoints: int = 128,
        extension: str = ".vtp"
    ):  
        centerlines, contours = {}, {} # Empty dictionary.
        for filename in filenames:
            try:
                centerlines[filename] = pv.read(glob.glob(os.path.join(root_dir_centerlines, f"*{filename}*{extension}"))[0]).points
            except:
                print(f"Centerline not found for vessel: {filename}")

        for filename in filenames:
            try:
                contours[filename] = pv.read(glob.glob(os.path.join(root_dir_contours, f"*{filename}*{extension}"))[0]).points
            except:
                print(f"Contours not found for vessel : {filename}")

        # Optionally, we can also saved the mask from TS. Useful when defining INR local ROI.
        if dir_mask != None:
            try:
                mask_image = sitk.ReadImage(dir_mask)
            except:
                print(f"Skipping saving, mask could not be found at {dir_mask}")
                mask_image = None

        return cls(centerlines, contours, npoints, mask_image)

class MatchVascularModel:
    """
    Match arclength-wise the centerlines and contours of two CarotidVascularModel instances.

    Important:
    - Corrected centerlines are cropped using their own arclength.
    - Contours are cropped using the arclength of their reference centerline
      (computed as contour.mean(axis=1)), because contours and corrected centerlines
      do not necessarily have point-to-point correspondence.
    """

    def __init__(self,
                 series_1: CarotidVascularModel,
                 series_2: CarotidVascularModel,
                 split_vessels : bool,
                 output_dir: str):

        self.output_dir = output_dir
        self.split_vessels = split_vessels

        print("Matching the carotid models.")
        # # First we have to merge the possible distal ICA with the other ICA.
        # self._merge_centerlines_contours(series_1)
        # self._merge_centerlines_contours(series_2)

        left_series_1, right_series_1 = self._get_carotid_trunk(series_1)
        left_series_2, right_series_2 = self._get_carotid_trunk(series_2)

        if left_series_1.keys() != left_series_2.keys():
            raise KeyError("Left carotid dictionaries do not have the same keys.")

        if right_series_1.keys() != right_series_2.keys():
            raise KeyError("Right carotid dictionaries do not have the same keys.")

        # Match both sides automatically
        self.left_matched = self._match_single_side(
            side_name="left",
            series_1=left_series_1,
            series_2=left_series_2
        )

        self.right_matched = self._match_single_side(
            side_name="right",
            series_1=right_series_1,
            series_2=right_series_2
        )

        if isinstance(series_1.mask_image, sitk.Image) and isinstance(series_2.mask_image, sitk.Image):
            self._save_masks(series_1.mask_image, series_2.mask_image)

    def _save_masks(self,
                    series_1_image : sitk.Image,
                    series_2_image : sitk.Image):
        
        output_dir = os.path.join(self.output_dir, "series_1_matched", "final_masks")
        os.makedirs(output_dir, exist_ok=True)

        sitk.WriteImage(series_1_image, os.path.join(output_dir, "final_mask.nii.gz"))

        output_dir = os.path.join(self.output_dir, "series_2_matched", "final_masks")
        os.makedirs(output_dir, exist_ok=True)

        sitk.WriteImage(series_2_image, os.path.join(output_dir, "final_mask.nii.gz"))

    def _merge_centerlines_contours(self, 
                                    vascular_model: CarotidVascularModel):
        def merge_centerlines_by_closest_ends(A, B):
            options = [
                (A,     B,     np.linalg.norm(A[-1] - B[0])),   # A end -> B start
                (A,     B[::-1], np.linalg.norm(A[-1] - B[-1])), # A end -> B end
                (A[::-1], B,     np.linalg.norm(A[0] - B[0])),   # A start -> B start
                (A[::-1], B[::-1], np.linalg.norm(A[0] - B[-1])) # A start -> B end
            ]

            A_best, B_best, dist = min(options, key=lambda x: x[2])

            connection_point = (A_best[-1] + B_best[0]) / 2

            merged = np.vstack([
                A_best,
                connection_point[None, :],
                B_best
            ])

            return merged, dist

        def merge_contour_stacks_by_centerline_ends(A, B, add_bridge=True):
            def contour_centerline(contours):
                return np.mean(contours, axis=1)

            A = np.asarray(A)
            B = np.asarray(B)

            cl_A = contour_centerline(A)
            cl_B = contour_centerline(B)

            candidates = [
                (A,       B,       cl_A,       cl_B),
                (A,       B[::-1], cl_A,       cl_B[::-1]),
                (A[::-1], B,       cl_A[::-1], cl_B),
                (A[::-1], B[::-1], cl_A[::-1], cl_B[::-1]),
            ]

            best = min(candidates, key=lambda x: np.linalg.norm(x[2][-1] - x[3][0]))

            A_best, B_best, cl_A_best, cl_B_best = best

            connection_dist = np.linalg.norm(cl_A_best[-1] - cl_B_best[0])

            if add_bridge:
                bridge_contour = 0.5 * (A_best[-1] + B_best[0])
                merged = np.concatenate([
                    A_best,
                    bridge_contour[None, :, :],
                    B_best
                ], axis=0)
            else:
                merged = np.concatenate([A_best, B_best], axis=0)

            return merged, connection_dist

        distal_centerline_left = vascular_model.centerlines.get("distal_carotid_artery_left", None)
        distal_centerline_right = vascular_model.centerlines.get("distal_carotid_artery_right", None)

        # One can use torch cluster or scipy.

        if distal_centerline_left is not None:
            
            internal_carotid = vascular_model.centerlines.get("internal_carotid_artery_left", None)
            merged, _= merge_centerlines_by_closest_ends(internal_carotid, distal_centerline_left)

            vascular_model.centerlines.pop("internal_carotid_artery_left") # Removing the one we already had.
            vascular_model.centerlines["internal_carotid_artery_left"] = merged

            # Also affecting the contours.
            internal_carotid_contour = vascular_model.contours.get("internal_carotid_artery_left", None)
            distal_contour = vascular_model.contours.get("distal_carotid_artery_left", None)

            merged_contours, _= merge_contour_stacks_by_centerline_ends(internal_carotid_contour, distal_contour)
            vascular_model.contours["internal_carotid_artery_left"] = merged_contours

        else: # Is None
            print("No left distal carotid artery information was found.")
        

        if distal_centerline_right is not None:

            internal_carotid = vascular_model.centerlines.get("internal_carotid_artery_right", None)
            merged, _= merge_centerlines_by_closest_ends(internal_carotid, distal_centerline_right)

            vascular_model.centerlines.pop("internal_carotid_artery_right") # Updating.
            vascular_model.centerlines["internal_carotid_artery_right"] = merged
            
            # Also affecting the contours.
            internal_carotid_contour = vascular_model.contours.get("internal_carotid_artery_right", None)
            distal_contour = vascular_model.contours.get("distal_carotid_artery_right", None)

            merged_contours, _= merge_contour_stacks_by_centerline_ends(internal_carotid_contour, distal_contour)
            vascular_model.contours["internal_carotid_artery_right"] = merged_contours
        else: # Is None
            print("No right distal carotid artery information was found.")

    # Main matching methods
    def _match_single_side(self,
                           side_name: str,
                           series_1: Dict,
                           series_2: Dict):

        # Matching corrected centerlines
        min_arclength_centerlines = self._compute_min_arclength_centerlines(series_1, series_2)

        masked_centerlines_series_1 = self._mask_centerlines(series_1, min_arclength_centerlines)
        masked_centerlines_series_2 = self._mask_centerlines(series_2, min_arclength_centerlines)

        # Matching contours using reference centerlines
        min_arclength_contours = self._compute_min_arclength_reference_contours(series_1, series_2)
        masked_contours_series_1 = self._mask_contours(series_1, min_arclength_contours)
        masked_contours_series_2 = self._mask_contours(series_2, min_arclength_contours)

        # Merge outputs per series
        matched_series_1 = {
            **masked_centerlines_series_1,
            **masked_contours_series_1,
            # "bifurcation_point": series_1["bifurcation_point"]
        }

        matched_series_2 = {
            **masked_centerlines_series_2,
            **masked_contours_series_2,
            # "bifurcation_point": series_2["bifurcation_point"]
        }

        if not self.split_vessels:

            centerlines_names = ['centerline_internal_carotid_artery',
                                 'centerline_external_carotid_artery']
            contour_names = ['contour_internal_carotid_artery',
                              'contour_external_carotid_artery']

            matched_series_1 = self._stitch_vessels(matched_series_1)
            matched_series_2 = self._stitch_vessels(matched_series_2)

            if "centerline_distal_carotid_artery" in matched_series_1.keys() and "centerline_distal_carotid_artery" in matched_series_1.keys():
                centerlines_names.append("centerline_distal_carotid_artery")
                contour_names.append("contour_distal_carotid_artery")

            # Save
            CarotidVascularModel.save_data(split_dict = matched_series_1,
                                           output_dir = os.path.join(self.output_dir, "series_1_matched"),
                                           prefix = f"",
                                           sufix = f"_{side_name}",
                                           branch_names = centerlines_names,
                                           contour_names = contour_names)

            CarotidVascularModel.save_data(split_dict = matched_series_2,
                                           output_dir = os.path.join(self.output_dir, "series_2_matched"),
                                           prefix = f"",
                                           sufix = f"_{side_name}",
                                           branch_names = centerlines_names,
                                           contour_names = contour_names)
            
            return {
                "min_arclength_centerlines": min_arclength_centerlines,
                "min_arclength_contours": min_arclength_contours,
                "series_1": matched_series_1,
                "series_2": matched_series_2
            }
        
        ##
        CarotidVascularModel.save_data(
            split_dict = matched_series_1,
            output_dir = os.path.join(self.output_dir, "series_1_matched"),
            prefix = f"{side_name}_",
            sufix = f"")

        CarotidVascularModel.save_data(
            split_dict=matched_series_2,
            output_dir=os.path.join(self.output_dir, "series_2_matched"),
            prefix = f"{side_name}_",
            sufix = f"")
        
        return {
            "min_arclength_centerlines": min_arclength_centerlines,
            "min_arclength_contours": min_arclength_contours,
            "series_1": matched_series_1,
            "series_2": matched_series_2
        }
    
    def _stitch_vessels(self,
                        matched_series):
        
        # Only two centerlines instead of 4. This might fail, check the [::-1]
        ICA_and_CCA = np.vstack((matched_series['CCA_from_ICA'][::-1],
                                 matched_series['ICA']))
        
        ECA_and_CCA = np.vstack((matched_series['CCA_from_ECA'][::-1],
                                 matched_series['ECA']))
        
        # Same thing for the contours.
        ICA_and_CCA_contours = np.vstack((matched_series['ICA_contours'],
                                          matched_series['CCA_from_ICA_contours']))
        
        ECA_and_CCA_contours = np.vstack((matched_series['ECA_contours'],
                                          matched_series['CCA_from_ECA_contours']))
        
        matched_series_stitched = {'centerline_internal_carotid_artery' : ICA_and_CCA,
                                   'centerline_external_carotid_artery' : ECA_and_CCA,
                                   'contour_internal_carotid_artery' : ICA_and_CCA_contours,
                                   'contour_external_carotid_artery' : ECA_and_CCA_contours}
        
        try:
            matched_series_stitched["centerline_distal_carotid_artery"] = matched_series["distal_ICA"]
            matched_series_stitched["contour_distal_carotid_artery"] = matched_series["distal_ICA_contours"]
        except:
            print()

        return matched_series_stitched

    # Centerline masking
    def _mask_centerlines(self,
                          series: Dict,
                          min_arclength: Dict):

        masked_centerline = {}
        bifurcation_point = series["bifurcation_point"]

        bifurcation_point_distal_left = series.get('bifurcation_point_distal_left', None)
        bifurcation_point_distal_right = series.get('bifurcation_point_distal_right', None)

        for vessel, data in series.items():
            if vessel.endswith("_contours") or "bifurcation_point" in vessel:
                continue

            data = np.asarray(data)

            if "distal_" in vessel and "_left" in vessel and bifurcation_point_distal_left != None:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point_distal_left)

            elif "distal_" in vessel and "_right" in vessel and bifurcation_point_distal_right != None:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point_distal_right)

            else:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point)

            arclength = self._cumulative_arclength(data)
            mask = arclength <= min_arclength[vessel]

            masked_centerline[vessel] = data[mask]

        return masked_centerline

    # Contour masking, the strategy is different, we use a reference centerline.
    # That is generated same as SIRE, with average of the contours.
    def _mask_contours(self,
                       series: Dict,
                       min_arclength: Dict):

        masked_contours = {}
        bifurcation_point = series["bifurcation_point"]
        bifurcation_point_distal_left = series.get('bifurcation_point_distal_left', None)
        bifurcation_point_distal_right = series.get('bifurcation_point_distal_right', None)

        for vessel, contour in series.items():
            if not vessel.endswith("_contours"):
                continue

            contour = np.asarray(contour)
            # reference centerline has same number of samples as contour rings
            ref_centerline = contour.mean(axis=1)
            # From the ref_centerline we obtain the mask.

            if "distal_" in vessel and "_left" in vessel and bifurcation_point_distal_left != None:
                ref_centerline, flipped = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point_distal_left)

            elif "distal_" in vessel and "_right" in vessel and bifurcation_point_distal_right != None:
                ref_centerline, flipped = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point_distal_right)

            else:
                ref_centerline, flipped = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point)

            if flipped:
                contour = contour[::-1]

            arclength = self._cumulative_arclength(ref_centerline)
            mask = arclength <= min_arclength[vessel]

            masked_contours[vessel] = contour[mask, :, :]

        return masked_contours

    # Arclength computation
    def _compute_min_arclength_centerlines(self,
                                           series_1: Dict,
                                           series_2: Dict):

        arc_1 = self._compute_arclength_centerlines(series_1)
        arc_2 = self._compute_arclength_centerlines(series_2)

        if arc_1.keys() != arc_2.keys():
            raise KeyError("Centerline arclength dictionaries do not have the same keys.")

        return {vessel: min(arc_1[vessel], arc_2[vessel]) for vessel in arc_1.keys()}

    def _compute_min_arclength_reference_contours(self,
                                                  series_1: Dict,
                                                  series_2: Dict):

        arc_1 = self._compute_arclength_reference_contours(series_1)
        arc_2 = self._compute_arclength_reference_contours(series_2)

        if arc_1.keys() != arc_2.keys():
            raise KeyError("Reference contour arclength dictionaries do not have the same keys.")

        return {vessel: min(arc_1[vessel], arc_2[vessel]) for vessel in arc_1.keys()}

    def _compute_arclength_centerlines(self,
                                       vessel_dict: Dict):

        vessel_arclength = {}
        bifurcation_point = vessel_dict["bifurcation_point"]
        bifurcation_point_distal_left = vessel_dict.get('bifurcation_point_distal_left', None)
        bifurcation_point_distal_right = vessel_dict.get('bifurcation_point_distal_right', None)

        for vessel, data in vessel_dict.items():
            if vessel.endswith("_contours") or "bifurcation_point" in vessel:
                continue

            data = np.asarray(data)
            if "distal_" in vessel and "_left" in vessel and bifurcation_point_distal_left != None:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point_distal_left)

            elif "distal_" in vessel and "_right" in vessel and bifurcation_point_distal_right != None:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point_distal_right)

            else:
                data, _ = self._orient_curve_from_bifurcation(data, bifurcation_point)

            arclength = self._cumulative_arclength(data)
            vessel_arclength[vessel] = arclength[-1]

        return vessel_arclength

    def _compute_arclength_reference_contours(self,
                                              vessel_dict: Dict):

        vessel_arclength = {}
        bifurcation_point = vessel_dict["bifurcation_point"]
        bifurcation_point_distal_left = vessel_dict.get('bifurcation_point_distal_left', None)
        bifurcation_point_distal_right = vessel_dict.get('bifurcation_point_distal_right', None)

        for vessel, data in vessel_dict.items():
            if not vessel.endswith("_contours"):
                continue

            contour = np.asarray(data)
            ref_centerline = contour.mean(axis=1)

            if "distal_" in vessel and "_left" in vessel and bifurcation_point_distal_left != None:
                ref_centerline, _ = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point_distal_left)

            elif "distal_" in vessel and "_right" in vessel and bifurcation_point_distal_right != None:
                ref_centerline, _ = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point_distal_right)

            else:
                ref_centerline, _ = self._orient_curve_from_bifurcation(ref_centerline, bifurcation_point)

            arclength = self._cumulative_arclength(ref_centerline)
            vessel_arclength[vessel] = arclength[-1]

        return vessel_arclength

    # Geometry helpers
    def _orient_curve_from_bifurcation(self,
                                       curve: np.ndarray,
                                       bifurcation_point: np.ndarray):
        """
        Orient a curve so that the end closest to the bifurcation becomes the first point.
        Returns:
            oriented_curve, flipped
        """
        curve = np.asarray(curve)
        distances = np.linalg.norm(curve - bifurcation_point, axis=1)
        idx = np.argmin(distances)

        flipped = False
        if idx > len(curve) // 2:
            curve = curve[::-1]
            flipped = True

        return curve, flipped

    def _cumulative_arclength(self,
                              curve: np.ndarray):
        diffs = np.diff(curve, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0.0], np.cumsum(dists)])

    # Build left/right carotid trunks
    def _get_carotid_trunk(self,
                           vascular_model: CarotidVascularModel) -> Dict:

        bifurcation_left, _, _ = vascular_model.detect_bifurcation(
            "external_carotid_artery_left",
            "internal_carotid_artery_left",
            merge_tol=2
        )
        result_left = vascular_model.split_carotid_branches(
            ica_cca="internal_carotid_artery_left",
            eca_cca="external_carotid_artery_left",
            bif_point=bifurcation_left
        )

        bifurcation_right, _, _ = vascular_model.detect_bifurcation(
            "external_carotid_artery_right",
            "internal_carotid_artery_right",
            merge_tol=2
        )
        result_right = vascular_model.split_carotid_branches(
            ica_cca="internal_carotid_artery_right",
            eca_cca="external_carotid_artery_right",
            bif_point=bifurcation_right
        )

        try:
            result_left['distal_ICA'] = vascular_model.centerlines['distal_carotid_artery_left']
            result_left['distal_ICA_contours'] = vascular_model.contours['distal_carotid_artery_left']

            bifurcation_point_distal_left, _, _ = vascular_model.detect_bifurcation('internal_carotid_artery_left',
                                                                                    'distal_carotid_artery_left',
                                                                                    merge_tol = 2)
            result_left['bifurcation_point_distal_left'] = bifurcation_point_distal_left

        except: print()

        return result_left, result_right

    # Save helpers
    def _save_carotid_trunks(self,
                             output_dir: str,
                             left_data: Dict,
                             right_data: Dict):

        CarotidVascularModel.save_data(
            split_dict=left_data,
            output_dir=output_dir,
            prefix="left_"
        )

        CarotidVascularModel.save_data(
            split_dict=right_data,
            output_dir=output_dir,
            prefix="right_"
        )

## <== Utils for mesh reconstruction ==> ##

def plot_contours_3d(contours, every=1, show_points=False, figsize=(8, 8)):

    if contours.ndim != 3 or contours.shape[2] != 3:
        raise ValueError(
            f"Expected contours with shape (N_contours, N_points, 3), "
            f"got {contours.shape}"
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for i in range(0, contours.shape[0], every):
        c = contours[i]

        # Close contour for visualization
        c_closed = np.vstack([c, c[0]])

        ax.plot(
            c_closed[:, 0],
            c_closed[:, 1],
            c_closed[:, 2],
            linewidth=1
        )

        if show_points:
            ax.scatter(
                c[:, 0],
                c[:, 1],
                c[:, 2],
                s=5
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D contours ({contours.shape[0]} total)")

    ax.set_box_aspect([
        np.ptp(contours[:, :, 0]),
        np.ptp(contours[:, :, 1]),
        np.ptp(contours[:, :, 2])
    ])

    plt.tight_layout()
    plt.show()

def load_contours(PATH_contours: str,
                  min_circularity: float = 0.7):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(PATH_contours)
    reader.Update()

    contour = reader.GetOutput()
    points = contour.GetPoints()
    num_points = points.GetNumberOfPoints()
    contour_array = np.zeros((num_points, 3))

    for i in range(num_points):
        contour_array[i, :] = points.GetPoint(i)

    contour_array = contour_array.astype(np.float32)
    contours = contour_array.reshape(-1, 128, 3)
    valid_contours = []

    for contour in contours:
        circ = contour_circularity(contour)

        if circ >= min_circularity:
            valid_contours.append(contour)

    if len(valid_contours) == 0:
        return np.empty((0, 128, 3), dtype=np.float32)

    return np.stack(valid_contours)

def contour_circularity(contour_points):

    pts_2d = PCA(n_components=2).fit_transform(contour_points)
    pol = Polygon(pts_2d)

    if not pol.is_valid:
        pol = pol.buffer(0)

    if pol.length < 1e-8:
        return 0.0

    return (4*np.pi*pol.area)/(pol.length**2)

def load_centerline(PATH_centerline : str):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(PATH_centerline)
    reader.Update()
    centerline = reader.GetOutput()
    points = centerline.GetPoints()
    num_points = points.GetNumberOfPoints()
    centerline_array = np.zeros((num_points, 3))
    for i in range(num_points):
        centerline_array[i, :] = points.GetPoint(i)
    # Cast centerline_array to float32
    centerline_array = centerline_array.astype(np.float32)

    return centerline_array

# Torch-native featurizers (recommended when networks are in Torch)
def featurizer_raw_torch(s_t: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
    """Default features: [s, theta] as Torch (B,2)."""
    return torch.stack([s_t, theta_t], dim=1).to(dtype=torch.float32)

# NumPy featurizers (backward-compatible)
def featurizer_raw(s, theta):
    """Default features: [s, theta] as NumPy."""
    return np.stack([s, theta], axis=-1).astype(np.float32)

class SphericalCoordinates:
    def __init__(self, 
                 contour_array : np.array,
                 max_diameter_threshold : int = 15,
                 max_gap_size : int = 10,
                 device : str = "cpu",
                 dtype: torch.dtype = torch.float32):
        
        self.contour_array = contour_array
        self.max_diameter_threshold = max_diameter_threshold
        self.max_gap_size = max_gap_size
        centroid, masked_contour = self.get_centroid(contour_array)

        self.centroid = centroid
        self.masked_contour = masked_contour

        self.device = device

        self.dtype = dtype
        self.P = torch.as_tensor([contour.mean(axis = 0) for contour in masked_contour], device = device)

    def get_contour_mask(self, contours : np.array):

        diameter = []
        for contour in contours:
            center = np.mean(contour, axis = 0)

            max_diameter = 2 * np.linalg.norm(contour - center, axis = -1).max()
            diameter.append(max_diameter > self.max_diameter_threshold)
        
        diameter_mask = np.array(diameter, dtype = bool)
        diameter_mask = self.fill_small_gaps(diameter_mask)
        # largest_region_mask = self.keep_largest_contiguous_region(diameter_mask)

        selected_region_mask = self.keep_upper_contiguous_region(
            diameter_mask,
            contours,
            min_region_length=8
        )

        return selected_region_mask
    
    def subsample_contours(self, contours, n_points_per_contour=32):

        if n_points_per_contour is None:
            return contours

        n_contours, n_points, _ = contours.shape
        idx = np.linspace(0, n_points - 1,
                          n_points_per_contour,
                          dtype=int)

        return contours[:, idx, :]

    def get_centroid(self, contours : np.array):

        mask = self.get_contour_mask(contours) # Should work as well for the centerline.
        self.mask = mask

        masked_contour = contours[mask == 1]
        
        return masked_contour.reshape(-1, 3).mean(axis = 0), masked_contour
    
    def get_polar_coordinates(self, samples = 32):
        
        aneurysm_points = self.subsample_contours(self.masked_contour, samples)
        aneurysm_points = aneurysm_points.reshape(-1, 3)
        diff = aneurysm_points - self.centroid

        x = diff[:, 0]
        y = diff[:, 1]
        z = diff[:, 2]

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / (r + 1e-8))

        return torch.as_tensor(r, device = self.device), torch.as_tensor(theta, device = self.device), torch.as_tensor(phi, device = self.device)
    
    def fill_small_gaps(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool).copy()
        n = len(mask)

        i = 0
        while i < n:
            if not mask[i]:
                start = i
                while i < n and not mask[i]:
                    i += 1

                end = i
                gap_size = end - start
                left_valid = start > 0 and mask[start - 1]
                right_valid = end < n and mask[end]
                if left_valid and right_valid and gap_size <= self.max_gap_size:
                    mask[start:end] = True

            else:
                i += 1

        return mask
    
    def keep_upper_contiguous_region(self,
                                    mask: np.ndarray,
                                    contours: np.ndarray,
                                    min_region_length: int = 3) -> np.ndarray:
        mask = mask.astype(bool)
        if not mask.any():
            return mask

        # Find contiguous True regions
        padded = np.pad(mask.astype(np.int32), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        candidates = []

        for start, end in zip(starts, ends):
            length = end - start

            if length < min_region_length:
                continue

            region_contours = contours[start:end]

            # Mean z of all points in this connected region
            mean_z = region_contours[:, :, 2].mean()

            candidates.append({
                "start": start,
                "end": end,
                "length": length,
                "mean_z": mean_z
            })

        if len(candidates) == 0:
            return np.zeros_like(mask, dtype=bool)

        # Pick the connected region with highest z
        selected = max(candidates, key=lambda c: c["mean_z"])

        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[selected["start"]:selected["end"]] = True

        return new_mask
     
    def keep_largest_contiguous_region(self, 
                                       mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)

        if not mask.any():
            return mask

        # Find transitions
        padded = np.pad(mask.astype(np.int32), (1, 1), constant_values=0)
        changes = np.diff(padded)

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        lengths = ends - starts
        largest_idx = np.argmax(lengths)

        new_mask = np.zeros_like(mask, dtype=bool)
        new_mask[starts[largest_idx]:ends[largest_idx]] = True

        return new_mask

    def Finv_vec(self, coord): # For only one coordinate.

        diff = coord.cpu().numpy() - self.centroid
        x = diff[:, 0]
        y = diff[:, 1]
        z = diff[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / (r + 1e-8))

        return torch.as_tensor(r, device = self.device), torch.as_tensor(theta, device = self.device), torch.as_tensor(phi, device = self.device)
    
    def evaluate_phi(self,
                     coord : np.array,
                     net : torch.nn.Module,
                     device):
        
        r_t, theta_t, phi_t = self.Finv_vec(coord)

        try:
            feats_t = featurizer_raw_torch(theta_t, phi_t).to(device=device, dtype=torch.float32)  # (B,F)
        except:
            feats_np = featurizer_raw(theta_t.detach().cpu().numpy(), phi_t.detach().cpu().numpy())  # (B,F)
            feats_t = torch.from_numpy(feats_np).to(device = device, dtype=torch.float32)
        out = net(feats_t)
        r_pred_t = out.reshape(-1).float()  # (B,)
        phi_i_t = (r_pred_t - r_t).float()  # (B,)

        return phi_i_t

class Sine(torch.nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class SphericalSIREN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, w0=30.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = 4 if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            linear = torch.nn.Linear(in_dim, out_dim)

            if i == 0:
                torch.nn.init.uniform_(linear.weight, -1/in_dim, 1/in_dim)
            else:
                torch.nn.init.uniform_(linear.weight, -np.sqrt(6/in_dim)/w0, np.sqrt(6/in_dim)/w0)

            layers.append(linear)

            if i < num_layers - 1:
                layers.append(Sine(w0))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, theta_phi):
        
        theta = theta_phi[:, 0:1] # (N, 1)
        phi = theta_phi[:, 1:2]  # (N, 1)

        theta_phi_mapped = torch.cat([torch.sin(theta), torch.cos(theta), 
                                      torch.sin(phi), torch.cos(phi)], dim=1)  # (N, 4)

        return torch.nn.functional.softplus(self.net(theta_phi_mapped)).squeeze(-1)

def crop_centerline_to_contours(
    centerline: np.ndarray,
    contours_selected: np.ndarray,
    margin_points: int = 5):

    contour_centroids = contours_selected.mean(axis=1)  
    distances = np.linalg.norm(
        contour_centroids[:, None, :] - centerline[None, :, :],
        axis=-1
    )  # (N_contours, N_centerline)

    nearest_centerline_idx = np.argmin(distances, axis=1)

    i0 = nearest_centerline_idx.min()
    i1 = nearest_centerline_idx.max()

    i0 = max(0, i0 - margin_points)
    i1 = min(len(centerline) - 1, i1 + margin_points)

    return centerline[i0:i1 + 1] #, (i0, i1)

def fit_siren_to_contour_spherical(sc, device, w0 : int = 1):

    r_gt, theta, phi = sc.get_polar_coordinates()
    theta_phi = featurizer_raw_torch(theta, phi).to(device = device, dtype=torch.float32)

    net = SphericalSIREN(w0 = w0).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)

    losses = []

    for epoch in tqdm.tqdm(range(1000)):
        optimizer.zero_grad()
        # Input to the network is (d, theta) = (s, theta) since s is the independent variable along the centerline and theta is the angular coordinate. The network should learn to predict rho as a function of s and theta.
        r_pred = net(theta_phi)
        loss = torch.nn.functional.mse_loss(r_pred, r_gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return net, losses

def mesh_reconstruction(PATH : str, 
                        vessel_names : List[str],
                        device: torch.device,
                        allow_mixed_reconstruction : bool = False,
                        output_file_name : str = "vessels_fused_final.stl",
                        study : str = 'final',
                        **kwargs) -> trimesh.Trimesh:
    
    # Getting the values from kwargs #
    omega = kwargs.get("siren_omega", 5)
    voxel_size = kwargs.get("voxel_size", 0.2)
    soft_union_tau = kwargs.get("soft_union_tau", 0.25)
    gaussian_sigma = kwargs.get("gaussian_sigma", 2.5)
    batch_size = kwargs.get("batch_size", 300000)

    if study in ['raw', 'correction']:
        contour_dir = os.path.join(PATH, f"contour_{study}","lumen")
        centerline_dir = os.path.join(PATH,f"centerline_{study}")
        output_dir = os.path.join(PATH, f"mesh_{study}")

    else:
        contour_dir = os.path.join(PATH, "final_contours")
        centerline_dir = os.path.join(PATH, "final_centerlines")
        output_dir = os.path.join(PATH, "final_meshes")

    os.makedirs(output_dir, exist_ok = True)

    print(f"Selected device: {device}")
    print(f"Mixed reconstruction allowed: {allow_mixed_reconstruction}")
    print(f"Output folder: {output_dir}")

    vessels = []

    if allow_mixed_reconstruction:
        mixed_vessels = []

        if "distal_carotid_artery_left" in vessel_names:
            mixed_vessels = ["distal_carotid_artery_left",
                            "internal_carotid_artery_left"]
            
            vessel_names.remove("internal_carotid_artery_left")
            vessel_names.remove("distal_carotid_artery_left")
            
        elif "distal_carotid_artery_right" in vessel_names:
            mixed_vessels = ["distal_carotid_artery_right",
                            "internal_carotid_artery_right"]
            
            vessel_names.remove("internal_carotid_artery_right")
            vessel_names.remove("internal_carotid_artery_right")
            
        if len(mixed_vessels) > 0:
            
            for vessel in mixed_vessels:
                contour = load_contours(os.path.join(contour_dir, f"contour_{vessel}.vtp"), min_circularity = 0.1)
                
                sc = SphericalCoordinates(contour_array = contour,
                                          max_diameter_threshold = 20,
                                          max_gap_size = 1,
                                          device = device)
                r_gt, _, _ = sc.get_polar_coordinates()
                trained_model, losses = fit_siren_to_contour_spherical(sc, device, w0 = 1)

                # This are the parts modeled by polar coordinates rather than tubular coordinates.
                c = sc.centroid.reshape(3)
                r = r_gt.cpu().numpy().max() * 1.5

                bounds = (c - r, c + r)
                vessel_config = {'name' : f'aneurysm_{vessel}',
                                'tc' : sc,
                                'net' : trained_model,
                                'device' : device,
                                'bounds' : bounds,
                                'band_radius': r_gt.cpu().numpy().max() * 1.5,
                                'coord_type': 'spherical'}
                vessels.append(vessel_config)
                
                # Now for the part that should be modeled by tubular coordinates.
                if "distal" in vessel:
                    mask = sc.keep_upper_contiguous_region(~sc.mask, sc.contour_array)
                else:
                    mask = sc.keep_largest_contiguous_region(~sc.mask)
                    
                contours_vessel = contour[mask == 1]
                centerline_vessel = np.array([i.mean(axis = 0) for i in contours_vessel])

                plot_contours_3d(contours_vessel)
                
                centerline = load_centerline(os.path.join(centerline_dir, f"centerline_{vessel}.vtp"))
                tc = utils_torch.TubeCoordinates(crop_centerline_to_contours(centerline, contours_vessel), 
                                                 device = device, dtype = torch.float32)

                contour_array = contours_vessel.reshape(-1, 3)
                s, rho, theta = tc.Finv_vec(contour_array)

                rho = rho.cpu().numpy()
                max_radius = np.max(rho)

                trained_model, losses = utils_torch.fit_siren_to_contour(contour_array, tc, device, visualize=False, w0=0.5)

                vessel_config = {'name': 'extra_vessel',
                                'tc': tc,
                                'net': trained_model,
                                'device': device,
                                'bounds': None,  
                                'band_radius': max_radius * 1.5}  # add some margin to the max radius for the band,
                vessels.append(vessel_config)

            print(f"Length of the vessel list so far: {len(vessels)}")

    # Normal reconstruction.
    for vessel in vessel_names:
        try:
            centerline = load_centerline(os.path.join(centerline_dir, f"centerline_{vessel}.vtp"))
            contour_array = load_contours(os.path.join(contour_dir, f"contour_{vessel}.vtp")).reshape(-1, 3)

            tc = utils_torch.TubeCoordinates(centerline, device=device, dtype=torch.float32)
            s, rho, theta = tc.Finv_vec(contour_array) 
            rho = rho.cpu().numpy()
            max_radius = np.max(rho)
        
            trained_model, losses = utils_torch.fit_siren_to_contour(contour_array, 
                                                                     tc, 
                                                                     device, 
                                                                     visualize = False, 
                                                                     w0 = omega)

            vessels.append({
                'name': vessel,
                'tc': tc,
                'net': trained_model,
                'device': device,
                'bounds': None,  
                'band_radius': max_radius * 1.2,  # add some margin to the max radius for the band,
                # 'featurizer_torch': utils_torch.featurizer_raw_torch
            })
        
        except:
            print(f"Information for vessel: {vessel} not found, skipping...")

    mesh, extra = utils_torch.mesh_vessels_implicit_fusion_narrowband(vessels = vessels,
                                                                      voxel_size = voxel_size,
                                                                      margin = 20.0,
                                                                      soft_union_tau = soft_union_tau, # 0.25,  # smooth blending
                                                                      gaussian_sigma = gaussian_sigma,  # optional smoothing
                                                                      batch_size = batch_size,  # number of points to evaluate in parallel (adjust based on GPU memory)
                                                                      export_path = os.path.join(output_dir, output_file_name),
                                                                      verbose = True)
    # mesh.export(os.path.join(output_dir, output_file_name))
    
    return mesh