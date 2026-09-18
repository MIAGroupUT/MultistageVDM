import logging
import os

from dataclasses import asdict
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import torch
import trimesh
import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

from scipy.ndimage import label
from scipy.interpolate import interp1d

from logdecorator import log_on_end, log_on_start
from monai.transforms import Compose, ScaleIntensityRanged
from sklearn.metrics.pairwise import haversine_distances
from tqdm.auto import tqdm

from src.sire.inference.inference_models import SegmentationInferenceModel, TrackerInferenceModel
from src.sire.inference.utils.step_schedulers import ConstantStepScheduler, StepSchedulerBase
from src.sire.inference.utils.correction_methods import ParallelTransportFrame
from src.sire.inference.utils.stopping_criterions import AlreadyTrackedStoppingCriterion, StoppingCriterionBase
from src.sire.inference.utils.tracked_vessel import TrackedVessel, VesselContour
from src.sire.inference.utils.vessel_config import VesselConfig
from src.sire.utils.affine import cart2spher, get_affine

RICH_INFO = 25
logging._levelToName[RICH_INFO] = "RICH_INFO"
logging._nameToLevel["RICH_INFO"] = RICH_INFO


class SegmentatorTrackerPipeline:
    """Tracking and contour regression pipeline for SIRE models.

    Args:
        tracker_model (TrackerInferenceModel): loaded tracker model
        segmentation_models (List[SegmentationInferenceModel]): list of loaded segmentation models
        logging_level (int, optional): at which level the logging should be perform, use default RICH_INFO
                                       to see how the segments are tracked, from which points, where they stop etc.
                                       Defaults to RICH_INFO.
    """

    def __init__(
        self,
        tracker_model: TrackerInferenceModel,
        segmentation_models: List[SegmentationInferenceModel],
        logging_level: int = RICH_INFO,
    ):
        self.logging_level = logging_level
        logging.basicConfig(level=logging_level, format="%(levelname)s (%(asctime)s): %(message)s")

        self.tracker_model = tracker_model
        self.segmentation_models = segmentation_models

        self.pre_transforms = Compose(
            [ScaleIntensityRanged(keys=["image"], 
                                  a_min = -400,
                                  a_max = 800, 
                                  b_min = 0, 
                                  b_max = 1, 
                                  clip = False)]
        )
        self.last_n_directions = []
        self.directions = []

    @log_on_start(RICH_INFO, "Preparing data")
    def _prepare_data(self, image: np.array, affine: np.array, seed_point: np.array):
        """Convert image data to the input dictionary."""
        data = {
            "id": torch.tensor([0]),
            "image": torch.from_numpy(image),
            "image_meta_dict": {"affine": affine},
        }

        data = self.pre_transforms(data)
        data = {
            "tracking": self.tracker_model.transform_input(data),
            "segmentation": [
                segmentation_model.transform_input(data) for segmentation_model in self.segmentation_models
            ],
        }

        return (
            data,
            torch.from_numpy(seed_point).reshape(3),
        )

    def _get_direction(self, heatmap: torch.tensor, prev_direction: torch.tensor):
        """Get tracking direction from the heatmap - with previous direction masking."""
        heatmap = heatmap.clone()
        prev_direction /= torch.linalg.norm(prev_direction)
        prev_dir_spher = cart2spher(prev_direction.reshape(1, -1))[:, 1:] - np.array([np.pi / 2, 0])

        dists = torch.from_numpy(haversine_distances(self.tracker_model.sampler.sphereverts, 
                                                     prev_dir_spher))
        heatmap[:, dists > np.pi / 3] = 0
        ind = torch.argmax(heatmap, dim=1)

        return torch.from_numpy(self.tracker_model.sampler.sphere.cartverts[ind, :])

    def _init_tracking(self, data: Dict[str, Any], point: torch.tensor):
        """Initilize tracking at the point by finding two-side directions."""

        # Call model
        heatmap = self.tracker_model(data["tracking"], point)

        # Get leading direction
        ind_1 = torch.argmax(heatmap, dim=1)
        direction_spher = self.tracker_model.sampler.sphereverts[ind_1, :].view(-1, 2)

        # Get opposite direction
        # We assume that the other direction is on the opposite side.
        dists = torch.from_numpy(haversine_distances(self.tracker_model.sampler.sphereverts, direction_spher))
        heatmap[:, dists < np.pi / 2] = 0
        ind_2 = torch.argmax(heatmap, dim=1)

        directions = torch.from_numpy(
            np.stack([self.tracker_model.sampler.sphere.cartverts[ind, :] for ind in [ind_1, ind_2]])
        )

        return directions # We get the directions in cartesian

    def _stop_tracking(
        self,
        stopping_criterions: List[StoppingCriterionBase],
        iteration: int,
        point: torch.tensor,
        data: Dict[str, Any],
        vessel_contour: VesselContour,
    ) -> bool:
        """Evaluates list of stopping criterions to check whether tracking should terminate."""
        return [
            stopping_criterion(iteration=iteration, point=point, data=data["tracking"], vessel_contour=vessel_contour)
            for stopping_criterion in stopping_criterions
        ]

    @log_on_end(RICH_INFO, "Stopped after {iteration!r} iterations: {result!r}")
    def _check_triggered_criterion(
        self, stopping_criterions: List[StoppingCriterionBase], criterions_state: List[bool], iteration: int
    ):
        """Checks which stopping criterion triggered - for logging sake."""
        return [
            str(stopping_criterion)
            for stopping_criterion, state in zip(stopping_criterions, criterions_state)
            if state is True
        ]

    def _iteration(
        self,
        iteration: int,
        data: Dict[str, Any],
        point: torch.tensor,
        prev_direction: torch.tensor,
        segment_every_n_steps: int,
        average_last_n_directions: int = 5,
    ) -> VesselContour:
        """Run single tracker iteration - get contour and direction at the given point."""

        # Call model and get the heatmap.
        heatmap = self.tracker_model(data["tracking"], point)

        # Until get direction we mask the values.
        direction = self._get_direction(heatmap, prev_direction)
        self.last_n_directions.append(direction)

        if len(self.last_n_directions) > average_last_n_directions:
            self.last_n_directions = self.last_n_directions[1:]

        # Consider directions as mean of last n directions
        direction = torch.stack(self.last_n_directions).mean(dim=0)
        direction /= torch.linalg.norm(direction)

        self.last_n_directions[-1] = direction
        contours = {}

        # Run segmentation for each provided model seperately (if provided and n-step)
        # The model here gives us the contour in polar coordinates, we transform it into cartesian coords and
        # we output the VesselContour, has a center, a normal and the contour itself.

        scale = None
        if self.segmentation_models is not None and iteration % segment_every_n_steps == 0:

            for segmentation_model, data in zip(self.segmentation_models, data["segmentation"]):

                # We save the heatmap here, when returning a vessel contour.
                _, polar_contour, _, scale = segmentation_model(data, point, direction)
                padding = segmentation_model.model.head.padding

                # Split channels to the
                for i, contour_name in enumerate(segmentation_model.names):
                    cartesian_contour = segmentation_model.model.polar_sampler.inverse(
                        polar_contour[:, :, i].unsqueeze(2),
                        point.view(-1, 3),
                        direction.view(-1, 3),
                        scale,
                        padding=padding,
                    ).view(-1, 3)

                    contours[contour_name] = cartesian_contour

        contour = VesselContour(point, direction, contours, heatmap, scale) # Saving also the heatmaps and scales

        return contour

    @log_on_start(RICH_INFO, "Direction: {direction!r}")
    def _track_direction(
        self,
        data: Dict[str, Any],
        seed_point: torch.tensor,
        direction: torch.tensor,
        segment_every_n_steps: int,
        step_scheduler: StepSchedulerBase,
        stopping_criterions: List[StoppingCriterionBase],
    ) -> TrackedVessel:
        iteration = 0
        self.last_n_directions = []

        # Init tracked vessel
        tracked_vessel = TrackedVessel(data["tracking"]["image"], 
                                       data["tracking"]["image_meta_dict"]["affine"])
        vessel_contour = VesselContour(seed_point, direction, None, None)
        point = seed_point

        # Run tracking and segmentation until any of conditions is triggered (at least one iteration)
        while iteration == 0 or not np.any(
            criterions_state := self._stop_tracking(stopping_criterions, iteration, point, data, vessel_contour)
        ):
            # vessel_contour.normal (direction) of the last contour.
            vessel_contour = self._iteration(iteration, data, point, vessel_contour.normal, segment_every_n_steps)

            # We move the tracked point, in the last direction.
            # Then we add the vessel contour to a contours list.
            point = vessel_contour.center + vessel_contour.normal * step_scheduler()
            tracked_vessel.update(vessel_contour)

            iteration += 1

        # A criterion quicked us out of the loop.
        self._check_triggered_criterion(stopping_criterions, criterions_state, iteration)

        # Scrap last if tracking succeeded - lasted more than one iteration
        # We tracked something, but got out of the loop, therefore a criterion was triggered.
        if iteration > 1:
            tracked_vessel.scrap_last()

        return tracked_vessel
    
    @log_on_start(RICH_INFO, "Looking for weird stuff in the vessel, possibly aneurysms")
    def _flag_contours(self,
                       data: Dict[str, Any],
                       already_tracked_vessel: TrackedVessel):
        
        # Returns the same TrackedVessel but flagging the VesselContours.
        proper_contours_idx = [idx for idx, contour in enumerate(already_tracked_vessel.contours) if len(contour.points.keys()) != 0]
        landmark_keys = self.anatomical_landmarks.keys()

        for i in proper_contours_idx: # Iteration around the contours.
            diameter = already_tracked_vessel.contours[i]._estimate_scale().item()

            if diameter >= 7.5: # Shouldn't be this above/around aprox C3 -> label 48
                center = already_tracked_vessel.contours[i].center.cpu().numpy() # Center of the suspicious contour.

                print(center) # Should be a 3D coordinate.
                if center[2] > self.anatomical_landmarks[48]['min_xyz'][2]:
                    print("Above min C3 -> possible aneurysm")

                    already_tracked_vessel.contours[i].flag = True # Flagging the contour.

        # Probably then we should check if there is randomly a non-flagged contour among the flagged ones.

        centroid = np.mean([contour.center.cpu().numpy() for contour in already_tracked_vessel.contours if contour.flag], axis = 0)
        max_diameter = np.max([contour._estimate_scale().item() for contour in already_tracked_vessel.contours if contour.flag])

        # Here we have the icosphere:
        print("First coordinate of the icosphere: {}".format(self.tracker_model.sampler.sphere.cartverts[0]))
        print("Adjusted first coordinate: {}".format(np.array(self.tracker_model.sampler.sphere.cartverts[0]) + centroid))

        print("Centroid: {}".format(centroid))
        print("Max_diameter found: {}".format(max_diameter))

        image_np = data["tracking"]["image"].cpu().numpy()
        affine = data["tracking"]["image_meta_dict"]["affine"]
        if isinstance(affine, torch.Tensor):
            affine = affine.detach().cpu().numpy()

        M = affine[:3, :3]
        origin = affine[:3, 3]

        spacing = np.linalg.norm(M, axis = 0)
        direction = np.zeros_like(M)
        nonzero = spacing > 0
        direction[:, nonzero] = M[:, nonzero]/spacing[np.newaxis, nonzero]

        step = 0.1
        print("Generat the theta and phi...")
        mesh = trimesh.creation.icosphere(subdivisions = 3,
                                          radius = 0.5)
        vertices = mesh.vertices # This has shape 642, 3
        theta = np.arcsin(vertices[:,2])
        phi = np.arctan2(vertices[:,1], vertices[:,0])

        verts = np.array(self.tracker_model.sampler.sphere.sphereverts)
        print("Shape of the spheric vertices: {}".format(np.shape(verts)))
        print("First of the spheric vertices: {}".format(verts[0]))

        gradient_threshold, final_points = 30.0, []  

        for i in range(642):
            prev_intensity = None
            last_valid_point = centroid.copy()

            r = 0.5
            while r < max_diameter:
                point_world = np.array([
                    centroid[0] + r * np.cos(theta[i]) * np.cos(phi[i]),
                    centroid[1] + r * np.cos(theta[i]) * np.sin(phi[i]),
                    centroid[2] + r * np.sin(theta[i])
                ])

                # world -> voxel
                point_voxel = np.linalg.inv(direction) @ (point_world - origin)
                point_voxel = point_voxel / spacing
                point_voxel = np.round(point_voxel).astype(int)

                x, y, z = point_voxel.astype(int)
                intensity = image_np[z, y, x]

                if prev_intensity is not None:
                    grad = intensity - prev_intensity

                    if abs(grad) > gradient_threshold:
                        break

                prev_intensity = intensity
                last_valid_point = point_world.copy()
                r += step

            final_points.append(last_valid_point)

        # np.save("valid_points.npy", final_points)
                        
        # ---> This should be other function <---
        # After that, we get all the centers of the flagged contours and find the centroid of those points.
        # We get all the points and also find which contour had the biggest/largest diameter.
        # That will be the seed.
        # Then we project all the points of the icosphere -> looking for a big gradient change
        # The projection shouldn't be larger than the diameter, or the radius?

    @log_on_start(RICH_INFO, "Vessel correction")
    def _vessel_correction(self,
                           data: Dict[str, Any],
                           already_tracked_vessel: TrackedVessel,
                           stopping_criterions : List[StoppingCriterionBase],
                           centerline_resampling: float = -1,
                           savgol_window_length: int = 151, # Change for preset.
                           segment_every_n_steps: int = 5) -> TrackedVessel:
        
        corrected_vessel = TrackedVessel(data["tracking"]["image"], 
                                         data["tracking"]["image_meta_dict"]["affine"]) # Corrected vessel initialization.
        centers = already_tracked_vessel.get_centers()

        if centerline_resampling > -1:
            centers = self._centerline_resampling(centerline_resampling, centers)
        
        ptf = ParallelTransportFrame(centers = centers,
                                     filter_fn = ParallelTransportFrame.savgol(window_length = savgol_window_length))
                                     # output_filter_fn = ParallelTransportFrame.savgol(window_length = 51))
        T, _, _, filtered_centers = ptf()
        
        for i in range(0, len(filtered_centers)):

            scale, heatmap, contours = None, None, {}
            center = torch.tensor(filtered_centers[i], dtype=torch.float32)
            direction = torch.tensor(T[i], dtype = torch.float32)

            if self.segmentation_models is not None and i % segment_every_n_steps == 0:
                for segmentation_model, seg_data in zip(self.segmentation_models, data["segmentation"]):

                    _, polar_contour, _, scale = segmentation_model(seg_data, center, direction)
                    padding = segmentation_model.model.head.padding

                    # Split channels to the
                    for j, contour_name in enumerate(segmentation_model.names):
                        cartesian_contour = segmentation_model.model.polar_sampler.inverse(
                            polar_contour[:, :, j].unsqueeze(2),
                            center.view(-1, 3),
                            direction.view(-1, 3),
                            scale,
                            padding=padding,
                        ).view(-1, 3)

                        contours[contour_name] = cartesian_contour # Saves "lumen" contour.

            vessel_contour = VesselContour(center, direction, contours, heatmap, scale)

            if not np.any(self._stop_tracking(stopping_criterions, i, center, data, vessel_contour)):
                corrected_vessel.update(vessel_contour)

        return corrected_vessel
    
    def _centerline_resampling(self,
                               resampling_distance: float,
                               centers_tracked_vessel: np.array) -> np.array:
    
        num_centers = len(centers_tracked_vessel) 

        distances = np.zeros(num_centers)
        for i in range(1, num_centers):
            distances[i] = distances[i-1] + np.linalg.norm(centers_tracked_vessel[i] - centers_tracked_vessel[i-1])

        distances_new = np.arange(0, distances[-1], resampling_distance)

        fx = interp1d(distances, centers_tracked_vessel[:,0], kind='linear')
        fy = interp1d(distances, centers_tracked_vessel[:,1], kind='linear')
        fz = interp1d(distances, centers_tracked_vessel[:,2], kind='linear')

        centers_resampled = np.vstack([fx(distances_new), 
                                       fy(distances_new), 
                                       fz(distances_new)]).T # Getting the new centers.
        
        return centers_resampled

    def _prune_contours(self,
                        already_tracked_vessel: TrackedVessel) -> TrackedVessel:
        
        proper_contours_idx = [idx for idx, contour in enumerate(already_tracked_vessel.contours) if len(contour.points.keys()) != 0]
        
        if already_tracked_vessel.contours[0].center[2].item() > already_tracked_vessel.contours[-1].center[2].item():
            proper_contours_idx = proper_contours_idx[::-1]

        to_prune = []
        for idx in range(0, len(proper_contours_idx) - 1):

            current = already_tracked_vessel.contours[proper_contours_idx[idx]]
            next_contour = already_tracked_vessel.contours[proper_contours_idx[idx + 1]]

            center = current.center.cpu().numpy()
            direction = current.normal.cpu().numpy()
            points = next_contour.points['lumen'].cpu().numpy()

            # To check for the direction.
            travel = next_contour.center.cpu().numpy() - current.center.cpu().numpy() 
            direction /= np.linalg.norm(direction)

            if np.dot(travel, direction) < 0: # Wrong direction
                direction = -direction

            d = np.dot(points - center, direction)
            percentage = np.sum(d < -0.3)/len(d)

            if percentage > 0.2:
                to_prune.append(proper_contours_idx[idx + 1])

        for idx in to_prune:
            already_tracked_vessel.contours[idx].points = {}

    @log_on_start(RICH_INFO, "Seed_point: {seed_point!r}")
    def _track_from_point(
        self,
        data: Dict[str, Any],
        seed_point: torch.tensor,
        segment_every_n_steps: int,
        step_scheduler: StepSchedulerBase,
        stopping_criterions: List[StoppingCriterionBase],
        centerline_resampling: float,
        vessel_correction: bool
    ) -> TrackedVessel | List[TrackedVessel]:
        """Perform tracking from the given seed point - both directions."""
        directions = self._init_tracking(data, seed_point)

        assert len(directions) == 2

        forward_track = self._track_direction(
            data, seed_point, directions[0], segment_every_n_steps, step_scheduler, stopping_criterions
        )

        backward_track = self._track_direction(
            data, seed_point, directions[1], segment_every_n_steps, step_scheduler, stopping_criterions
        )        
        forward_track.merge_at_start(backward_track)

        # By this point we already have the centers and the contours of the vessel

        # self._flag_contours(data, forward_track)

        if vessel_correction:   
            corrected_vessel = self._vessel_correction(data, forward_track, stopping_criterions, centerline_resampling)
            self._prune_contours(corrected_vessel)
            return [forward_track, corrected_vessel]

        return forward_track

    @log_on_end(RICH_INFO, "Finished")
    def run_single(
        self,
        image: np.array,
        affine: np.array,
        seed_point: np.array,
        segment_every_n_steps: int = 1,
        step_scheduler: StepSchedulerBase = ConstantStepScheduler(1),
        stopping_criterions: Tuple[StoppingCriterionBase] = (),
        centerline_resampling: float = -1,
        vessel_correction: bool = False,
        **kwargs,
    ) -> TrackedVessel | List[TrackedVessel]:
        """Run pipeline for single segment.

        Args:
            image (np.array): loaded image in numpy
            affine (np.array): loaded image affine matrice in numpy
            seed_point (np.array): 3D seed point where the tracking should start
            segment_every_n_steps (int, optional): every how many steps should the contour be delineated. Defaults to 1.
            step_scheduler (StepSchedulerBase, optional): how to schedule tracker step size. Defaults to ConstantStepScheduler(1).
            stopping_criterions (Tuple[StoppingCriterionBase], optional): stopping criterions for the tracking. Defaults to ().

        Returns:
            TrackedVessel: tracked vessel segment
        """

        data, seed_point = self._prepare_data(image, affine, seed_point)
        tracked_vessel = self._track_from_point(
            data,
            seed_point,
            segment_every_n_steps,
            step_scheduler,
            stopping_criterions,
            centerline_resampling,
            vessel_correction
        )
        return tracked_vessel

    @log_on_start(RICH_INFO, "Pipeline started")
    @log_on_end(RICH_INFO, "Pipeline finished")
    def run(
        self,
        image: Union[str, np.array],
        affine: np.array = None,
        vessel_configs: List[VesselConfig] = None,
        anatomical_landmarks : dict = {},
        already_tracked_distance: float = 0,
        output_dir: str = None,
        save_planar_projections : bool = False,
        save_diameters: bool = False
    ) -> Dict[str, Dict[str, pv.PolyData]]:
        """Run pipeline for multiple segments on one image.
        Single segments are provided as VesselConfig objects.

        Args:
            image (Union[str, np.array]): loaded image in numpy or path to load an image from
            affine (np.array, optional): if image provided in numpy, the affine matrice needs to be provided as well. Defaults to None.
            vessel_configs (List[VesselConfig], optional): single segment configurations for tracking. Defaults to None.
            already_tracked_distance (float, optional): specify whether tracking should stop
                                                        if it was already tracked by other segment. Defaults to 0.
            output_dir (str, optional): path to output directory, if None then not saving. Defaults to None.

        Returns:
            Dict[str, Dict[str, pv.PolyData]]: dictionary of tracked centerlines and contours for all provided segments
        """
        # Load from path if str given
        if isinstance(image, str):
            itk_image = sitk.ReadImage(image)
            affine, _ = get_affine(itk_image)
            image = sitk.GetArrayFromImage(itk_image)

        # Verify if we have a mask to blackout some given structures, like the jugulars.
        # This is completely optional, but encouraged in some cases.
        vessel_path = os.path.join(output_dir, "totalsegmentator", "mask_vessels.nii.gz")

        if os.path.isfile(vessel_path):
            labels = [11, 12]
            mask_vessels_image = sitk.ReadImage(vessel_path)
            mask_vessels_array = sitk.GetArrayFromImage(mask_vessels_image)

            for l in labels:
                jugular_mask = mask_vessels_array == l
                _, num_objects = label(jugular_mask.astype(np.int32))

                if num_objects > 1:
                    continue

                image[jugular_mask] = image[jugular_mask] // 2

            modified_itk = sitk.GetImageFromArray(image)
            modified_itk.CopyInformation(itk_image)

            sitk.WriteImage(modified_itk, os.path.join(output_dir, "totalsegmentator", "raw_modified.nii.gz"))

        # Save seed point if output dir given
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            points = [vc.seed_point.reshape(3) for vc in vessel_configs]
            names = [vc.name for vc in vessel_configs]

            pd.DataFrame.from_records(points, index=names).to_csv(os.path.join(output_dir, "seeds.csv"))

        # Run pipeline for each vessel configuration provided
        results = {}
        already_tracked_stopping = AlreadyTrackedStoppingCriterion(None, already_tracked_distance)

        if len(anatomical_landmarks.keys()) != 0:
            self.anatomical_landmarks = anatomical_landmarks

        with tqdm(total=len(vessel_configs), disable=self.logging_level > RICH_INFO) as pbar:
            for vessel_config in vessel_configs:
                name = vessel_config.name

                pbar.set_description(name)
                vessel_config.stopping_criterions.append(already_tracked_stopping)
                tracked_vessel: TrackedVessel = self.run_single(image=image, affine=affine, **asdict(vessel_config))

                if not isinstance(tracked_vessel, list):
                    tracked_vessel = [tracked_vessel]

                assert all(isinstance(x, TrackedVessel) for x in tracked_vessel) # Verifying that everything in this list is TrackedVessel.

                tracking_type = ['_raw']
                if len(tracked_vessel) == 2:
                    tracking_type.append('_correction')
                
                for info, vessel in zip(tracking_type, tracked_vessel):
                    centerline = vessel.build_centerline()
                    results[name] = {
                        "centerline": centerline,
                        "contour": vessel.build_contours()
                    }

                    if output_dir is not None:
                        # Save centerline
                        os.makedirs(os.path.join(output_dir, "centerline{}".format(info)), exist_ok=True)
                        results[name]["centerline"].save(os.path.join(output_dir, "centerline{}".format(info), f"centerline_{name}.vtp"))

                        # Save contours
                        for contour_name, contour in results[name]["contour"].items():
                            os.makedirs(os.path.join(output_dir, "contour{}".format(info), contour_name), exist_ok=True)
                            contour.save(os.path.join(output_dir, "contour{}".format(info), contour_name, f"contour_{name}.vtp"))

                        # Save polar projections
                        if save_planar_projections:
                            os.makedirs(os.path.join(output_dir, "planar_projections{}".format(info), name), exist_ok=True)
                            vessel.save_planar_projections(os.path.join(output_dir, "planar_projections{}".format(info), name))

                        # Save the diameters
                        if save_diameters:
                            os.makedirs(os.path.join(output_dir, "diameters{}".format(info), name), exist_ok = True)
                            vessel.save_diameters(os.path.join(output_dir, "diameters{}".format(info), name))

                    already_tracked_stopping.update_points(torch.tensor(results[name]["centerline"].points))
                    pbar.update()

        return results
