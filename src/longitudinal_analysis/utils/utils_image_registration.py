import os
import vtk
import glob
import trimesh
import numpy as np
import SimpleITK as sitk


from typing import List, Dict, Optional

from skimage.measure import regionprops
from vtk.util.numpy_support import vtk_to_numpy
from skimage.morphology import binary_dilation, binary_erosion

def world2voxel(world_point : np.array, 
                image: sitk.Image) -> np.array:
    
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3,3)

    voxel = np.linalg.inv(direction) @ (world_point - origin).T
    voxel = (voxel.T / spacing)
    voxel = np.round(voxel).astype(int)

    return voxel

class Roi:
    """Creates region of interest based on totalsegmentator mask and given conditions.

    Args:
        labels (List[int]): TotalSegmentator labels to be included in binary mask creation
        mode (str): how to define ROI based on binary mask: "inside", "outside", "above", "below", "left", "right", "front", "back"
        anchor (str, optional): modes different than "inside" and "outside" require anchoring to specify in respect to
                                which part of mask "max" or "min" the ROI should be placed. Defaults to None.
        dilate (int, optional): dilation rate of the binary mask, if negative perform erosion. Defaults to 0.
    """

    def __init__(self, labels: List[int], mode: str, anchor: str = None, dilate: int = 0):
        self.labels = labels
        self.mode = mode
        self.anchor = anchor
        self.dilate = dilate

    def _get_slice_mask(self, label_mask: np.array, axis: int):
        label_bbox = regionprops(label_mask.astype(int))[0].bbox
        min_idx, max_idx = label_bbox[axis], label_bbox[axis + 3]

        if self.anchor == "max":
            return max_idx

        elif self.anchor == "min":
            return min_idx

        else:
            raise NotImplementedError(f"Anchor '{self.anchor}' is not supported.")

    def __call__(self, mask: np.array):
        roi = np.zeros_like(mask).astype(bool)
        label_mask = np.sum(np.stack([(mask == L) for L in self.labels]), axis=0).astype(bool)

        if self.mode == "inside":
            roi = label_mask

        elif self.mode == "outside":
            roi = ~label_mask

        elif self.mode == "above":
            slice_idx = self._get_slice_mask(label_mask, 0)
            roi[slice_idx:] = True

        elif self.mode == "below":
            slice_idx = self._get_slice_mask(label_mask, 0)
            roi[:slice_idx] = True

        elif self.mode == "front":
            slice_idx = self._get_slice_mask(label_mask, 1)
            roi[:, slice_idx:] = True

        elif self.mode == "back":
            slice_idx = self._get_slice_mask(label_mask, 1)
            roi[:, :slice_idx] = True

        elif self.mode == "left":
            slice_idx = self._get_slice_mask(label_mask, 2)
            roi[:, :, slice_idx:] = True

        elif self.mode == "right":
            slice_idx = self._get_slice_mask(label_mask, 2)
            roi[:, :, :slice_idx] = True

        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not supported.")

        # Dilate or erode the mask
        if self.dilate >= 0:
            func = binary_dilation
        else:
            func = binary_erosion

        for _ in range(self.dilate):
            roi = func(roi)

        return roi

    @staticmethod
    def union(mask: np.array, roi_list):
        return np.sum(np.stack([roi(mask) for roi in roi_list]), axis=0) > 0

    @staticmethod
    def intersection(mask: np.array, roi_list):
        result = roi_list[0](mask).astype(bool, copy=True)
        for roi in roi_list[1:]:
            np.logical_and(result, roi(mask), out=result)
        return result

class PatientRegistration:

    """
    Idea: The image registration will be perfomed in two steps, first using a general mask that goes around
    the neck, let's say around C1-T2, then we use the centerlines of the experiment to make a finer registration
    using then as a tube-like mask.
    """

    def __init__(self,
                 PATH_input : str,
                 patient : str,
                 name_fixed_study : str = None,
                 name_moving_study : str = None,
                 multi_stage_registration: bool = False,
                 refinement_side = "left",
                 name_raw_file = "raw.nii.gz",
                 name_mask_file = "mask_normal.nii.gz",
                 quiet = False):
        
        # Specify name of the folder of the studies (preferable)
        self.name_fixed_study = name_fixed_study
        self.name_moving_study = name_moving_study
        
        self.PATH_patient = PATH_input
        self.PATH_output = os.path.join(PATH_input, patient, "registration")

        self.refinement_side = refinement_side

        self.name_raw_file = name_raw_file
        self.name_mask_file = name_mask_file

        self.multi_stage = multi_stage_registration # Registration in two steps.

        print("\nReading the data\n")
        self._read_data()
        roi_fixed = self._roi_definition(self.mask_fixed)
        roi_moving = self._roi_definition(self.mask_moving)

        print("\nCreating ROI, hang in there! ;)\n")
        roi_fixed_intersection = Roi.intersection(mask = self.mask_fixed,
                                                  roi_list = roi_fixed)
        
        roi_moving_intersection = Roi.intersection(mask = self.mask_moving,
                                                   roi_list = roi_moving)
        
        # Regions of interest as masks for the registration method.
        self.fixed_mask_sitk = sitk.GetImageFromArray(roi_fixed_intersection.astype(np.int32))
        self.fixed_mask_sitk.CopyInformation(self.fixed)

        self.moving_mask_sitk = sitk.GetImageFromArray(roi_moving_intersection.astype(np.int32))
        self.moving_mask_sitk.CopyInformation(self.moving)

        # Saving the ROIs
        os.makedirs(os.path.join(self.PATH_output, "roi"), exist_ok=True)
        sitk.WriteImage(self.moving_mask_sitk, os.path.join(self.PATH_output, "roi", "moving_roi_sitk.nii.gz"))
        sitk.WriteImage(self.fixed_mask_sitk, os.path.join(self.PATH_output, "roi", "fixed_roi_sitk.nii.gz"))

        print("\nSetting registration method\n")
        self._set_registration_method(quiet)
        print("\nExecuting registration\n")
        self.transforms = self._execute_registration(quiet)

        assert self.transforms is not None # Checking if we actually got a self.final_transform.
        print("\nDone! Let's hope that worked\n")

    def _read_data(self):

        try:
            studies = sorted([study for study in os.listdir(self.PATH_patient) if not study.endswith(".DS_store") and not study.endswith("registration")])

            if len(studies) > 2 and self.name_fixed_study is None and self.name_moving_study is None : # More than two studies, we need to specify which one is gonna be used.
                raise ValueError("More than 2 folders are in the specified PATH, please provided full PATH for moving and fixed study.")
            
            # Just grabbing the first two which is probably not ideal.
            if self.name_fixed_study is None and self.name_moving_study is None:
                self.name_fixed_study = studies[0]
                self.name_moving_study = studies[1]

            PATH_fixed = os.path.join(self.PATH_patient, self.name_fixed_study, "totalsegmentator")
            PATH_moving = os.path.join(self.PATH_patient, self.name_moving_study, "totalsegmentator")

            print(f"PATH fixed study: {os.path.join(PATH_fixed, self.name_raw_file)}")
            print(f"PATH moving study: {os.path.join(PATH_moving, self.name_raw_file)}")

            # The data we need is in the totalsegmentator folder -> we need the raw image + mask.nii.gz
            image_fixed = sitk.ReadImage(os.path.join(PATH_fixed, self.name_raw_file))
            image_moving = sitk.ReadImage(os.path.join(PATH_moving, self.name_raw_file))

            self.fixed = sitk.Cast(image_fixed, sitk.sitkFloat32)
            self.moving = sitk.Cast(image_moving, sitk.sitkFloat32)

            # Mask fixed.
            mask_image_fixed = sitk.ReadImage(os.path.join(PATH_fixed, self.name_mask_file))
            mask_image_moving = sitk.ReadImage(os.path.join(PATH_moving, self.name_mask_file))

            self.mask_fixed = sitk.GetArrayFromImage(mask_image_fixed)
            self.mask_moving = sitk.GetArrayFromImage(mask_image_moving)

        except:
            print("There was an error reading the data from the PATH: {}".format(self.PATH_patient))

        if self.multi_stage:

            path_centerline_fixed = os.path.join(self.PATH_patient, self.name_fixed_study, "centerline_correction")
            path_centerline_moving = os.path.join(self.PATH_patient, self.name_moving_study, "centerline_correction")

            if os.path.isdir(path_centerline_fixed) and os.path.isdir(path_centerline_moving):
                
                self.fixed_vessel_mask_sitk = self._read_centerlines(path_centerline_fixed, image_fixed) # Cast the .vtp to an array.
                self.moving_vessel_mask_sitk = self._read_centerlines(path_centerline_moving, image_moving)

                os.makedirs(os.path.join(self.PATH_output, "roi"), exist_ok=True)
                sitk.WriteImage(self.moving_vessel_mask_sitk, os.path.join(self.PATH_output, "roi", "moving_refine_roi_sitk.nii.gz"))
                sitk.WriteImage(self.fixed_vessel_mask_sitk, os.path.join(self.PATH_output, "roi", "fixed_refine_roi_sitk.nii.gz"))

    def _read_centerlines(self,
                          PATH : str,
                          image : sitk.Image,
                          dilation = np.array([8.0, 8.0, 8.0])) -> sitk.Image:
        
        list_vessels = [file for file in os.listdir(PATH) if file.endswith('.vtp')]
        image_array = sitk.GetArrayFromImage(image)
        aux_array = np.zeros_like(image_array)

        if self.refinement_side == "left" or self.refinement_side == "right":

            print(f"Side {self.refinement_side} was provided to perform the refinement registration.")

            aux = list_vessels.copy()
            list_vessels = [file for file in aux if self.refinement_side in file]

        for vessel in list_vessels:
            aux_centerline_array = np.zeros_like(image_array)
            PATH_vessel = os.path.join(PATH, vessel)

            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(PATH_vessel)
            reader.Update()

            centerline = reader.GetOutput()
            points = centerline.GetPoints()

            vtk_array = points.GetData()
            numpy_world_points = vtk_to_numpy(vtk_array) # At this point we have world 3d points.

            voxel_points = world2voxel(numpy_world_points, image)
            z, y, x = voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]
            aux_centerline_array[x, y, z] = 1 # I think world2voxel is causing this change.

            aux_array += aux_centerline_array

        aux_image = sitk.GetImageFromArray(aux_array)
        aux_image.CopyInformation(image)

        bin_filter = sitk.BinaryThresholdImageFilter()
        bin_filter.SetLowerThreshold(1)
        bin_filter.SetUpperThreshold(255)
        binary_image = bin_filter.Execute(aux_image)

        spacing = np.array(binary_image.GetSpacing())

        radius_mm = dilation
        radius_vox = np.maximum(np.round(radius_mm / spacing).astype(int), 1)

        dilated_image = sitk.BinaryDilate(binary_image, radius_vox.tolist())

        return dilated_image

    def _execute_registration(self, quiet = False):

        transforms = {}
  
        global_transform = self.registration_method.Execute(self.fixed, self.moving)
        global_transform_copy = sitk.Transform(global_transform)

        if not quiet:
            print("Final metric value:", self.registration_method.GetMetricValue())
            print("Optimizer stop condition:", self.registration_method.GetOptimizerStopConditionDescription())

        if self.multi_stage:
            self._set_refine_registration_method(
                initial_transform=sitk.Transform(global_transform),
                quiet=quiet
            )
            refine_transform = self.registration_method_refine.Execute(self.fixed, self.moving)

            transforms["global_transform"] = global_transform_copy
            transforms["final_transform"] = refine_transform

        return transforms

    def _set_registration_method(self,
                                 quiet = False):

        initial_transform = sitk.CenteredTransformInitializer(self.fixed,
                                                              self.moving,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        # GEOMETRY alligns the center of the images, is only comparable if the FOV are mostly the same.
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        registration_method.SetMetricAsCorrelation() # Sensible to changes of the HU as long as the changes are similar.

        # Interpolation.
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer.
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,
                                                                     minStep=1e-5,
                                                                     numberOfIterations=200,
                                                                     gradientMagnitudeTolerance=1e-8)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        registration_method.SetShrinkFactorsPerLevel([6, 3, 1]) # 1/6 maybe to orient globally.
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0]) # -> higher if we have more noise.
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Set initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        if hasattr(self, "fixed_mask_sitk") and self.fixed_mask_sitk is not None:
            print("Using mask for the fixed image")
            registration_method.SetMetricFixedMask(self.fixed_mask_sitk)

        if hasattr(self, "moving_mask_sitk") and self.moving_mask_sitk is not None:
            print("Using mask for the moving image")
            registration_method.SetMetricMovingMask(self.moving_mask_sitk)

        if not quiet:
            def command_iteration():
                print(
                    f"Level: {registration_method.GetCurrentLevel()} | "
                    f"Iteration: {registration_method.GetOptimizerIteration()} | "
                    f"Metric: {registration_method.GetMetricValue()}"
                )

            registration_method.AddCommand(sitk.sitkIterationEvent, command_iteration)

        self.registration_method = registration_method

    def _set_refine_registration_method(self,
                                        initial_transform,
                                        quiet = False):

        registration_method = sitk.ImageRegistrationMethod()

        # Appareantly for CT-CT, MeanSquares is a strong baseline.
        # For instance wouldn't work trying to registrate MRI-CT
        registration_method.SetMetricSamplingStrategy(registration_method.NONE) # Using all the points since we are using masks.
        registration_method.SetMetricAsCorrelation()

        registration_method.SetInterpolator(sitk.sitkLinear)

        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,
                                                                     minStep=1e-5,
                                                                     numberOfIterations=200,
                                                                     gradientMagnitudeTolerance=1e-8)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform, inPlace=True) # Then is not a composite transform

        if hasattr(self, "fixed_vessel_mask_sitk") and self.fixed_vessel_mask_sitk is not None:
            registration_method.SetMetricFixedMask(self.fixed_vessel_mask_sitk)

        if hasattr(self, "moving_vessel_mask_sitk") and self.moving_vessel_mask_sitk is not None:
            registration_method.SetMetricMovingMask(self.moving_vessel_mask_sitk)

        if not quiet:
            def command_iteration():
                print(
                    f"[REFINE] Level: {registration_method.GetCurrentLevel()} | "
                    f"Iteration: {registration_method.GetOptimizerIteration()} | "
                    f"Metric: {registration_method.GetMetricValue()}"
                )

            registration_method.AddCommand(sitk.sitkIterationEvent, command_iteration)

        self.registration_method_refine = registration_method

    def _roi_definition(self, mask) -> Roi:

        '''
        Definition of the first step of the image registration, this is a mask around the neck, it
        is around C1 until T2, to the sides is cropped by the skull. Futhermore, it uses the trachea as well
        to eliminate some regions of the face.
        '''

        # Same as in SIRE, maybe we can make this in a more intelligent way.
        roi = []
        labels = {"SKULL" : 91,
                  "C1" : 50,
                  "C2" : 49,
                  "C7" : 44,
                  "C1_C6_T2" : [50, 45, 44, 43, 42],
                  "T2" : 42,
                  "TRACHEA" : 16}
        
        if np.any(mask == labels["C1"]): # Just above max point of c1
            roi.append(Roi([labels["C1"]], "below", "max"))

        elif np.any(mask == labels["C2"]): 
            roi.append(Roi([labels["C2"]], "below", "max"))

        if np.any(mask == labels["SKULL"]):
            roi.append(Roi([labels["SKULL"]], "right", "max")) # Cropping to the sides.
            roi.append(Roi([labels["SKULL"]], "left", "min"))

            roi.append(Roi([labels["SKULL"]], "front", "min"))
            roi.append(Roi([labels["SKULL"]], "outside")) # Sometimes sticks to the mandible.

        if np.any(mask == labels["TRACHEA"]):
            roi.append(Roi([labels["TRACHEA"]], "front", "min", dilate = 30))

        else:
            roi.append(Roi([labels["SKULL"]], "front", "min"))
            roi.append(Roi([labels["SKULL"]], "back", "max"))
                
        vertebrae_found = [np.any(mask == l) for l in labels["C1_C6_T2" ]] # We check the mask.
        roi_vertebrae = [l for v, l in zip(vertebrae_found, labels["C1_C6_T2" ]) if v] # We only keep the found ones.
        roi.append(Roi(roi_vertebrae, "outside"))

        if np.any(mask == labels["T2"]):
            roi.append(Roi([labels["T2"]], "above", "min"))
            roi.append(Roi([labels["T2"]], "back", "max"))
            
        return roi

    def save_registered_data(self,
                             registration_type : str,
                             save_fixed_data : bool = True,
                             register_masks : bool = True,
                             register_meshes : bool = True):
        
        moving_series = os.path.basename(self.name_moving_study)
        fixed_series = os.path.basename(self.name_fixed_study)

        os.makedirs(self.PATH_output, exist_ok = True) # Makes the folder /registration/
        # Making the first part of the registration.
        aux_path = os.path.join(self.PATH_output, 
                                f"{moving_series}__to__{fixed_series}")
        
        # Creating necessary folders.
        os.makedirs(aux_path, exist_ok = True) 
        os.makedirs(os.path.join(aux_path, "transforms"), exist_ok = True)
        os.makedirs(os.path.join(aux_path, "registered_volumes"), exist_ok = True)
        os.makedirs(os.path.join(aux_path, "registered_mesh_correction"), exist_ok = True)

        for transform_name, transform_obj in self.transforms.items():
            if registration_type == "multistage":
                image_transform = transform_obj
                mesh_transform = transform_obj

            elif registration_type == "polyrigid":
                image_transform = transform_obj
                mesh_transform = transform_obj

            else:
                raise ValueError(f"Unknown registration_type: {registration_type}")

            registered_image = sitk.Resample(self.moving,
                                             self.fixed,
                                             image_transform,
                                             sitk.sitkLinear,
                                             0.0,
                                             self.moving.GetPixelID())

            sitk.WriteTransform(image_transform, os.path.join(aux_path, "transforms", f"{transform_name}.tfm"))
            sitk.WriteImage(registered_image, os.path.join(aux_path, "registered_volumes", f"registered_{transform_name}.nii.gz"))

            if register_masks:
                
                PATH_moving = os.path.join(self.PATH_patient, self.name_moving_study, "totalsegmentator", "mask_normal.nii.gz")
                mask_moving = sitk.ReadImage(PATH_moving)

                PATH_fixed = os.path.join(self.PATH_patient, self.name_fixed_study, "totalsegmentator", "mask_normal.nii.gz")
                mask_fixed = sitk.ReadImage(PATH_fixed)

                registered_mask = sitk.Resample(mask_moving,
                                                mask_fixed,
                                                image_transform,
                                                sitk.sitkNearestNeighbor,
                                                0.0,
                                                mask_moving.GetPixelID())
                
                sitk.WriteImage(registered_mask, os.path.join(aux_path, "registered_volumes", f"registered_mask_{transform_name}.nii.gz"))

        # Saving also the fixed, we don't need the for cycle for that.
        if save_fixed_data:
            sitk.WriteImage(self.fixed, os.path.join(aux_path, "registered_volumes", f"fixed.nii.gz"))
            sitk.WriteImage(mask_fixed, os.path.join(aux_path, "registered_volumes", f"mask_fixed.nii.gz"))

        if register_meshes:
            MeshRegistration._register_mesh(
                PATH_input=self.get_PATH_moving_image(),
                PATH_output=os.path.join(aux_path, "registered_mesh_correction"),
                sitk_transform=mesh_transform,
                file_name=f"complete_mesh_{transform_name}.stl")

        # Remember to save the transformation data as well.
        print("\nData successfully saved in PATH: {}\n".format(self.PATH_output))

    # Some getters
    def get_transform(self):
        if self.transforms is None:
            print("The transform wasn't computed, check the path of the provided images.")
            return None
        else:
            return self.transforms

    def get_PATH_moving_image(self):
        return os.path.join(self.PATH_patient, self.name_moving_study) 

    def get_PATH_fixed_image(self):
        return os.path.join(self.PATH_patient, self.name_fixed_study) 

class PolyRigidPatientRegistration(PatientRegistration):
    '''
    Based on the already implemented registration method, this class aims to implement
    the polyrigid registration over the cervicals, especifically from cervicals C1 to T2.
    The segmentations are collected from TotalSegmentator.
    '''

    def __init__(self,
                 PATH_input : str,
                 name_fixed_study : str = None,
                 name_moving_study : str = None,
                 load_transforms: bool = False,
                 mode = "fast",
                 name_raw_file = "raw.nii.gz",
                 name_mask_file = "mask_normal.nii.gz",
                 quiet = False):
        
        if mode not in ["full", "balanced", "fast"]:
            raise ValueError("Mode not supported, should be full, balanced or fast. Default set to fast")
        
        if mode == "full":
            self.cervical_labels = np.arange(50, 41, -1)
            self.cervical_names = {50:'C1', 49:'C2', 48:'C3', 47:'C4', 46:'C5', 45:'C6', 44:'C7', 43:'T1', 42:'T2'}

        elif mode == "balanced":
            self.cervical_labels = np.array([50, 49, 47, 45, 42])
            self.cervical_names = {50:'C1', 49:'C2', 47:'C4', 45:'C6', 42:'T2'}

        elif mode == "fast":
            self.cervical_labels = np.array([50, 47, 44, 42])
            self.cervical_names = {50:'C1', 47:'C4', 44:'C7', 42:'T2'}

        self.name_fixed_study = name_fixed_study
        self.name_moving_study = name_moving_study

        self.PATH_patient = PATH_input # Contains both studies.
        self.PATH_output = os.path.join(PATH_input, f"polyrigid_registration_{mode}")

        self.name_raw_file = name_raw_file
        self.name_mask_file = name_mask_file

        print("\nReading the data\n")
        self._read_data()
        self.polyrigid_transform = None #self._load_polyrigid_transform()
        
        # Similarly to PatientRegistration class, we use a mask around the neck to help the global registration method.
        roi_fixed = self._roi_definition(self.mask_fixed)
        roi_moving = self._roi_definition(self.mask_moving)

        roi_fixed_intersection = Roi.intersection(mask = self.mask_fixed,
                                                  roi_list = roi_fixed)
        
        roi_moving_intersection = Roi.intersection(mask = self.mask_moving,
                                                   roi_list = roi_moving)
        
        # Regions of interest as masks for the registration method.
        self.fixed_mask_sitk = sitk.GetImageFromArray(roi_fixed_intersection.astype(np.int32))
        self.fixed_mask_sitk.CopyInformation(self.fixed)

        self.moving_mask_sitk = sitk.GetImageFromArray(roi_moving_intersection.astype(np.int32))
        self.moving_mask_sitk.CopyInformation(self.moving)

        # Now we do start the polyrigid registration.
        if self.polyrigid_transform is None:
            self.fixed_masks = {self.cervical_names[label] : self._roi_intersection(self.mask_fixed_sitk, 
                                                            self.mask_fixed, label) for label in self.cervical_labels}
                
            self.moving_masks = {self.cervical_names[label] : self._roi_intersection(self.mask_moving_sitk, 
                                                            self.mask_moving, label) for label in self.cervical_labels}

            if load_transforms:
                self.cervical_transforms = self.load_transforms()
                self._create_polyrigid_displacement_field()
            else:
                print("\nSetting the global registration method\n")
                self._set_registration_method(quiet)
                print("\nExecuting the polyrigid registration method\n")
                self.cervical_transforms = self._execute_polyrigid_registration(quiet)
                print("\nYou waited a long time! Hopefully that worked\n")
                print("\nBut now you have to wait a bit more, sorry...\n")
                self._create_polyrigid_displacement_field()

        self._apply_polyrigid_transform()

    def _load_polyrigid_transform(self):

        print("\nAttempting to read the polyrigid transform\n")


        if os.path.exists(os.path.join(self.PATH_output, "polyrigid", "polyrigid.tfm")):
            polyrigid_transform = sitk.ReadTransform(os.path.join(self.PATH_output, "polyrigid", "polyrigid.tfm"))
            print("Transform succesfully read from: {}".format(os.path.join(self.PATH_output, "polyrigid", "polyrigid.tfm")))

            return polyrigid_transform
        else:
            return None
        
    def _apply_polyrigid_transform(self):

        registered_image = sitk.Resample(self.moving,
                                         self.fixed,
                                         self.polyrigid_transform,
                                         sitk.sitkLinear,
                                         0.0,
                                         self.moving.GetPixelID())
        
        sitk.WriteImage(registered_image, "registered_image.nii.gz")
            
    def _read_data(self):

        try:
            studies = sorted([study for study in os.listdir(self.PATH_patient) if not study.endswith(".DS_store") and not study.endswith("registration")])

            if len(studies) > 2 and self.name_fixed_study is None and self.name_moving_study is None : # More than two studies, we need to specify which one is gonna be used.
                raise ValueError("More than 2 folders are in the specified PATH, please provided full PATH for moving and fixed study.")
            
            # Just grabbing the first two which is probably not ideal.
            if self.name_fixed_study is None and self.name_moving_study is None:
                self.name_fixed_study = studies[0]
                self.name_moving_study = studies[1]

            PATH_fixed = os.path.join(self.PATH_patient, self.name_fixed_study, "totalsegmentator")
            PATH_moving = os.path.join(self.PATH_patient, self.name_moving_study, "totalsegmentator")

            print(f"PATH fixed study: {os.path.join(PATH_fixed, self.name_raw_file)}")
            print(f"PATH moving study: {os.path.join(PATH_moving, self.name_raw_file)}")

            # The data we need is in the totalsegmentator folder -> we need the raw image + mask.nii.gz
            image_fixed = sitk.ReadImage(os.path.join(PATH_fixed, self.name_raw_file))
            image_moving = sitk.ReadImage(os.path.join(PATH_moving, self.name_raw_file))

            self.fixed = sitk.Cast(image_fixed, sitk.sitkFloat32)
            self.moving = sitk.Cast(image_moving, sitk.sitkFloat32)

            # Mask fixed, this mask contains all the labels we'll be needing.
            self.mask_fixed_sitk = sitk.ReadImage(os.path.join(PATH_fixed, self.name_mask_file))
            self.mask_moving_sitk = sitk.ReadImage(os.path.join(PATH_moving, self.name_mask_file))

            self.mask_fixed = sitk.GetArrayFromImage(self.mask_fixed_sitk)
            self.mask_moving = sitk.GetArrayFromImage(self.mask_moving_sitk)

        except:
            print("There was an error reading the data from the PATH: {}".format(self.PATH_patient))

    def _cervical_mask(self, 
                       mask : np.array,
                       label : int) -> Optional[List[Roi]]:

        if np.any(mask == label):
            return [Roi([label], 'inside')]
        return None
        
    def _roi_intersection(self, 
                          mask_sitk : sitk.Image,
                          mask_array : np.array,
                          label : int) -> sitk.Image:
        
        roi_cervical = self._cervical_mask(mask_array, label)
        if roi_cervical == None:
            return None # Cervical not found.

        roi_intersection = Roi.intersection(mask = mask_array,
                                            roi_list = roi_cervical)
        roi_image = sitk.GetImageFromArray(roi_intersection.astype(np.uint8))
        roi_image.CopyInformation(mask_sitk)

        return roi_image

    def _set_cervical_registration_method(self,
                                          initial_transform : sitk.Transform, 
                                          cervical_label : int,
                                          quiet = False) -> Optional[sitk.ImageRegistrationMethod]:

        # Initial transform is a global transform.
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
        registration_method.SetMetricAsCorrelation() # Sensible to changes of the HU as long as the changes are similar.

        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,
                                                                     minStep=1e-5,
                                                                     numberOfIterations=100,
                                                                     gradientMagnitudeTolerance=1e-8)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        registration_method.SetShrinkFactorsPerLevel([3, 2, 1]) # 1/6 maybe to orient globally.
        registration_method.SetSmoothingSigmasPerLevel([1, 1, 0]) # -> higher if we have more noise.
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform, inPlace = True)
        registration_method.SetMetricFixedMask(self.fixed_masks[self.cervical_names[cervical_label]]) 
        registration_method.SetMetricMovingMask(self.moving_masks[self.cervical_names[cervical_label]])
        
        if not quiet:
            def command_iteration():
                print(
                    f"Level: {registration_method.GetCurrentLevel()} | "
                    f"Iteration: {registration_method.GetOptimizerIteration()} | "
                    f"Metric: {registration_method.GetMetricValue()}"
                )
            registration_method.AddCommand(sitk.sitkIterationEvent, command_iteration)

        return registration_method
    
    def _execute_polyrigid_registration(self, 
                                        quiet=False) -> Dict:
        self.global_transform = self.registration_method.Execute(self.fixed, self.moving)

        cervical_transforms = {}
        for cervical in self.cervical_labels:

            global_transform_copy = sitk.Transform(self.global_transform)
            name = self.cervical_names[cervical] # Actual name of the cervical C1, C2, ...

            if self.fixed_masks[name] is None or self.moving_masks[name] is None:
                print(f"Skipping {name}: mask not found in fixed or moving.")
                continue

            print(f"\nExecuting registration for {name}\n")     
            cervical_registration_method = self._set_cervical_registration_method(initial_transform = global_transform_copy,
                                                                                  cervical_label = cervical,
                                                                                  quiet = quiet)

            cervical_transform = cervical_registration_method.Execute(self.fixed, self.moving)
            
            if not quiet:
                print(f"{name} metric value:", cervical_registration_method.GetMetricValue())
                print(f"{name} stop condition:",
                    cervical_registration_method.GetOptimizerStopConditionDescription())

            cervical_transforms[name] = sitk.Transform(cervical_transform)

        return cervical_transforms

    # def _create_polyrigid_displacement_field(self,
    #                                          sigma_mm : float = 20.0):

    #     self.transforms = {}
    #     reference_image = self.fixed
    #     displacement_sum, weight_sum = None, None

    #     for label, transform in self.cervical_transforms.items():

    #         self.transforms.update({label : transform}) # To save the results.
    #         if label == "global":
    #             continue

    #         fixed_mask = self.fixed_masks[label]
    #         if fixed_mask is None:
    #             print(f"Skipping cervical {label} reason: mask not found.")
    #             continue

    #         distance_map = sitk.SignedMaurerDistanceMap(fixed_mask,
    #                                                     insideIsPositive = False,
    #                                                     squaredDistance = False,
    #                                                     useImageSpacing = True)
            
    #         distance_np = sitk.GetArrayFromImage(distance_map).astype(np.float32)
    #         distance_np = np.abs(distance_np)

    #         weight_np = np.exp(-(distance_np**2)/(2 * sigma_mm ** 2)).astype(np.float32)
    #         # weight_image = sitk.GetImageFromArray(weight_np.astype(np.float32))
    #         # weight_image.CopyInformation(reference_image)

    #         displacement_field = sitk.TransformToDisplacementField(transform,
    #                                                                sitk.sitkVectorFloat64,
    #                                                                reference_image.GetSize(),
    #                                                                reference_image.GetOrigin(),
    #                                                                reference_image.GetSpacing(),
    #                                                                reference_image.GetDirection())

    #         displacement_np = sitk.GetArrayFromImage(displacement_field).astype(np.float32)
    #         weighted_displacement_np = displacement_np * weight_np[..., np.newaxis]
            
    #         weighted_displacement = sitk.GetImageFromArray(weighted_displacement_np)
    #         weighted_displacement.CopyInformation(displacement_field)

    #         if displacement_sum is None:
    #             displacement_sum = weighted_displacement_np
    #             weight_sum = weight_np

    #         else:
    #             displacement_sum += weighted_displacement_np
    #             weight_sum += weight_np

    #     eps = 1e-8
    #     final_displacement_np = displacement_sum / (weight_sum[..., np.newaxis] + eps)

    #     final_displacement = sitk.GetImageFromArray(
    #         final_displacement_np.astype(np.float64),
    #         isVector=True
    #     )
    #     final_displacement.CopyInformation(reference_image)

    #     self.polyrigid_displacement_field = final_displacement
    #     self.polyrigid_transform = sitk.DisplacementFieldTransform(final_displacement)

    #     self.transforms["final_transform"] = self.polyrigid_transform

    def _create_polyrigid_displacement_field(self, sigma_mm: float = 20.0):

        self.transforms = {}
        reference_image = self.fixed

        displacement_sum = None  # float32, shape: z,y,x,3
        weight_sum = None        # float32, shape: z,y,x

        for label_name, transform in self.cervical_transforms.items():

            self.transforms[label_name] = transform

            if label_name == "global":
                continue

            fixed_mask = self.fixed_masks[label_name]

            if fixed_mask is None:
                print(f"Skipping cervical {label_name} reason: mask not found.")
                continue

            distance_map = sitk.SignedMaurerDistanceMap(
                fixed_mask,
                insideIsPositive=False,
                squaredDistance=False,
                useImageSpacing=True
            )

            distance_np = sitk.GetArrayFromImage(distance_map).astype(np.float32)
            np.abs(distance_np, out=distance_np)

            weight_np = np.empty_like(distance_np, dtype=np.float32)
            np.square(distance_np, out=weight_np)
            weight_np *= -1.0 / (2.0 * sigma_mm ** 2)
            np.exp(weight_np, out=weight_np)

            displacement_field = sitk.TransformToDisplacementField(
                transform,
                sitk.sitkVectorFloat64,
                reference_image.GetSize(),
                reference_image.GetOrigin(),
                reference_image.GetSpacing(),
                reference_image.GetDirection()
            )

            displacement_np = sitk.GetArrayFromImage(displacement_field).astype(np.float32)

            if displacement_sum is None:
                displacement_sum = np.zeros_like(displacement_np, dtype=np.float32)
                weight_sum = np.zeros_like(weight_np, dtype=np.float32)

            displacement_sum += displacement_np * weight_np[..., None]
            weight_sum += weight_np

            del distance_map, distance_np, weight_np, displacement_field, displacement_np

        eps = np.float32(1e-8)
        displacement_sum /= (weight_sum[..., None] + eps)

        final_displacement = sitk.GetImageFromArray(
            displacement_sum.astype(np.float64),
            isVector=True
        )
        final_displacement.CopyInformation(reference_image)

        self.polyrigid_displacement_field = final_displacement
        self.polyrigid_transform = sitk.DisplacementFieldTransform(final_displacement)
        self.transforms["final_transform"] = self.polyrigid_transform

    def load_transforms(self):
        PATH = os.path.join(self.PATH_output, "polyrigid")

        PATH_transforms = sorted(glob.glob(os.path.join(PATH, '*.tfm')))
        transforms_names = sorted([name.split(".")[0] for name in os.listdir(PATH) if name != '.DS_store'])

        loaded_transforms = {}
        loaded_transforms.update({name : sitk.ReadTransform(path) for path, name in zip(PATH_transforms, transforms_names)})

        return loaded_transforms

    def save_transforms(self):
        PATH = os.path.join(self.PATH_output, "polyrigid")
        os.makedirs(PATH, exist_ok=True)
        sitk.WriteTransform(self.global_transform, os.path.join(PATH, "global.tfm"))

        if not hasattr(self, "cervical_transforms") or self.cervical_transforms is None:
            raise ValueError("cervical_transforms does not exist. Run _execute_polyrigid_registration() first.")

        for cervical, transform in self.cervical_transforms.items():
            name = self.cervical_names[cervical]
            sitk.WriteTransform(transform, os.path.join(PATH, f"{name}.tfm"))

        sitk.WriteTransform(self.polyrigid_transform, os.path.join(PATH, "final_transform.tfm"))
        
    def get_transform(self):
        return self.polyrigid_transform
    
class SIREOutputRegistration:
    
    @staticmethod
    def _read_vtp(PATH_input):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(PATH_input)
        reader.Update()

        return reader.GetOutput()
    
    @staticmethod
    def _write_vtp(polydata : vtk.vtkPolyData, 
                   PATH):
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(PATH)
        writer.SetInputData(polydata)
        writer.Write()

    @staticmethod
    def register_data(PATH_input : str,
                       PATH_output : str,
                       sitk_transform : sitk.Transform):
        
        data = SIREOutputRegistration._read_vtp(PATH_input)
        transformed_data = vtk.vtkPolyData()
        transformed_data.DeepCopy(data)

        points = transformed_data.GetPoints()
        transformed_points = vtk.vtkPoints()
        transformed_points.SetNumberOfPoints(points.GetNumberOfPoints())

        inv_transform = sitk_transform.GetInverse()
        
        for i in range(points.GetNumberOfPoints()):
            p = points.GetPoint(i)
            p_transformed = inv_transform.TransformPoint(p)

            transformed_points.SetPoint(i, p_transformed)

        transformed_data.SetPoints(transformed_points)
        
        SIREOutputRegistration._write_vtp(transformed_data, PATH_output)
        
# Class to save also the meshes:
class MeshRegistration:

    @staticmethod
    def _register_mesh(PATH_input : str,
                       PATH_output : str,
                       sitk_transform : sitk.Transform,
                       file_name = "registered_mesh.stl"):
        
        inv_transform = sitk_transform.GetInverse()
        
        # Should be .../patient_series/meshes_correction/

        try:
            PATH_mesh = os.path.join(PATH_input, "mesh_correction", "vessels_fused_final.stl")
            read_mesh = trimesh.load(PATH_mesh)
        except:

            print("[WARNING] The complete meshes could not be found in the expected folder: mesh_correction/vessels_fused_final.stl")
            return

        vertices = read_mesh.vertices
        transformed_vertices = np.array([
            inv_transform.TransformPoint(tuple(v)) for v in vertices
        ])

        transformed_mesh = trimesh.Trimesh(
            vertices = transformed_vertices,
            faces = read_mesh.faces,
            process = False
        )

        transformed_mesh.export(os.path.join(PATH_output, file_name)) 