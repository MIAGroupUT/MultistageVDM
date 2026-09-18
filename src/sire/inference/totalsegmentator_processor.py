import logging
import os

from typing import List

import numpy as np
import SimpleITK as sitk
import torch

from logdecorator import log_on_start
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion
from totalsegmentator.python_api import totalsegmentator

from src.sire.utils.affine import get_affine, transform_points

RICH_INFO = 25
logging._levelToName[RICH_INFO] = "RICH_INFO"
logging._nameToLevel["RICH_INFO"] = RICH_INFO


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
        return np.prod(np.stack([roi(mask) for roi in roi_list]), axis=0) > 0

class TotalSegmentatorProcessor:
    """Automatic seed and roi detector based on totalsegmentator."""

    def __call__(self, logging_level: int):
        self.logging_level = logging_level
        logging.basicConfig(level=logging_level, format="%(levelname)s (%(asctime)s): %(message)s")

    def _get_seed(self, mask: np.array, affine: torch.tensor, roi: np.array, index: int, p: float = 1.0):
        """Seed extraction from TotalSegmentator rough segmentations.
        Note that if the TotalSegmentator doesn't support the class this
        method won't work.

        Args:
            mask (np.array): global TotalSegmentator mask with all the classes
            affine (torch.tensor): affine matrice of the volume
            roi (np.array): region of interest for given vessel
            index (int): label of the TotalSegmentator class to extract seed from
            p (float, optional): percentile in [0, 1], meaning how high along z-axis the seed
                                 should be extracted. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        labelled = label(mask * roi.astype(int) == index)
        mask = labelled == np.argmax(np.bincount(labelled.flatten())[1:]) + 1

        z_indices = np.argwhere(np.any(np.any(mask, axis=1), axis=1))
        z = np.percentile(z_indices, 100 * (1 - p)).astype(int)

        seed_point = np.array([z] + list(center_of_mass(mask[z])))[[2, 1, 0]] #2,1,0
        return transform_points(torch.from_numpy(seed_point).reshape(1, 3), affine).numpy()
    
    def _get_seed_common_carotid(self, mask_path: str, label: int):

        """Experimental seed extraction from TotalSegmentator rough segmentations.
        Note that if the TotalSegmentator doesn't support the class this
        method won't work.

        Args:
            mask_path (str): path to the mask generated by the TotalSegmentator.
            label (int): label of the structure of interest.

        Returns:
            _type_: _description_
        """

        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt16)
        mask_img = sitk.DICOMOrient(mask_img, "LPS") # Reading the mask and checking the orientation.

        binary = sitk.Equal(mask_img, label) # Creating a binary mask for the indicated label.

        stats = sitk.LabelShapeStatisticsImageFilter() # Getting statistics of the binary mask.
        stats.Execute(binary)

        if not stats.HasLabel(1): # We couldn't find the label.
            raise ValueError(f"Label {label} not found in TS mask")

        return np.array(stats.GetCentroid(1))  # Getting the seed, TODO: be able to get percentile X.

    @log_on_start(RICH_INFO, "Running TotalSegmentator preprocessing")
    def run(self, image: str, 
            output_dir: str, 
            device: str = "gpu",
            fast: bool = True,
            mask_name: str = "mask.nii.gz"):
        
        os.makedirs(os.path.join(output_dir, "totalsegmentator"), exist_ok=True)
        nifti_path = os.path.join(output_dir, "totalsegmentator", "raw.nii.gz")
        mask_path = os.path.join(output_dir, "totalsegmentator", mask_name)

        itk_image = sitk.ReadImage(image)
        affine, _ = get_affine(itk_image)

        # First check if raw.nii.gz exists, since might be already be created by ts preprocessing.
        if not os.path.isfile(nifti_path):
            sitk.WriteImage(itk_image, nifti_path)

        try:
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)
            print("Mask {} succesfully read from {}".format(mask_name, mask_path))

        except Exception as e:
            print(f"Mask will be created by TotalSegmentator... Reason: {e}")

            if not fast:
                mask_name = "mask_normal.nii.gz"

            totalsegmentator(input = nifti_path,
                             output = mask_path,
                             device = device,
                             ml = True,
                             fast = fast,
                             quiet = True)
            mask_sitk = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(mask_sitk)

        return mask, affine
