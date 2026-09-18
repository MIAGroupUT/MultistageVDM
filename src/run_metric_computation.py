import os
import vtk

import json
import torch
import typer
import trimesh
import numpy as np
import SimpleITK as sitk

from src.longitudinal_analysis.utils.utils_metric_computation import trimesh_to_vtk, stl_to_filled_mask_like
from src.longitudinal_analysis.utils.utils_metric_computation import dice_score, compute_asd, compute_hd
from src.longitudinal_analysis.utils.utils_reconstruction import mesh_reconstruction

app = typer.Typer()

def find_series_path(root_dir: str,
                     patient: str,
                     series_name: str) -> str:

    patient_dir = os.path.join(root_dir, patient)
    for study in os.listdir(patient_dir):
        study_path = os.path.join(patient_dir, study)

        if not os.path.isdir(study_path):
            continue

        candidate = os.path.join(study_path, series_name)

        if os.path.exists(candidate):
            return os.path.join(patient, study, series_name)

    raise FileNotFoundError(f"Series '{series_name}' not found for patient {patient}")

@app.command()
def match(patient: str = typer.Option(..., "-p", "--patient"),
          series: str = typer.Option(..., "-m", "--series"),
          study: str = typer.Option(..., "-s", "--study")):
    
    PATH_input = f'/deepstore/datasets/mia/UMCU_ECAA/D_Output/'
    
    vessels = ['external_carotid_artery_left',
               'external_carotid_artery_right',
               'internal_carotid_artery_left',
               'internal_carotid_artery_right']

    reconstruction_config = {"soft_union_tau" : 0.25,
                             "gaussian_sigma" : 2.0,
                             "omega" : 3,
                             "voxel_size" : 0.15}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_PATH = find_series_path(root_dir = PATH_input,
                                    patient = patient,
                                    series_name = series)

    mesh_reconstruction(PATH = os.path.join(PATH_input,patient_PATH), 
                        vessel_names = vessels,
                        device = device,
                        allow_mixed_reconstruction = False,
                        study = study,
                        **reconstruction_config)
    
    # Metric estimation
    image_PATH = os.path.join(PATH_input, patient_PATH, 'totalsegmentator', 'raw.nii.gz')
    image_sitk = sitk.ReadImage(image_PATH)

    # Upload segmentations.
    gt_PATH = os.path.join(PATH_input, patient_PATH, 'annotation','gt.nii')

    gt_sitk = sitk.ReadImage(gt_PATH)
    gt_sitk.CopyInformation(image_sitk)

    bin_filter = sitk.BinaryThresholdImageFilter()
    bin_filter.SetLowerThreshold(1)
    bin_filter.SetUpperThreshold(10) # Should be no more than 5
    binary_mask = bin_filter.Execute(gt_sitk)

    # Connected components
    cc = sitk.ConnectedComponent(binary_mask)
    cc_sorted = sitk.RelabelComponent(cc, sortByObjectSize=True)

    # Keep the two largest components: carotid left + carotid right
    two_largest = sitk.BinaryThreshold(
        cc_sorted,
        lowerThreshold=1,
        upperThreshold=2,
        insideValue=1,
        outsideValue=0
    )

    two_largest = sitk.Cast(two_largest, sitk.sitkUInt8)
    two_largest.CopyInformation(image_sitk)
    np_mask = sitk.GetArrayFromImage(two_largest)

    # Path mesh.
    mesh_PATH = os.path.join(PATH_input, patient_PATH, f'mesh_{study}', 'vessels_fused_final.stl')
    mesh_sitk, np_mesh = stl_to_filled_mask_like(mesh_PATH, image_sitk)

    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(two_largest, mesh_sitk)

    hd = hd_filter.GetHausdorffDistance()
    print("Dice: ", dice_score(np_mesh, np_mask))
    print("Hausdorff Distance: ", hd)
    print("Average Surface Distance: ", compute_asd(two_largest, mesh_sitk))


if __name__ == "__main__":
    app()  