import os
import vtk

import json
import torch
import typer
import trimesh
import numpy as np
import SimpleITK as sitk

# from utils_deformation_inr import *
from src.longitudinal_analysis.utils.utils_reconstruction import mesh_reconstruction
from src.longitudinal_analysis.utils.utils_reconstruction import CarotidVascularModel, MatchVascularModel
from src.longitudinal_analysis.utils.utils_image_registration import PatientRegistration, PolyRigidPatientRegistration, SIREOutputRegistration

app = typer.Typer()

from dataclasses import dataclass

@dataclass
class SeriesPaths:
    root: str
    patient: str
    moving: str
    fixed: str
    registration_folder: str

    @property
    def moving_series(self):
        return os.path.basename(self.moving)

    @property
    def fixed_series(self):
        return os.path.basename(self.fixed)

    @property
    def pair_name(self):
        return f"{self.moving_series}__to__{self.fixed_series}"

    @property
    def output(self):
        return os.path.join(self.root, self.patient, self.registration_folder, self.pair_name)

    @property
    def transforms(self):
        return os.path.join(self.output, "transforms")

    @property
    def final_transform(self):
        return os.path.join(self.transforms, "final_transform.tfm")

    @property
    def series_matching(self):
        return os.path.join(self.output, "series_matching")

    def series_dir(self, series):
        return os.path.join(self.root, series)

    def centerlines(self, series):
        return os.path.join(self.series_dir(series), "centerline_correction")

    def contours(self, series):
        return os.path.join(self.series_dir(series), "contour_correction", "lumen")

    def mask(self, series):
        return os.path.join(self.series_dir(series), "totalsegmentator", "mask_normal.nii.gz")

    @property
    def registered_centerlines(self):
        return os.path.join(self.output, "registered_centerline_correction")

    @property
    def registered_contours(self):
        return os.path.join(self.output, "registered_contour_correction", "lumen")

    @property
    def registered_mask(self):
        return os.path.join(self.output,
                            "registered_volumes",
                            "registered_mask_final_transform.nii.gz")

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

def register_centerlines_contours(PATH_centerlines,
                                  PATH_contours,
                                  PATH_output,
                                  final_transform):
    # Registering the centerlines and the contours, starting with the centerlines.
    os.makedirs(os.path.join(PATH_output, "registered_centerline_correction"), exist_ok=True)

    for vessel in [file for file in os.listdir(PATH_centerlines) if not file.endswith('.DS_store')]:

        path = os.path.join(PATH_centerlines, vessel)
        path_output = os.path.join(PATH_output, "registered_centerline_correction", vessel)

        SIREOutputRegistration.register_data(PATH_input = path,
                                             PATH_output = path_output,
                                             sitk_transform = final_transform)

    # Now the contour registration, in the same way as the centerline registration.
    # The same function works for centerlines and contours.
    os.makedirs(os.path.join(PATH_output, "registered_contour_correction", "lumen"), exist_ok=True)

    for vessel in [file for file in os.listdir(PATH_contours) if not file.endswith('.DS_store')]:

        path = os.path.join(PATH_contours, vessel)
        path_output = os.path.join(PATH_output, "registered_contour_correction", "lumen", vessel)

        SIREOutputRegistration.register_data(PATH_input = path,
                                             PATH_output = path_output,
                                             sitk_transform = final_transform)

def save_json(metadata: dict, PATH):

    os.makedirs(PATH, exist_ok = True)

    with open(os.path.join(PATH, "registration_info.json"), "w") as f:
        json.dump(metadata, f, indent = 4)

def match_series(PATH_input,
                 patient,
                 moving,
                 fixed,
                 **kwargs):
    
    # Identification of the mode, and getting the values from kwargs for the registration and
    # reconstruction configuration.
    polyrigid_mode = kwargs.get("polyrigid_mode", "fast")
    force_registration = kwargs.get("force_registration", True)
    save_metadata = kwargs.get("save_json", True)
    refinement_side = kwargs.get("refinement_side", None)

    output_file_name = "vessels_fused_final.stl"
    allow_mixed_reconstruction = kwargs.get("mixed_reconstruction", False)
    reconstruction_config = kwargs.get("reconstruction_config", {})

    if polyrigid_mode in ["fast", "balanced", "full"]:
        polyrigid = True
    
    else:
        polyrigid = False
    registration_folder = "registration" if not polyrigid else f"polyrigid_registration_{polyrigid_mode}"
    registration_type = "multi_stage" if not polyrigid else "polyrigid"

    # First creating a instance of series path to divide the path.
    paths = SeriesPaths(root = PATH_input,
                        patient = patient,
                        moving = moving,
                        fixed = fixed,
                        registration_folder = registration_folder)
    pair_name = paths.pair_name

    # Dictionary to save data for later reference, will be saved in a json
    json_dict = {"root_dir" : PATH_input,
                 "moving_file" : moving,
                 "fixed_file" : fixed,
                 "registration_type" : registration_type}
    if polyrigid:
        json_dict.update({"polyrigid_mode" : polyrigid_mode})

    # Starting the registration.    
    if not force_registration: # We try to read the transform.
        try:
            # First we verify if there is a transform already saved.
            PATH_transform = paths.final_transform
            final_transform = sitk.ReadTransform(PATH_transform)

        except Exception as e:
            print(f"Could not read existing transform, regenerating it. Reason: {e}")
            final_transform = None

    else:
        final_transform = None # To enter if to generate the registration again.

    if final_transform is None: # Forced to generate the registration again.
        if polyrigid:
            # Here moving and fixed are swapped since the generated is not invertible (needed for VTK and Trimesh).
            # This is INTENTIONAL, you can change PolyRigidPatientRegistration to make it less confusion but is not
            # completely necessary.
            registrator = PolyRigidPatientRegistration(PATH_input,
                                                       name_moving_study = fixed,
                                                       name_fixed_study = moving,
                                                       mode = polyrigid_mode)
            registrator.save_registered_data(registration_type = "polyrigid",
                                             register_meshes = False)
            final_transform = registrator.get_transform()

        else:
            registrator = PatientRegistration(PATH_input,
                                              patient,
                                              name_moving_study = moving,
                                              name_fixed_study = fixed,
                                              multi_stage_registration = True,
                                              refinement_side = refinement_side,
                                              quiet = False)
            
            registrator.save_registered_data(registration_type = "multistage",
                                             register_meshes = True) # Change to verify if this is possible with try/except
            transforms = registrator.get_transform()
            final_transform = transforms["final_transform"]

    # Now we register the centerlines and the contours using the final transform.
    # At this point, only the volumes (CT) are registered.
    register_centerlines_contours(PATH_centerlines = paths.centerlines(moving),
                                  PATH_contours = paths.contours(moving),
                                  PATH_output = paths.output,
                                  final_transform = final_transform)
    
    # At this point we have a set of registered centerlines and contours, so we have to match the centerlines
    # and contours, before creating again the meshes.
    # In this context:
    #   * model_1 -> Is the fixed one.
    #   * model_2 -> Is the moving one.

    vessels = ['distal_carotid_artery_left',
               'distal_carotid_artery_right',
               'external_carotid_artery_left',
               'external_carotid_artery_right',
               'internal_carotid_artery_left',
               'internal_carotid_artery_right']

    model_1 = CarotidVascularModel.load_from_directory(root_dir_centerlines = paths.centerlines(fixed),
                                                       root_dir_contours = paths.contours(fixed),
                                                       dir_mask = paths.mask(fixed),
                                                       filenames = vessels)

    model_2 = CarotidVascularModel.load_from_directory(root_dir_centerlines = os.path.join(paths.output, "registered_centerline_correction"),
                                                       root_dir_contours = os.path.join(paths.output, "registered_contour_correction", "lumen"),
                                                       dir_mask = os.path.join(paths.output, "registered_volumes", "registered_mask_final_transform.nii.gz"),
                                                       filenames = vessels)
    
    # This folder will keep the pruned contours and centerlines.
    os.makedirs(paths.series_matching, exist_ok = True)
    
    MatchVascularModel(series_1 = model_1,
                       series_2 = model_2,
                       split_vessels = False, # This creates ICA, ECA and CCA. Experimental.
                       output_dir = paths.series_matching)
    
    if save_metadata: # Saving this data for reference.
        json_dict["series_1_matched"] = fixed
        json_dict["series_2_matched"] = moving
        json_dict.update(reconstruction_config)
        save_json(json_dict, paths.output)

    # Finally we have to redo the meshes, now that they are registered, and matched.
    # This is the final step to have them ready for INR deformation fields.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_reconstruction(PATH = os.path.join(paths.series_matching, "series_1_matched"), 
                        vessel_names = vessels,
                        device = device,
                        allow_mixed_reconstruction = allow_mixed_reconstruction,
                        output_file_name = output_file_name,
                        **reconstruction_config)
    mesh_reconstruction(PATH = os.path.join(paths.series_matching, "series_2_matched"),
                        vessel_names = vessels,
                        device = device,
                        allow_mixed_reconstruction = allow_mixed_reconstruction,
                        output_file_name = output_file_name,
                        **reconstruction_config)
    
@app.command()
def match(patient: str = typer.Option(..., "-p", "--patient"),
          moving: str = typer.Option(..., "-m", "--moving"),
          fixed: str = typer.Option(..., "-f", "--fixed")):

    reconstruction_config = {"voxel_size" : 0.15,
                             "soft_union_tau" : 0.5,
                             "gaussian_sigma" : 3,
                             "siren_omega" : 4.5}

    config = {"polyrigid_mode" : None,
              "force_registration" : False,
              "reconstruction_config" : reconstruction_config}

    PATH_input = f'/deepstore/datasets/mia/UMCU_ECAA/D_Output/'

    moving_PATH = find_series_path(root_dir = PATH_input,
                                   patient = patient,
                                   series_name = moving)
    
    fixed_PATH = find_series_path(root_dir = PATH_input,
                                  patient = patient,
                                  series_name = fixed)

    match_series(PATH_input,
                 patient,
                 moving_PATH,
                 fixed_PATH,
                 **config)
    
if __name__ == "__main__":
    app()  