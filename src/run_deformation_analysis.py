import os
import json
import torch
import typer
import trimesh
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from tqdm import tqdm
from typing import Dict, List
from torchdiffeq import odeint
from dataclasses import dataclass
from src.longitudinal_analysis.utils.utils_deformation_inr import SIREN_VVF
from src.longitudinal_analysis.utils.utils_deformation_inr import VelocityODE, MiniLocalHybridODE, HybridVelocityODE, MultiGaussVelocityODE, LocalODE
from src.longitudinal_analysis.utils.utils_deformation_inr import get_bbox, normalize_points, chamfer_distance_torch, define_time_span, read_meshes, denormalize_points
from src.longitudinal_analysis.utils.utils_deformation_inr import define_local_inr_roi, deformation_field, apply_heatmap_and_save, strain_map

from src.longitudinal_analysis.utils.utils_metric_computation import stl_to_filled_mask_like, compute_hd, compute_asd

app = typer.Typer()

## <== Utils for setting up the training and creating the training loop ==> ##

def save_json(metadata: dict, PATH):
    with open(PATH, "w") as f:
        json.dump(metadata, f, indent = 4)

def train_global_model(ode_func : VelocityODE,
                       optimizer_global,
                       matched_mesh_1 : trimesh.Trimesh,
                       matched_mesh_2 : trimesh.Trimesh,
                       device : torch.device,
                       t_span : torch.Tensor,
                       samples : int = 20000,
                       epochs : int = 1000):

    bbox_min, bbox_max = get_bbox(matched_mesh_1 = matched_mesh_1,
                                  matched_mesh_2 = matched_mesh_2)
    
    print(f"Training the global model, samples : {samples} epochs {epochs}")

    for param in ode_func.model.parameters():
        param.requires_grad = True
    ode_func.model.train()
    
    loss_plot_global = []
    for epoch in tqdm(range(epochs)):

        points_0, _ = trimesh.sample.sample_surface(matched_mesh_2, count = samples) # Points from the moving one.
        points_1, _ = trimesh.sample.sample_surface(matched_mesh_1, count = samples) # Points from the fixed one.

        # Normalized points.
        points_0 = normalize_points(points_0, bbox_min, bbox_max) 
        points_1 = normalize_points(points_1, bbox_min, bbox_max)

        p0 = torch.tensor(points_0, dtype=torch.float32, device=device).unsqueeze(0) # Moving 
        p1 = torch.tensor(points_1, dtype=torch.float32, device=device).unsqueeze(0) # Fixed

        trajectory = odeint(ode_func,
                            p0,
                            t_span,
                            method = "euler")

        # Building the outout of the model.
        p0_warped = trajectory[-1]
        loss = chamfer_distance_torch(p0_warped, p1) # This one is bidirectional, therefore the result is the avg, maybe try with unidirectional? 

        optimizer_global.zero_grad()
        loss.backward()
        optimizer_global.step()
        loss_plot_global.append(loss.item())

    # torch.save(ode_func.model.state_dict(), os.path.join(PATH_output, file_name))
    # print(f"Global model is trained! Saved in: {os.path.join(PATH_output, file_name)}")

    return loss_plot_global

def train_mini_local_models(ode_func: MiniLocalHybridODE,
                            PATH_output: str,
                            model_optimizers: List[torch.optim.Adam],
                            matched_mesh_1: trimesh.Trimesh,
                            matched_mesh_2: trimesh.Trimesh,
                            device: torch.device,
                            file_name : str = 'local_model.pt',
                            samples : int = 2000,
                            epochs : int = 500):

    
    PATH_output = os.path.join(PATH_output, "models")
    os.makedirs(PATH_output, exist_ok = True)

    bbox_min, bbox_max = get_bbox(matched_mesh_1 = matched_mesh_1,
                                  matched_mesh_2 = matched_mesh_2)
    
    all_loss_plots = []
    
    for i in range(0, len(model_optimizers)):
        loss_plot_local = []
        print(f"Training the local model {i}/{len(model_optimizers)}, samples : {samples} epochs {epochs}")
        ode_func.freeze_every_local_model_but(idx = i)

        for epoch in tqdm(range(epochs)):

            points_0, _ = trimesh.sample.sample_surface(matched_mesh_2, count = samples) # Points from the moving one.
            points_1, _ = trimesh.sample.sample_surface(matched_mesh_1, count = samples) # Points from the fixed one.

            # Normalized points.
            points_0 = normalize_points(points_0, bbox_min, bbox_max) 
            points_1 = normalize_points(points_1, bbox_min, bbox_max)

            p0 = torch.tensor(points_0, dtype=torch.float32, device=device).unsqueeze(0) # Moving 
            p1 = torch.tensor(points_1, dtype=torch.float32, device=device).unsqueeze(0) # Fixed

            trajectory = odeint(ode_func,
                                p0,
                                define_time_span(device = device),
                                method = "euler")

            p0_warped = trajectory[-1]
            loss = chamfer_distance_torch(p0_warped, p1) # This one is bidirectional, therefore the result is the avg, maybe try with unidirectional? 

            model_optimizers[i].zero_grad()
            loss.backward()
            model_optimizers[i].step()
            loss_plot_local.append(loss.item())

        ode_func.mark_model_as_trained(idx = i)
        all_loss_plots.append(loss_plot_local)

    return all_loss_plots

def train_local_model(local_ode_func: LocalODE,
                      global_ode_func: VelocityODE,
                      optimizer_local: torch.optim,
                      matched_mesh_1: trimesh.Trimesh,
                      matched_mesh_2: trimesh.Trimesh,
                      device: torch.device,
                      t_span: torch.Tensor,
                      samples: int = 2000,
                      epochs: int = 1000):

    bbox_min, bbox_max = get_bbox(matched_mesh_1 = matched_mesh_1,
                                  matched_mesh_2 = matched_mesh_2)
    
    print(f"Training the local model, samples : {samples} epochs {epochs}")

    for param in local_ode_func.local_model.parameters():
        param.requires_grad = True
    local_ode_func.local_model.train()

    # Freezing the global model first.
    global_ode_func.freeze_model()

    loss_plot_local = []
    for epoch in tqdm(range(epochs)):

        points_0, _ = trimesh.sample.sample_surface(matched_mesh_2, count = samples) # Points from the moving one.
        points_1, _ = trimesh.sample.sample_surface(matched_mesh_1, count = samples) # Points from the fixed one.

        # Normalized points.
        points_0 = normalize_points(points_0, bbox_min, bbox_max) 
        points_1 = normalize_points(points_1, bbox_min, bbox_max)

        p0 = torch.tensor(points_0, dtype=torch.float32, device=device).unsqueeze(0) # Moving 
        p1 = torch.tensor(points_1, dtype=torch.float32, device=device).unsqueeze(0) # Fixed

        global_trajectory = odeint(global_ode_func,
                                   p0,
                                   t_span,
                                   method = "euler")
        
        global_p0_warped = global_trajectory[-1]

        final_trajectory = odeint(local_ode_func,
                                  global_p0_warped,
                                  t_span,
                                  method = "euler")
        
        optimizer_local.zero_grad()

        # Building the outout of the model.
        final_p0_warped = final_trajectory[-1]
        loss = chamfer_distance_torch(final_p0_warped, p1) # This one is bidirectional, therefore the result is the avg, maybe try with unidirectional? 

        loss.backward()
        optimizer_local.step()
        loss_plot_local.append(loss.item())

    return loss_plot_local

def global_model_setup(device : torch.device,
                       layers = [3, 256, 256, 256, 3],
                       omega : int = 5):
    
    global_model = SIREN_VVF(layers = layers,
                             omega = omega).to(device) # Make some experiments with the scales.
    optimizer_global = torch.optim.Adam(global_model.parameters(), lr = 1e-4)
    ode_func = VelocityODE(model = global_model)

    return ode_func, optimizer_global

def local_model_setup(global_model : SIREN_VVF,
                      device : torch.device,
                      **kwargs):
    
    # Default parameters of the model.
    layers = kwargs.get('layers', [3, 256, 256, 256, 3])
    omega = kwargs.get('omega', 12)

    # Getting specified mode.
    mode = kwargs.get('mode', 'multianchor')

    if mode == 'point_cloud':
        norm_roi_points = kwargs.get('norm_roi_points', None) # If we are using the point cloud approach.
        gaussian_sigma = kwargs.get('gaussian_sigma', 3)
    
    elif mode == 'multianchor':
        anchors_info = kwargs.get('anchors_info', None) # If we are using the multigaussian approach.
    
    else:
        raise ValueError("Unknow mode, should be point_cloud or multianchor")
    
    print(f"Selected mode for local model setup : {mode}")

    local_model = SIREN_VVF(layers = layers,
                            omega = omega).to(device) # Make some experiments with the scales.
    optimizer_local = torch.optim.Adam(local_model.parameters(), lr = 1e-4)

    if mode == 'point_cloud' and norm_roi_points is not None:

        roi_cloud = torch.as_tensor(norm_roi_points,
                                    dtype = torch.float32,
                                    device = device)
        hybrid_ode_func = LocalODE(roi_cloud = roi_cloud,
                                   local_model = local_model,
                                   sigma = gaussian_sigma,
                                   device = device)
        
    elif mode == 'multianchor' and anchors_info is not None:

        hybrid_ode_func = MultiGaussVelocityODE(global_model = global_model,
                                                local_model = local_model,
                                                anchors_info = anchors_info,
                                                device = device)
        
    return hybrid_ode_func, optimizer_local

def multigaussian_local_model_setup(device: torch.device,
                                    anchor_info: Dict,
                                    global_model: SIREN_VVF,
                                    layers = [3, 256, 256, 256, 3],
                                    omega : int = 15):
    
    local_model = SIREN_VVF(layers = layers, omega = omega).to(device)
    optimizer_local = torch.optim.Adam(local_model.parameters(), lr = 1e-4)

    hybrid_ode_func = MultiGaussVelocityODE(global_model = global_model,
                                            local_model = local_model,
                                            anchor_info = anchor_info,
                                            device = device)
    
    return hybrid_ode_func, optimizer_local

def fit_models(PATH_input : str,
               moving : str, 
               fixed : str,
               bbox_max,
               bbox_min,
               roi_config_dict : dict):

    PATH_output = os.path.join(PATH_input, "registration", "{}__to__{}".format(moving, fixed), "series_matching", "deformation_analysis")

    matched_mesh_1, matched_mesh_2 = read_meshes(PATH_input = PATH_input,
                                                 moving = moving,
                                                 fixed = fixed)

    # This config dict indicates where the local INR should have effect on.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Mac usage, change to gpu if needed.

    ode_func, optimizer_global = global_model_setup(device = device)
    train_global_model(ode_func = ode_func,
                       PATH_output = PATH_output,
                       optimizer_global = optimizer_global,
                       matched_mesh_1 = matched_mesh_1,
                       matched_mesh_2 = matched_mesh_2,
                       device = device,
                       t_span = define_time_span(device))

    # Local model setup
    # roi_points = define_local_inr_roi(PATH_matched_series = os.path.join(PATH_input, "registration", "{}__to__{}".format(moving, fixed), "series_matching"), # Outputs all the points stacked.
    #                                 config_dict = roi_config_dict) 
    # norm_roi_points = normalize_points(points = roi_points,
    #                                 bbox_min = bbox_min,
    #                                 bbox_max = bbox_max)

    hybrid_ode_func, optimizer_local = multigaussian_local_model_setup(device = device,
                                                                       anchor_info = roi_config_dict,
                                                                       global_model = ode_func.model,
                                                                       omega = 20)
    train_local_model(hybrid_ode_func = hybrid_ode_func,
                      PATH_output = PATH_output,
                      optimizer_local = optimizer_local,
                      matched_mesh_1 = matched_mesh_1,
                      matched_mesh_2 = matched_mesh_2,
                      device = device,
                      t_span = define_time_span(device))
    
    return ode_func, hybrid_ode_func

def global_inference(global_ode_func : VelocityODE,
                     original_mesh : trimesh.Trimesh, 
                     bbox_min, 
                     bbox_max,
                     device : torch.device,
                     PATH_output : str,
                     t_span : torch.Tensor,
                     filename : str = "output.stl") -> trimesh.Trimesh:
    
    all_points_np = normalize_points(original_mesh.vertices, bbox_min, bbox_max)

    with torch.no_grad():
        all_points = torch.tensor(all_points_np, dtype = torch.float32, device = device).unsqueeze(0)
        trajectory = odeint(global_ode_func,
                            all_points,
                            t_span,
                            method = "euler")
        
        all_points_warped = trajectory[-1].squeeze(0).detach().cpu().numpy()

    warped_vertices = denormalize_points(all_points_warped, bbox_min, bbox_max)

    faces = original_mesh.faces 
    warped_mesh = trimesh.Trimesh(vertices = warped_vertices,faces = faces,process = False)
    path_mesh_save = os.path.join(PATH_output, filename)
    warped_mesh.export(path_mesh_save)
        
    return warped_mesh

def inference(global_ode_func : VelocityODE,
              local_ode_func : LocalODE,
              original_mesh, 
              bbox_min, 
              bbox_max,
              PATH_output : str,
              t_span : torch.Tensor,
              device : torch.device,
              filename : str = "output.stl") -> trimesh.Trimesh:

    all_points_np = normalize_points(original_mesh.vertices, bbox_min, bbox_max)

    with torch.no_grad():
        all_points = torch.tensor(all_points_np, dtype = torch.float32,
                                  device = device).unsqueeze(0)
        global_trajectory = odeint(global_ode_func,
                                   all_points,
                                   t_span,
                                   method="euler")
        all_points_global_warped = global_trajectory[-1]

        final_trajectory = odeint(local_ode_func,
                                  all_points_global_warped,
                                  t_span,
                                  method = "euler")
        all_points_warped = final_trajectory[-1].squeeze(0).detach().cpu().numpy()

    warped_vertices = denormalize_points(all_points_warped, bbox_min, bbox_max)

    faces = original_mesh.faces 
    warped_mesh = trimesh.Trimesh(vertices = warped_vertices,faces = faces,process = False)
    path_mesh_save = os.path.join(PATH_output, filename)
    warped_mesh.export(path_mesh_save)

    return warped_mesh

def inference_mini_local_inr(ode_func: MiniLocalHybridODE ,
                             original_mesh: trimesh.Trimesh, 
                             bbox_min: np.ndarray, 
                             bbox_max: np.ndarray,
                             PATH_output: str,
                             t_span: torch.Tensor,
                             device: torch.device,
                             filename : str = "output.stl"):

    all_points_np = normalize_points(original_mesh.vertices, bbox_min, bbox_max)
    with torch.no_grad():
        all_points = torch.tensor(all_points_np, dtype = torch.float32,
                                  device = device).unsqueeze(0)
        trajectory = odeint(ode_func,
                            all_points,
                            t_span,
                            method="euler")
        
        all_points_warped = trajectory[-1].squeeze(0).detach().cpu().numpy()

    warped_vertices = denormalize_points(all_points_warped, bbox_min, bbox_max)
    faces = original_mesh.faces 
    warped_mesh = trimesh.Trimesh(vertices = warped_vertices,faces = faces,process = False)
    path_mesh_save = os.path.join(PATH_output, filename)
    warped_mesh.export(path_mesh_save)

    return warped_mesh

def load_models(PATH_saved_models : str,
                device : torch.device,
                ode_func : VelocityODE | HybridVelocityODE | LocalODE,
                hybrid_ode_func : VelocityODE | HybridVelocityODE | LocalODE):

    PATH_global_model = os.path.join(PATH_saved_models, 'global_model.pt')
    PATH_local_model = os.path.join(PATH_saved_models, 'local_model.pt')

    ode_func.model.load_state_dict(torch.load(PATH_global_model, map_location = device))
    
    hybrid_ode_func.global_model.load_state_dict(torch.load(PATH_global_model, map_location = device))
    hybrid_ode_func.local_model.load_state_dict(torch.load(PATH_local_model, map_location = device))

    return ode_func, hybrid_ode_func

@dataclass
class SeriesPath:
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
    def input(self):
        return os.path.join(self.root, self.patient, self.registration_folder, self.pair_name)
    
    @property
    def output(self):
        return os.path.join(self.series_matching, "deformation_analysis")

    @property
    def pair_name(self):
        return f"{self.moving_series}__to__{self.fixed_series}"

    @property
    def series_matching(self):
        return os.path.join(self.input, "series_matching")
    
    @property
    def series_1(self):
        return os.path.join(self.series_matching, "series_1_matched")
    
    @property
    def series_2(self):
        return os.path.join(self.series_matching, "series_2_matched")

def run_deformation_analysis(PATH_input, # patient/registration/pair_name
                             patient,
                             moving,
                             fixed,
                             save_models: bool = True,
                             **kwargs):
    
    ## <== Extraction of the configuration of the models from kwargs ==> ##

    roi_config_dict = kwargs.get("roi_config_dict", {})
    local_model_config = kwargs.get("local_model_config", {})
    force_training = kwargs.get("force_training", False)

    ## <== Initializing the PATH manager and selection of the device ==> ##

    paths = SeriesPath(root = PATH_input,
                       patient = patient,
                       moving = moving,
                       fixed = fixed,
                       registration_folder = "registration")   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  * Selected device: {device}")

    ## <=== Definition and creation of the output folders ===> ##

    PATH_matched_series = paths.series_matching
    PATH_saved_models = os.path.join(paths.output, "models")
    PATH_output_meshes = os.path.join(paths.output, "meshes")
    PATH_output_deformation_fields = os.path.join(paths.output, "deformation_fields")

    os.makedirs(PATH_saved_models, exist_ok = True)
    os.makedirs(PATH_output_meshes, exist_ok = True)
    os.makedirs(PATH_output_deformation_fields, exist_ok = True)

    ## <=== Reading the meshes ===> ##

    mesh_1, mesh_2 = read_meshes(fixed = paths.series_1,
                                 moving = paths.series_2)
    bbox_min, bbox_max = get_bbox(matched_mesh_1 = mesh_1,
                                  matched_mesh_2 = mesh_2)

    ## <=== Definition of the ROI that will be used for the training of the local model ===> ##
    
    roi_points = define_local_inr_roi(PATH_matched_series = PATH_matched_series,
                                      config_dict = roi_config_dict)
    local_model_config.update({'norm_roi_points' : normalize_points(roi_points, bbox_min, bbox_max)})

    ## <=== Training of both models ===> ##

    if force_training:
        ode_func, global_optimizer = global_model_setup(device = device,
                                                        omega = 5)
        global_loss = train_global_model(ode_func = ode_func,
                                         optimizer_global = global_optimizer,
                                         matched_mesh_1 = mesh_1,
                                         matched_mesh_2 = mesh_2,
                                         device = device,
                                         t_span = define_time_span(device = device))
        
        if save_models:
            torch.save(ode_func.model.state_dict(), os.path.join(PATH_saved_models, "global_model.pt"))

        local_ode_func, local_optimizer = local_model_setup(global_model = ode_func.model,
                                                             device = device,
                                                             **local_model_config)
        local_loss = train_local_model(local_ode_func = local_ode_func,
                                       global_ode_func = ode_func,
                                       optimizer_local = local_optimizer,
                                       matched_mesh_1 = mesh_1,
                                       matched_mesh_2 = mesh_2,
                                       device = device,
                                       t_span = define_time_span(device = device))    
        
        if save_models:
            torch.save(local_ode_func.local_model.state_dict(), os.path.join(PATH_saved_models, "local_model.pt"))

    else:
        try:
            ode_func, global_optimizer = global_model_setup(device = device,
                                                            omega = 5)
            local_ode_func, local_optimizer = local_model_setup(global_model = ode_func.model,
                                                                 device = device,
                                                                 **local_model_config)

            ode_func, local_ode_func = load_models(PATH_saved_models = PATH_saved_models,
                                                    ode_func = ode_func,
                                                    device = device,
                                                    hybrid_ode_func = local_ode_func)

        except:
            print(f"No .pt files could be found in the specified folder {PATH_saved_models}")


    ## <=== Creation and saving of the meshes and deformation arrows ===> ##

    global_warped_mesh = global_inference(global_ode_func = ode_func,
                                          original_mesh = mesh_2,
                                          bbox_min = bbox_min,
                                          bbox_max = bbox_max,
                                          device = device,
                                          PATH_output = PATH_output_meshes,
                                          t_span = define_time_span(device = device),
                                          filename = "global.stl")

    final_warped_mesh = inference(global_ode_func = ode_func,
                                  local_ode_func = local_ode_func,
                                  original_mesh = mesh_2,
                                  bbox_min = bbox_min,
                                  bbox_max = bbox_max,
                                  device = device,
                                  PATH_output = PATH_output_meshes,
                                  t_span = define_time_span(device = device),
                                  filename = "final.stl")
    
    ## <=== Creation and saving of the meshes and deformation fields ===> ##

    heatmap_global, _ = deformation_field(original_mesh = mesh_2, # This is the moving mesh, the fixed mesh is 1
                                          warped_mesh = global_warped_mesh,
                                          PATH_save = PATH_output_deformation_fields,
                                          output_file_name = "deformation_field_global.vtp",
                                          mode = "all",
                                          sample_ratio = 0.1,
                                          use_fixed_max_displacement = True)

    heatmap_local, _ = deformation_field(original_mesh = global_warped_mesh,
                                         warped_mesh = final_warped_mesh,
                                         PATH_save = PATH_output_deformation_fields,
                                         output_file_name = "deformation_field_normal_local.vtp",
                                         mode = "normal",
                                         sample_ratio = 0.1,
                                         threshold = 0.4,
                                         use_fixed_max_displacement = False)
    
    apply_heatmap_and_save(mesh = mesh_1,
                           heatmap = heatmap_local,
                           PATH_ouput = PATH_output_meshes,
                           filename = 'normal_prj_over_t1.ply') # This one is the moving one.
    
    apply_heatmap_and_save(mesh = mesh_2,
                           heatmap = heatmap_local,
                           PATH_ouput = PATH_output_meshes,
                           filename = 'normal_prj_over_t0.ply') # This one is the moving one.

    strain_map(original_mesh = mesh_2,
               warped_mesh = final_warped_mesh,
               PATH_output = PATH_output_meshes)
    
    # <== Metrics estimation, ASD, HD after rigid registration, after global INR and after local INR ==> ##

    reference_image = sitk.ReadImage(os.path.join(paths.series_1, "final_masks", "final_mask.nii.gz"))

    fixed_mesh_sitk, fixed_mesh_np = stl_to_filled_mask_like(mesh_1, reference_image)
    moving_mesh_stik, moving_mesh_np = stl_to_filled_mask_like(mesh_2, reference_image)

    global_mesh_sitk, global_mesh_np = stl_to_filled_mask_like(global_warped_mesh, reference_image)
    hybrid_mesh_sitk, hybrid_mesh_np = stl_to_filled_mask_like(final_warped_mesh, reference_image)

    json_metrics = {"ASD_post_registration" : float(compute_asd(fixed_mesh_sitk, moving_mesh_stik)),
                    "HD_post_registration" : float(compute_hd(fixed_mesh_sitk, moving_mesh_stik)),

                    "ASD_global_reconstruction" : float(compute_asd(fixed_mesh_sitk, global_mesh_sitk)),
                    "HD_global_reconstruction" : float(compute_hd(fixed_mesh_sitk, global_mesh_sitk)),

                    "ASD_hybrid_reconstruction" : float(compute_asd(fixed_mesh_sitk, hybrid_mesh_sitk)),
                    "HD_hybrid_reconstruction" : float(compute_hd(fixed_mesh_sitk, hybrid_mesh_sitk))}

    save_json(metadata = json_metrics,
              PATH = os.path.join(paths.output, "metrics_info.json"))


@app.command()
def deformation_analysis(patient: str = typer.Option(..., "-p", "--patient"),
                         moving: str = typer.Option(..., "-m", "--moving"),
                         fixed: str = typer.Option(..., "-f", "--fixed")):

    # This is the ROI to define the effect of the local model.
    # Ideally both ICA -> C1-C4
    roi_config_dict = {'left' : {'target_vessel' : 'internal',
                                 'upper_limit' : 'C1',
                                 'lower_limit' : 'C2'},
                       'right' : {'target_vessel' : 'internal',
                                  'upper_limit' : 'C1',
                                  'lower_limit' : 'C3'}}

    # Mode point_cloud is prefered, multi-anchor is expermiental.
    local_model_config = {'mode' : 'point_cloud',
                          'omega' : 12,
                          'gaussian_sigma' : 0.05}
    
    analysis_config = {"force_training" : True,
                       "roi_config_dict" : roi_config_dict,
                       "local_model_config" : local_model_config}
    
    PATH_input = f'/deepstore/datasets/mia/UMCU_ECAA/D_Output/' # patient/registration/pair_name

    run_deformation_analysis(PATH_input = PATH_input,
                             patient = patient,
                             moving = moving,
                             fixed = fixed,
                             **analysis_config)
    
if __name__ == "__main__":
    app()