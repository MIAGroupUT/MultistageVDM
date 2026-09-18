
import os
import vtk
import glob
import time
import torch

import trimesh
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torch import nn
from torch_cluster import knn
from typing import Dict, List

from scipy.spatial.distance import cdist
from matplotlib.colors import TwoSlopeNorm
from src.longitudinal_analysis.utils.utils_image_registration import Roi

## <=== Usual vessels that we should have ===> ##

vessels =  ['external_carotid_artery_left',
            'internal_carotid_artery_left',
            'external_carotid_artery_right',
            'internal_carotid_artery_right']

## <=== Utils for loading SIRE outputs and generated matched meshes ===> ##

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
    
    centerline_array = centerline_array.astype(np.float32)

    return centerline_array

def load_contours(PATH_contours: str):

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

    return np.stack(contours)

def join_contours(dict_contours: str):

    joined = []
    for vessel in dict_contours.keys():
        pts = dict_contours[vessel].reshape(-1, 3)
        joined.append(pts)

    return np.vstack(joined)

def read_meshes(moving : str,
                fixed : str):

    # PATH_matched_series = os.path.join(PATH_input, "registration", "{}__to__{}".format(moving, fixed), "series_matching")

    # Training of the hybrid VVF.
    path_matched_mesh_1 = os.path.join(fixed, "final_meshes", "vessels_fused_final.stl")
    path_matched_mesh_2 = os.path.join(moving, "final_meshes", "vessels_fused_final.stl")

    # Loading the meshes.
    matched_mesh_1 = trimesh.load(path_matched_mesh_1)
    matched_mesh_2 = trimesh.load(path_matched_mesh_2)

    return matched_mesh_1, matched_mesh_2

def apply_heatmap_and_save(mesh: trimesh.Trimesh,
                           heatmap: np.ndarray,
                           PATH_ouput: str,
                           filename: str = "colored_output.ply"):
    
    mesh.visual.vertex_colors = heatmap
    mesh.export(os.path.join(PATH_ouput, filename))

## <== Training utils ===> ##

def chamfer_distance_torch(x, y, bidirectional=True, squared=True):
    # x: [B, N, 3]
    # y: [B, M, 3]

    dist = torch.cdist(x, y)  # [B, N, M]

    if squared:
        dist = dist ** 2

    x_to_y = dist.min(dim=2)[0].mean()

    if not bidirectional:
        return x_to_y

    y_to_x = dist.min(dim=1)[0].mean()

    return x_to_y + y_to_x

def chamfer_distance_torch_match_old(x, y):
    # x: [B, N, 3]
    # y: [B, M, 3]

    dist = torch.cdist(x, y) ** 2   # squared distances

    # forward only: x -> y
    x_to_y = dist.min(dim=2)[0]     # [B, N]
    # point_reduction="sum"
    x_to_y = x_to_y.sum(dim=1)      # [B]
    # batch_reduction="mean"
    return x_to_y.mean()

def normalize_points(points, bbox_min, bbox_max):
    points_norm = 2.0 * (points - bbox_min) / (bbox_max - bbox_min) - 1.0

    return points_norm.astype(np.float32)

def denormalize_points(points_norm, bbox_min, bbox_max):
    points = (
        (points_norm + 1.0) * 0.5 * (bbox_max - bbox_min)
    ) + bbox_min

    return points.astype(np.float32)

def gaussian_weight(cloud_point,
                    point,
                    sigma: int = 5):

    d = -(np.min(np.linalg.norm(cloud_point - point, axis = 1))**2)
    return np.exp(d/(2*(sigma**2)))

def gaussian_weight_to_anchor_torch(anchor_coords,
                                    sigma,
                                    p):

    distances = torch.cdist(p, anchor_coords.unsqueeze(0))
    sigma = sigma.view(1, 1, -1)

    # Gaussian weights
    # weights = torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    # w, _ = torch.max(weights, dim=-1)

    return torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))

def gaussian_weight_to_multianchor_torch(anchors_info, p):

    anchors_coords = anchors_info['anchors']
    anchors_sigma = anchors_info['sigmas']

    distances = torch.cdist(p, anchors_coords.unsqueeze(0))
    anchors_sigma = anchors_sigma.view(1, 1, -1)

    # Gaussian weights
    weights = torch.exp(-(distances ** 2) / (2.0 * anchors_sigma ** 2))
    w, _ = torch.max(weights, dim=-1)

    return w

def gaussian_weight_torch(roi_cloud,
                          p,
                          sigma: int = 5):
    
    # Slower version.
    # distances = torch.cdist(p, roi_cloud.unsqueeze(0))
    # min_distances = distances.min(dim=-1).values
    # w = torch.exp(-(min_distances[..., None] ** 2) / (2 * sigma ** 2))

    # Possibly faster, solve issues.
    p = p.reshape(-1, 3).cpu()
    roi_cloud = roi_cloud.reshape(-1, 3).cpu()

    _, idx_roi = knn(roi_cloud, p, k=1)
    
    nearest = roi_cloud[idx_roi]
    dist2 = ((p - nearest)**2).sum(dim = -1)
    w = torch.exp(-(dist2) / (2 * sigma ** 2))[None, :, None]

    return w

# def global_inference(global_ode_func : VelocityODE,
#                      original_mesh : trimesh.Trimesh, 
#                      bbox_min, 
#                      bbox_max,
#                      device : torch.device,
#                      PATH_output : str,
#                      t_span : torch.Tensor,
#                      filename : str = "output.stl") -> trimesh.Trimesh:

# def inference(global_ode_func : VelocityODE,
#               local_ode_func : LocalODE,
#               original_mesh, 
#               bbox_min, 
#               bbox_max,
#               PATH_output : str,
#               t_span : torch.Tensor,
#               device : torch.device,
#               filename : str = "output.stl") -> trimesh.Trimesh:

def get_bbox(matched_mesh_1 : trimesh.Geometry,
             matched_mesh_2 : trimesh.Geometry):
    
    all_vertices = np.vstack([matched_mesh_1.vertices, matched_mesh_2.vertices])
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)

    return bbox_min, bbox_max

def define_time_span(device : torch.device,
                     n_steps = 6) -> torch.Tensor:

    return torch.as_tensor(np.linspace(0.0, 1.0, n_steps), dtype = torch.float32, device = device)

def define_local_inr_roi(PATH_matched_series : str,
                         config_dict : dict):
    
    total_points = []
    for key, item in config_dict.items():
        roi_selected_points = local_inr_roi_definition(PATH_matched_series = PATH_matched_series,
                                                       side = key,
                                                       target_vessel = item['target_vessel'],
                                                       upper_limit = item['upper_limit'],
                                                       lower_limit = item['lower_limit'])
        total_points.append(roi_selected_points)

    return np.vstack(total_points)

def local_inr_roi_definition(PATH_matched_series : str,
                             side: str,
                             target_vessel : str,
                             upper_limit : str,
                             lower_limit : str):
    
    vessels =  ['external_carotid_artery_left',
                'internal_carotid_artery_left',
                'external_carotid_artery_right',
                'internal_carotid_artery_right']

    # First we reed the contours since we are basing ourselves on this to generate the w gaussian weight.
    PATH_contour_series_2 = os.path.join(PATH_matched_series, "series_2_matched", "final_contours")
    PATH_mask_series_2 = os.path.join(PATH_matched_series, "series_2_matched", "final_masks", "final_mask.nii.gz")

    contours_s2 = {vessel: 
                load_contours(os.path.join(PATH_contour_series_2, f"contour_{vessel}.vtp")) for vessel in vessels if side in vessel and target_vessel in vessel}

    if len(contours_s2.keys()) == 0:
        print("No vessel fullfiled the criteria.")
        return 
    
    mask_img = sitk.ReadImage(PATH_mask_series_2)
    mask_np = sitk.GetArrayFromImage(mask_img)  # z, y, x

    joined_contours_s2 = join_contours(contours_s2)

    labels = {"C1" : 50, "C2" : 49, "C3" : 48, "C4" : 47, "C5" : 46, "C6" : 45, "C7" : 44,
              "T1" : 43, "T2" : 42}
        
    upper_label = labels.get(upper_limit, None)
    lower_label = labels.get(lower_limit, None)

    if upper_label is None or lower_label is None:
        print("No valid limits were provided.")
        return None

    roi_list = []
    if np.any(mask_np == upper_label):
        roi_list.append(Roi([upper_label], "below", "max"))
    else:
        print(f"Upper label {upper_limit} not found in mask.")

    if np.any(mask_np == lower_label):
        roi_list.append(Roi([lower_label], "above", "min"))
    else:
        print(f"Lower label {lower_limit} not found in mask.")

    if len(roi_list) == 0:
        print("No valid ROI could be created.")
        return None

    roi_local_inr = Roi.intersection(mask = mask_np, 
                                     roi_list = roi_list).astype(int)
    roi_image = sitk.GetImageFromArray(roi_local_inr)
    roi_image.CopyInformation(mask_img)

    selected_points = []
    size_x, size_y, size_z = mask_img.GetSize()

    for point in joined_contours_s2:
        try:
            ix, iy, iz = mask_img.TransformPhysicalPointToIndex(tuple(float(v) for v in point))
        except RuntimeError:
            continue

        if not (0 <= ix < size_x and 0 <= iy < size_y and 0 <= iz < size_z):
            continue

        if roi_local_inr[iz, iy, ix]:
            selected_points.append(point)

    if len(selected_points) == 0:
        print("No contour points were found inside the requested ROI.")
        return None

    selected_points = np.asarray(selected_points, dtype=np.float32)

    print(f"Selected {len(selected_points)} / {len(joined_contours_s2)} points "
        f"for {target_vessel} carotid artery on {side} side between {upper_limit}-{lower_limit}."
    )

    return selected_points

def strain_map(original_mesh: trimesh.Trimesh, 
               warped_mesh: trimesh.Trimesh,
               PATH_output: str,
               file_name = "strain_map.ply"):

    A_f = warped_mesh.area_faces
    A_0 = original_mesh.area_faces

    if np.shape(A_f) != np.shape(A_0):
        return

    epsilon = 1e-8
    A_strain = np.where(A_0 > epsilon, (A_f - A_0) / A_0, 0.0)

    # Creation of the colormap that can be added to the meshes.
    vmax = np.percentile(np.abs(A_strain), 98)
    vmax = max(vmax, 1e-8)
    norm = TwoSlopeNorm(vmin = -vmax,
                        vcenter = 0.0,
                        vmax = vmax)

    cmap = plt.get_cmap("Spectral")
    colors = cmap(norm(A_strain))
    colors = (colors * 255).astype(np.uint8)
    colors[:, 3] = 255

    warped_mesh.visual.face_colors = colors
    warped_mesh.export(os.path.join(PATH_output, file_name))


## <== Utils to obtain deformation analysis results ==> ##

def deformation_field(original_mesh : trimesh.Trimesh,
                      warped_mesh : trimesh.Trimesh,
                      PATH_save,
                      output_file_name = "deformation_field.vtp",
                      mode = "all",
                      sample_ratio = 0.05,
                      threshold = 0.0,
                      scale_factor = 1.0,
                      use_fixed_max_displacement = True) -> np.ndarray:
    
    '''
    Returns
        heatmap: Returns the deformation field as a np.array with shape (N, 4), using the coolwarm 
        cmap from matplotlib, this colormap/heatmap can be used to color any of the meshes.

        normal_disp: Dot product between the displacement vector and vertex normals of the original_mesh,
        these values have been already corrected since for some meshes, the +normals point towards the 
        inside of the mesh.
    '''
    warped_vertices = warped_mesh.vertices
    original_vertices = original_mesh.vertices
    normals = original_mesh.vertex_normals # A vector.

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8) # Making sure these are actually normalised, should be the case tho.
    u = warped_vertices - original_vertices # Whole displacement vector.

    normal_disp = np.sum(u * normals, axis=1) # Dot product
    u_normal = normal_disp[:, None] * normals # Scalar * scaled normal
    u_tangent = u - u_normal

    # Orentation score, since sometimes the normals seem to be positive if pointing inwards.
    center = warped_mesh.centroid
    radial = original_vertices - center
    radial = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8)
    orientation_score = np.mean(np.sum(radial * normals, axis=1))

    if orientation_score < 0:
        normals = -normals # So the colors won't be inverted.
        normal_disp = np.sum(u * normals, axis=1) # Re-computing JUST to invert the colors

    if mode == "all":
        vectors = u

    elif mode == "normal":
        vectors = u_normal

    elif mode == "tangent":
        vectors = u_tangent

    else:
        raise ValueError("mode must be: 'all', 'normal', or 'tangent'")

    magnitudes = np.linalg.norm(vectors, axis=1)
    mask = magnitudes > threshold

    points_filtered = original_vertices[mask] # This proves the vertices are the same index.

    vectors_filtered = vectors[mask]
    n_points = len(points_filtered)
    n_sample = max(1, int(sample_ratio * n_points))

    idx = np.random.choice(n_points,
                           size = n_sample,
                           replace=False)
    points_sample = points_filtered[idx]
    vectors_sample = vectors_filtered[idx]

    vtk_points = vtk.vtkPoints()

    vtk_vectors = vtk.vtkFloatArray()
    vtk_vectors.SetNumberOfComponents(3)
    vtk_vectors.SetName("vectors")

    for p, v in zip(points_sample, vectors_sample):
        vtk_points.InsertNextPoint(p.tolist())
        vtk_vectors.InsertNextTuple(v.tolist())

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.GetPointData().SetVectors(vtk_vectors)

    arrow = vtk.vtkArrowSource()

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(polydata)

    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()

    glyph.SetScaleFactor(scale_factor)

    glyph.OrientOn()
    glyph.Update()

    writer = vtk.vtkXMLPolyDataWriter()

    save_path = os.path.join(PATH_save, output_file_name)

    writer.SetFileName(save_path)
    writer.SetInputData(glyph.GetOutput())
    writer.Write()

    print(f"Saved deformation field -> {save_path}")

    if use_fixed_max_displacement:
        norm = TwoSlopeNorm(vmin = -2.0,
                            vcenter = 0.0,
                            vmax = 2.0)
    
    else: # This might make the red redder.
        vmax = np.percentile(np.abs(normal_disp), 98)
        vmax = max(vmax, 1e-8)
        norm = TwoSlopeNorm(vmin = -vmax,
                            vcenter = 0.0,
                            vmax = vmax)

    cmap = plt.get_cmap("coolwarm")
    colors = cmap(norm(normal_disp))
    colors = (colors * 255).astype(np.uint8)
    colors[:, 3] = 255

    return colors, normal_disp

## <== Basic classes for training and definition of de ODE ==> ##

class SIREN(nn.Module):
    def __init__(self, layers, 
                 weight_init=True, 
                 scale = 1,
                 omega = 30):
        """Initialize the network."""

        super(SIREN, self).__init__()

        self.n_layers = len(layers) - 1
        self.scale = scale
        self.omega = omega

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                    else:
                        self.layers[-1].weight.uniform_(
                            -np.sqrt(6 / layers[i]) / self.omega, np.sqrt(6 / layers[i]) / self.omega
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords):
        x = coords
        for layer in self.layers[:-1]: 
            x = torch.sin(self.omega * layer(x)) # Manual activation per layer.

        displacement = self.layers[-1](x) # This one is linear since we want dx, dy, dz, therefore no sin is needed.
        return self.scale * displacement  # Scaled output.

class SIREN_VVF(SIREN):

    def __init__(self, layers, weight_init=True, scale=1, omega=30):
        super().__init__(layers, weight_init, scale, omega)

    def forward(self, coords) -> torch.Tensor:
        x = coords
        for layer in self.layers[:-1]: 
            x = torch.sin(self.omega * layer(x)) # Manual activation per layer.

        displacement = self.layers[-1](x) # This one is linear 
        return displacement  # Output
    
class LocalSIRENVVF(SIREN):
    def __init__(self, layers, anchor: np.ndarray, sigma: float,
                 weight_init=True, scale=1, omega=30):
        super().__init__(layers, weight_init, scale, omega)
        self.anchor = anchor # Coords where the INR is centered.
        self.sigma = sigma # Estimated as 0.5*distance_to_closest_anchor

    def forward(self, coords) -> torch.Tensor:
        x = coords
        for layer in self.layers[:-1]: 
            x = torch.sin(self.omega * layer(x)) # Manual activation per layer.

        displacement = self.layers[-1](x) # This one is linear 
        return displacement  # Output
    
class VelocityODE:
    def __init__(self, model):
        self.model = model
        self.no_grad = False

    def __call__(self, t, p):

        if self.no_grad:
            with torch.no_grad():
                return self.model(p)

        else:
            return self.model(p)
    
    def freeze_model(self):
        self.no_grad = True

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        self.no_grad = False

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

class LocalODE:
    def __init__(self,
                 roi_cloud,
                 local_model : SIREN_VVF,
                 device : torch.device,
                 sigma = 5):
        
        self.local_model = local_model
        self.roi_cloud = roi_cloud
        self.device = device
        self.sigma = sigma

    def __call__(self, t, p):
        v_local = self.local_model(p)
        w = gaussian_weight_torch(self.roi_cloud, p, sigma = self.sigma)
        w = w.to(device = self.device)

        return w * v_local

class MiniLocalHybridODE:
    def __init__(self,
                 global_model: SIREN_VVF,
                 local_models: List[LocalSIRENVVF],
                 device: torch.device,
                 top_k: int = 2):

        self.already_trained = []
        self.model_in_training = 0

        self.global_model = global_model
        self.local_models = local_models
        self.device = device
        self.top_k = top_k

        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False

    def freeze_every_local_model_but(self, idx: int):
        self.model_in_training = idx

        for i in range(len(self.local_models)):
            requires_grad = (i == idx)

            for param in self.local_models[i].parameters():
                param.requires_grad = requires_grad

    def mark_model_as_trained(self, idx):
        if idx not in self.already_trained:
            self.already_trained.append(idx)

    def __call__(self, t, p):

        with torch.no_grad():
            v_total = self.global_model(p)

        candidate_ids = self.already_trained + [self.model_in_training]

        fields = []
        weights = []

        for i in candidate_ids:
            w_i = gaussian_weight_to_anchor_torch(self.local_models[i].anchor,
                                                  self.local_models[i].sigma,
                                                  p=p)  # [B, N, 1]

            if i == self.model_in_training:
                v_i = self.local_models[i](p)
            else:
                with torch.no_grad():
                    v_i = self.local_models[i](p)

            fields.append(v_i)
            weights.append(w_i)

        # [B, N, K]
        weights = torch.cat(weights, dim=-1)

        # [B, N, K, 3]
        fields = torch.stack(fields, dim=2)
        k = min(self.top_k, weights.shape[-1])

        # top_values: [B, N, k]
        # top_indices: [B, N, k]
        top_values, top_indices = torch.topk(weights,k=k,dim=-1)

        # Normalize only top-k weights
        top_weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather selected fields
        # Need indices shape [B, N, k, 3]
        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, -1, 3)

        selected_fields = torch.gather(fields,
                                       dim=2,
                                       index=gather_idx)  # [B, N, k, 3]
        local_total = torch.sum(top_weights.unsqueeze(-1) * selected_fields, dim=2)  # [B, N, 3]

        return v_total + local_total

# class MiniLocalHybridODE:
#     def __init__(self,
#                  global_model: SIREN_VVF,
#                  local_models: List[LocalSIRENVVF],
#                  device: torch.device):

#         self.already_trained = []
#         self.model_in_training = 0

#         self.global_model = global_model
#         self.local_models = local_models
#         self.device = device

#         self.global_model.eval()
#         for param in self.global_model.parameters():
#             param.requires_grad = False

#     def freeze_every_local_model_but(self, idx: int): # Idx of the model that should not be frozen.
#         self.model_in_training = idx
#         for i in range(0, len(self.local_models)):
#             if i == idx:
#                 for param in self.local_models[i].parameters():
#                     param.requires_grad = True        
#             else:
#                 for param in self.local_models[i].parameters():
#                     param.requires_grad = False

#     def mark_model_as_trained(self, idx):
#         if idx not in self.already_trained:
#             self.already_trained.append(idx)

#     def __call__(self, t, p):

#         with torch.no_grad():
#             v_total = self.global_model(p)

#         candidate_ids = self.already_trained + [self.model_in_training]

#         weighted_fields = []
#         weights = []

#         for i in candidate_ids:
#             w_i = gaussian_weight_to_anchor_torch(self.local_models[i].anchor,
#                                                   self.local_models[i].sigma,
#                                                   p=p)

#             if i == self.model_in_training:
#                 v_i = self.local_models[i](p)
#             else:
#                 with torch.no_grad():
#                     v_i = self.local_models[i](p)

#             weighted_fields.append(w_i * v_i)
#             weights.append(w_i)

#         weights = torch.cat(weights, dim=-1)
#         winner = torch.argmax(weights, dim=-1)

#         local_total = torch.zeros_like(v_total)

#         for k, field in enumerate(weighted_fields):
#             mask = (winner == k).unsqueeze(-1)
#             local_total = local_total + mask * field

#         return v_total + local_total

class HybridVelocityODE:
    def __init__(self,
                 roi_cloud,
                 global_model : SIREN_VVF,
                 local_model : SIREN_VVF,
                 device : torch.device,
                 sigma = 0.1): # Can be expanded to use a list or dictionary.
        
        self.global_model = global_model
        self.local_model = local_model
        self.roi_cloud = roi_cloud
        self.device = device
        self.sigma = sigma

        # First we freeze the global model.
        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False

    def __call__(self, t, p):

        with torch.no_grad():
            v_global = self.global_model(p)

        v_local = self.local_model(p)
        w = gaussian_weight_torch(self.roi_cloud, p, sigma = self.sigma)
        # w = torch.where(w > 0.1, w, torch.zeros_like(w))
        w = w.to(device = self.device)
  
        return v_global + w * v_local
    
class MultiGaussVelocityODE:
    def __init__(self,
                 global_model: SIREN_VVF,
                 local_model: SIREN_VVF,
                 anchors_info: dict,
                 device: torch.device):
        
        self.global_model = global_model
        self.local_model = local_model
        self.anchors_info = anchors_info
        self.device = device

        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False

    def __call__(self, t, p):
        with torch.no_grad():
            v_global = self.global_model(p)

        v_local = self.local_model(p)
        w = gaussian_weight_to_multianchor_torch(anchors_info = self.anchors_info,
                                                 p = p)

        return v_global + w * v_local

class MultiLocalINR:
    def __init__(self,
                 centerlines : Dict[str, np.array]):
        
        self.centerlines = {name: centerline for name, centerline in centerlines.items()}

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

    def detect_bifurcation(self,
                           vessel_1 : str,
                           vessel_2 : str,
                           merge_tol : float):
        
        point_1, _ = self._get_merge_point(self.centerlines[vessel_1],
                                           self.centerlines[vessel_2],
                                           tol = merge_tol)

        point_2, _ = self._get_merge_point(self.centerlines[vessel_2],
                                           self.centerlines[vessel_1],
                                           tol = merge_tol)
        bifurcation = 0.5 * (point_1 + point_2)

        return bifurcation, point_1, point_2
        
    def identify_common_trunk(self,
                              part_a: np.ndarray,
                              part_b: np.ndarray,
                              other_centerline: np.ndarray,
                              tol: float = 1.5):
        def overlap_score(part, other, tol):
            D = cdist(part, other)
            min_dist = D.min(axis=1)
            return np.sum(min_dist < tol)
        
        # Which is not necessarily the CCA since the code sometimes fails.

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
        def split_by_bifurcation(centerline, bif_point):
            distances = np.linalg.norm(centerline - bif_point, axis=1)
            idx = np.argmin(distances)

            part1 = centerline[:idx+1]
            part2 = centerline[idx:]
            return part1, part2, idx
        
        # Initialization of the dict.
        results = {}
        part1_ica, part2_ica, _ = split_by_bifurcation(self.centerlines[ica_cca], bif_point)
        part1_eca, part2_eca, _ = split_by_bifurcation(self.centerlines[eca_cca], bif_point)

        cca_from_ica, ica_part, s1, s2 = self.identify_common_trunk(
            part1_ica, part2_ica, self.centerlines[eca_cca], tol=1.5
        )

        cca_from_eca, eca_part, s3, s4 = self.identify_common_trunk(
            part1_eca, part2_eca, self.centerlines[ica_cca], tol=1.5
        )
        
        # These labels might not be accurate, again, it is possible that the tracker makes a U-turn
        # starts tracking the ECA and ends in the ICA.
        results.update({"ICA": ica_part,
                        "CCA_from_ICA" : cca_from_ica,
                        "ECA": eca_part,
                        "CCA_from_ECA": cca_from_eca,
                        # "CCA": self.build_common_cca(cca_from_eca, cca_from_ica, 1, True),                            
                        "bifurcation_point" : bif_point})
        
        return results

    def _get_carotid_trunk(self):

        # First the left side, computing the bif point between the internal and external.
        # If we have a distal section it is important to merge it with its correspondant side before.
        # However I believe is better do it before this.
        bif_left, _, _ = self.detect_bifurcation(vessel_1 = "external_carotid_artery_left",
                                                 vessel_2 = "internal_carotid_artery_left",
                                                 merge_tol = 2)
        
        results_left = self.split_carotid_branches(ica_cca = "internal_carotid_artery_left",
                                                   eca_cca = "external_carotid_artery_left",
                                                   bif_point = bif_left)
        
        bif_right, _, _ = self.detect_bifurcation(vessel_1 = "external_carotid_artery_right",
                                                  vessel_2 = "internal_carotid_artery_right",
                                                  merge_tol = 2)
        
        results_right = self.split_carotid_branches(ica_cca = "internal_carotid_artery_right",
                                                    eca_cca = "external_carotid_artery_right",
                                                    bif_point = bif_right)

        return results_left, results_right
    
    def _cumulative_arclength(self,
                              curve: np.ndarray):
        diffs = np.diff(curve, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0.0], np.cumsum(dists)])

    # Maybe start from the bif point and move outwards, the one labeled as CC will have the first
    # INR at the closest point to the bif point.
    def define_local_INRs_locations(self,
                                    device: torch.device,
                                    bbox_min: np.ndarray,
                                    bbox_max: np.ndarray,
                                    distance_between_points: float = 35):
        '''
        Args:
            distance_between_points: In mm, a new point is positioned every N mm for all the centerlines.
        '''
        def find_locations(vessel_centerline : np.ndarray,
                           bif_point : np.ndarray) -> np.ndarray:
            
            distance_to_bif_point = np.linalg.norm(vessel_centerline - bif_point, axis = 1)
            idx = np.argmin(distance_to_bif_point)

            if idx > len(vessel_centerline)//2:
                vessel_centerline = vessel_centerline[::-1]

            cumarclength = self._cumulative_arclength(curve = vessel_centerline)
            targets = distance_between_points * np.arange(1, 20) # Contains the maximum distances.

            valid = targets <= cumarclength[-1] # Only valid targets.
            targets = targets[valid]
            idxs = np.searchsorted(cumarclength, targets) # Finding the index of every max value.

            return vessel_centerline[idxs]
                
        def compute_sigmas(anchors):
            sigmas = []
            for i in range(0, len(anchors)):
                second_smallest = np.partition(np.linalg.norm(anchors - anchors[i], axis = 1), 1)[1]
                sigmas.append(0.5*second_smallest)

            return sigmas

        results_left, results_right = self._get_carotid_trunk()
    
        bif_point_left = results_left['bifurcation_point']
        bif_point_right = results_right['bifurcation_point']

        locations_ica_left = find_locations(results_left['ICA'], bif_point = bif_point_left)
        locations_eca_left = find_locations(results_left['ECA'], bif_point = bif_point_left)
        locations_ica_cca_left = find_locations(results_left['CCA_from_ICA'], bif_point = bif_point_left)

        all_points_left = np.vstack([[bif_point_left], locations_ica_left, locations_eca_left, locations_ica_cca_left])
        # sigmas_left = compute_sigmas(all_points_left)

        locations_ica_right = find_locations(results_right['ICA'], bif_point = bif_point_right)
        locations_eca_right = find_locations(results_right['ECA'], bif_point = bif_point_right)
        locations_ica_cca_right = find_locations(results_right['CCA_from_ICA'], bif_point = bif_point_right)

        all_points_right = np.vstack([[bif_point_right], locations_ica_right, locations_eca_right, locations_ica_cca_right])
        # sigmas_right = compute_sigmas(all_points_right)

        all_points = np.vstack([all_points_right, all_points_left])
        # all_sigmas = np.hstack([sigmas_right, sigmas_left])

        order = np.argsort(all_points[:, 2])  # z world coordinate
        all_points = all_points[order]
        # all_sigmas = all_sigmas[order]

        locations = {}
        normalized_anchors = normalize_points(all_points, bbox_min, bbox_max)

        locations['anchors'] = torch.as_tensor(normalized_anchors, 
                                               dtype = torch.float32, 
                                               device = device)

        locations['sigmas'] = torch.as_tensor(compute_sigmas(normalized_anchors),
                                              dtype = torch.float32,
                                              device = device)
        
        print("Sigmas for the anchors:", compute_sigmas(normalized_anchors))
        self.locations = locations
        
        return locations
    
    def setup_local_INRs(self,
                         device: torch.device,
                         global_model: SIREN_VVF,
                         layers = [3, 256, 256, 256, 3],
                         omega: int = 15):

        local_models = []
        local_models_optimizers = []
        n = len(self.locations['anchors'])
        for i in range(0, n):
            local_model_i = LocalSIRENVVF(layers = layers,
                                          anchor = self.locations['anchors'][i],
                                          sigma = self.locations['sigmas'][i],
                                          omega = omega).to(device)
            optimizer_i = torch.optim.Adam(local_model_i.parameters(), lr = 1e-4)

            local_models.append(local_model_i)
            local_models_optimizers.append(optimizer_i)

        mini_ode_func = MiniLocalHybridODE(global_model = global_model,
                                           local_models = local_models,
                                           device = device)
        
        return mini_ode_func, local_models_optimizers


    ## <== Method to load the centerlines from the directory ==> ##

    @classmethod
    def load_from_directory(cls,
                            root_dir_centerlines: str,
                            filenames: List[str],
                            extension: str = ".vtp"):
        
        centerlines = {}
        for filename in filenames:
            try:
                centerlines[filename] = pv.read(glob.glob(os.path.join(root_dir_centerlines, f"*{filename}*{extension}"))[0]).points
            except:
                print(f"Centerline not found for vessel: {filename}")

        return cls(centerlines)
