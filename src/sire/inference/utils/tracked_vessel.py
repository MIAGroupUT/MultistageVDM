import os

from typing import Any, Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from shapely.geometry import Polygon
from src.sire.utils.affine import get_rotation_matrix, transform_points


class VesselContour:
    def __init__(self, 
                 center: torch.Tensor, 
                 normal: torch.Tensor, points: Dict[str, Any],
                 heatmap: torch.Tensor = None, scale: torch.Tensor = None, flag: int = 0):
        self.center = center
        self.normal = normal
        self.points = points

        # Flagged: if 0 we have a normal contour.
        self.flag = flag # Secret tool for later :)
        self.heatmap = heatmap
        self.scale = scale

    def _make_plane(self, size: int = 128):
        x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="xy")
        z = torch.zeros_like(x)
        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    def _estimate_scale(self):
        '''
        Diameter estimation.
        '''
        return max(
            [2 * torch.linalg.norm(points - self.center, axis=1).max().view(1) for _, points in self.points.items()]
        )
    
    def _estimate_circularity(self, size: int = 128):
        '''
        Estimation of the circularity of the contours using C = (4*pi*area)/perimeter**2
        '''
        try: # The first vessel contour does not have points.
            spacing = size / (2 * self._estimate_scale())
            offset = size / 2
            rot_matrix = get_rotation_matrix(self.normal.cpu()).to(self.center)

            for name, points in self.points.items():
                planar_points = (torch.linalg.inv(rot_matrix).float() @ (points.float() - self.center.float()).T).T
                planar_coords = planar_points[:, :-1] * spacing + offset

            pol = Polygon(planar_coords)
            return (4*np.pi*pol.area)/(pol.length)**2
        except:
            return 1

    def planar_projection(self, image: torch.Tensor, affine: torch.Tensor, size: int = 128):
        planar_points_dict = {}
        scale = self._estimate_scale()
        rot_matrix = get_rotation_matrix(self.normal.cpu()).to(self.center)

        # Compute planar contours
        for name, points in self.points.items():
            planar_points = (torch.linalg.inv(rot_matrix).float() @ (points.float() - self.center.float()).T).T
            spacing = size / (2 * scale)
            planar_points_dict[name] = planar_points[:, :-1] * spacing + (size / 2)

        # Compute planar image
        plane = self._make_plane(size)
        oriented_plane = transform_points(
            (rot_matrix @ (scale.to(self.center) * plane.to(self.center)).T).T + self.center,
            torch.linalg.inv(affine),
        )
        norm_oriented_plane = 2 * (oriented_plane / torch.tensor(image.shape)[[2, 1, 0]][None, ...].to(self.center)) - 1
        planar_img = (
            F.grid_sample(
                image[None, None, ...].float(),
                norm_oriented_plane[None, None, None, ...].float(),
                align_corners=False,
            )
            .squeeze()
            .reshape(size, size)
        )
        return planar_img, planar_points_dict, scale # Image, points and scale.

class TrackedVessel:
    def __init__(self, 
                 image: torch.Tensor, 
                 affine: torch.Tensor, 
                 contours: List[VesselContour] = None,
                 window_length: int = 8):
        
        self.contours = [] if contours is None else self.contours
        self.edges = []
        self.image = image
        self.affine = affine
        self.stitching_index = 0

        # This window wil allow us to check the local values of the contours.
        self.window = []
    
    def check_vessel(self, window_size: int = 0.05):
        def window_diameter(window):
            return np.mean([w._estimate_scale().item()/2 for w in window], axis = 0)
        
        def window_circularity(window):
            return np.mean([w._estimate_circularity() for w in window], axis = 0)

        # valid_contours = [contour for contour in self.contours if len(contour.points) != 0]
        valid_indices = [i for i, contour in enumerate(self.contours) if len(contour.points) != 0]

        if not valid_indices:
            return
        
        for idx in valid_indices:
            #self.contours[idx]._estimate_scale().item()/2 # Diameter.
            if self.contours[idx]._estimate_circularity() < 0.95: # Circularity.
                self.contours[idx].flag = 2

    def scrap_last(self):
        self.contours = self.contours[:-1]
        self.edges = self.edges[:-1]

    def update(self, contour: VesselContour):
        if len(self.contours) > 0:
            self.edges.append([len(self.contours) - 1, len(self.contours)])
        
        # Flag indicates the stitching point.
        self.contours.append(contour)

    def get_heatmaps(self):
        
        # Creates a matrix that contains the heatmaps of all of the contours.
        heatmap_matrix = [contour.heatmap.cpu().numpy() for contour in self.contours if contour.points]
        full_matrix = np.array(heatmap_matrix, dtype=np.float32)
        final_matrix = full_matrix.squeeze()
    
        return final_matrix
    
    def get_scales(self):
        
        # (Possibly) gets the selected scale (weighted r) for each contour
        scale_vector = [contour.scale.cpu().numpy() for contour in self.contours if contour.points]
        full_scales = np.array(scale_vector, dtype=np.float32)
        final_scales = full_scales.squeeze()
    
        return final_scales

    def get_centers(self):

        return np.stack([contour.center.cpu().numpy() for contour in self.contours])

    def save_centers(self):
        np.save('saved_centers.npy', np.stack([contour.center.cpu().numpy() for contour in self.contours]))

    def build_centerline(self):
        centers = np.stack([contour.center.cpu().numpy() for contour in self.contours])
        edges = np.array(self.edges)

        flat_edges = np.c_[2 * np.ones(len(edges))[:, None], edges].flatten().astype(int)

        return pv.PolyData(centers, lines=flat_edges)

    def build_contours(self):
        point_dict = {}
        poly_contours = {}

        # Aggregate contours
        for contour in self.contours:
            for name, points in contour.points.items():
                if name not in point_dict.keys():
                    point_dict[name] = []

                point_dict[name].append(points)

        # Build polydata
        for name, points in point_dict.items():
            points = np.stack(points)
            num_contours, num_points, _ = points.shape

            print("num contours and num points",num_contours,num_points)

            contour_lines = np.array([[i, (i + 1) % num_points] for i in range(num_points)])
            all_contour_lines = np.concatenate([contour_lines + i * num_points for i in range(num_contours)])
            flat_lines = np.c_[2 * np.ones(len(all_contour_lines))[:, None], all_contour_lines].flatten().astype(int)

            poly_contour = pv.PolyData(points.reshape(-1, 3), lines=flat_lines)
            poly_contours[name] = poly_contour

        return poly_contours
    
    def save_diameters(self, output_dir: str):
        proper_contours = [contour for contour in self.contours if len(contour.points.keys()) != 0]
        diameters = []

        for i, contour in enumerate(proper_contours):
            _, _, scale = contour.planar_projection(self.image, self.affine)
            diameters.append(scale.item()/2)

        np.savetxt(os.path.join(output_dir, 'diameters.txt'), diameters, fmt='%f')

    def save_planar_projections(self, output_dir: str):
        proper_contours = [contour for contour in self.contours if len(contour.points.keys()) != 0]

        for i, contour in enumerate(proper_contours):
            planar_img, planar_points_dict, scale = contour.planar_projection(self.image, self.affine)

            fig, ax = plt.subplots()
            ax.imshow(planar_img.cpu().numpy(), cmap="gray") # Just the image.

            # Plot all the contours
            for j, (name, planar_points) in enumerate(planar_points_dict.items()): # Change to adapt to other contours
                closed_points = torch.cat([planar_points, planar_points[0][None]])

                # * is to plot x,y instead of using the index.
                if contour.flag != 0:
                    ax.plot(*closed_points.cpu().numpy().T, color=list(mcolors.TABLEAU_COLORS.values())[1], label=name) # Orange (not normal)
                else:
                    ax.plot(*closed_points.cpu().numpy().T, color=list(mcolors.TABLEAU_COLORS.values())[0], label=name) # Blue (normal)

            pol = Polygon(closed_points.cpu().numpy()) 
            ax.set_title(f"Max diameter: {(scale.item() / 2):.2f} mm, {(4*np.pi*pol.area)/(pol.length)**2:.3f}")
            ax.axis("off")

            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"planar_contour_{i}.png"))
            plt.close(fig)

    def merge_at_start(self, tracked_vessel):

        contours = tracked_vessel.contours[1:][::-1] # From the other direction
        self.stitching_index = len(contours) - 1
        print("Stitching index at: {}".format(self.stitching_index))

        self.contours = contours + self.contours
        edges = [[i, i + 1] for i in range(len(self.contours) - 1)]
        self.edges = edges
