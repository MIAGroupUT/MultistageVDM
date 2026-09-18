import os
import vtk
import trimesh
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from vtk.util import numpy_support
# from vtk.util.numpy_support import vtk_to_numpy

def dice_score(pred, true):
    intersection = np.sum(pred * true)
    return (2. * intersection) / (np.sum(pred) + np.sum(true))

def compute_hd(sitkmask1, sitkmask2):
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(sitkmask1, sitkmask2)

    hd = hd_filter.GetHausdorffDistance()

    return hd

def compute_asd(mask1, mask2):
    d1 = sitk.SignedMaurerDistanceMap(mask1, squaredDistance=False)
    d2 = sitk.SignedMaurerDistanceMap(mask2, squaredDistance=False)

    s1 = sitk.LabelContour(mask1)
    s2 = sitk.LabelContour(mask2)

    d1_arr = sitk.GetArrayFromImage(d1)
    d2_arr = sitk.GetArrayFromImage(d2)
    s1_arr = sitk.GetArrayFromImage(s1)
    s2_arr = sitk.GetArrayFromImage(s2)

    dist1 = np.abs(d2_arr[s1_arr == 1])
    dist2 = np.abs(d1_arr[s2_arr == 1])

    return np.mean(np.concatenate([dist1, dist2]))

def trimesh_to_vtk(mesh: trimesh.Trimesh):

    vtk_points = vtk.vtkPoints()

    for p in mesh.vertices:
        vtk_points.InsertNextPoint(
            float(p[0]),
            float(p[1]),
            float(p[2])
        )

    vtk_cells = vtk.vtkCellArray()

    for face in mesh.faces:
        triangle = vtk.vtkTriangle()

        triangle.GetPointIds().SetId(0, int(face[0]))
        triangle.GetPointIds().SetId(1, int(face[1]))
        triangle.GetPointIds().SetId(2, int(face[2]))

        vtk_cells.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    return polydata

def stl_to_filled_mask_like(stl_path, reference_sitk):

    size_x, size_y, size_z = reference_sitk.GetSize()

    mesh = pv.read(stl_path)
    mesh = mesh.clean().triangulate()

    # mesh = mesh.fill_holes(1000.0)

    points_world = np.asarray(mesh.points)

    points_idx = np.array([reference_sitk.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in p)
        )for p in points_world
    ])

    # Replace mesh points by index-space points
    mesh_idx = mesh.copy()
    mesh_idx.points = points_idx

    polydata = mesh_idx
    white_image = vtk.vtkImageData()
    white_image.SetSpacing(1, 1, 1)
    white_image.SetOrigin(0, 0, 0)
    white_image.SetDimensions(size_x, size_y, size_z)
    white_image.SetExtent(0, size_x - 1, 0, size_y - 1, 0, size_z - 1)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    arr = numpy_support.vtk_to_numpy(
        white_image.GetPointData().GetScalars()
    )
    arr[:] = 1

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(0, 0, 0)
    pol2stenc.SetOutputSpacing(1, 1, 1)
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    vtk_mask = imgstenc.GetOutput()
    vtk_array = numpy_support.vtk_to_numpy(
        vtk_mask.GetPointData().GetScalars()
    )

    # VTK is x, y, z; SimpleITK numpy is z, y, x
    np_mask = vtk_array.reshape((size_z, size_y, size_x))

    sitk_mask = sitk.GetImageFromArray(np_mask.astype(np.uint8))
    sitk_mask.CopyInformation(reference_sitk)

    return sitk_mask, np_mask