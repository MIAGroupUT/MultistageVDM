import os
import typer
import warnings
import numpy as np
import SimpleITK as sitk

from patient_seeds import all_patient_seeds
from totalsegmentator.python_api import totalsegmentator

app = typer.Typer()

@app.command()
def segmentation(root_dir: str = typer.Option("", "-r", "--root-dir"),
                 output_dir: str = typer.Option("results/test", "-o", "--output-dir"),

                 patient: str = typer.Option(..., "-p", "--patient"),
                 # Getting the arguments to define the region of interest
                 use_skull: bool = typer.Option(True, "--use-skull/--no-skull"),
                 use_vertebrae: bool = typer.Option(True, "--use-vertebrae/--no-vertebrae"),
                 normal_ts: bool = typer.Option(False, "--normal-ts"),
                 use_circularity: bool = typer.Option(True, "--circularity/--no-circularity"),

                 device: str = typer.Option("cpu", "-d", "--device")):
    
    image_path = root_dir
    output_path = output_dir
    # Building the name of the folder (suffix).
    suffix = ""
    if not use_skull: # The skull is outside the ROI.
        suffix += "_NS"

    if not use_vertebrae: # The spinal vertebrae is outside the ROI.
        suffix += "_NV"

    if normal_ts:
        suffix += "_NTS" # Normal mode totalsegmentator

    if not use_circularity: # NotAVesselStoppingCriterion is not used.
        suffix  += "_NC"

    for study in sorted(os.listdir(os.path.join(image_path, patient))):
    # Now we check the series and the study.
        study_path = os.path.join(image_path, patient, study)

        for series in sorted(os.listdir(study_path)):
            
            try: 
                key = f"{patient}_{series}"
                patient_entry = all_patient_seeds.get(key)

                if patient_entry is None: # Seeds where not in the dictionary.
                    warnings.warn(f"Skipping TS 2.13.0  preprocessing for {patient}/{study}/{series}")
                    continue

                path = os.path.join(image_path, patient, study, series, "DICOM", f"{patient}_{series}.mhd")
                
                out_path = os.path.join(output_path, patient, study, f"{series}{suffix}")

                os.makedirs(os.path.join(out_path, "totalsegmentator"), exist_ok=True)

                # First reading the original image and rewriting it into the totalsegmentator folder.
                itk_image = sitk.ReadImage(path)

                nifti_path = os.path.join(out_path, "totalsegmentator", "raw.nii.gz")
                mask_path = os.path.join(out_path, "totalsegmentator", "mask_vessels.nii.gz")
                sitk.WriteImage(itk_image, nifti_path)

                if os.path.isdir(mask_path): # It was already preprocessed by TS 2.13.0
                    print(f"TotalSegmentator 2.13.0 preprocessing skipped for {patient}/{study}/{series} reason: already processed.")

                else:
                    totalsegmentator(input = nifti_path,
                                    output = mask_path,
                                    device = "gpu" if device == "cuda" else "cpu",
                                    ml = True,
                                    fast = False,
                                    quiet = True,
                                    task = "headneck_bones_vessels")
                    print(f"TotalSegmentator 2.1.0+ preprocessing succesfully completed for {patient}/{study}/{series}!")
                
            except Exception as e:
                warnings.warn(f"Patient {patient}/{study}/{series} failed: {e}")

if __name__ == "__main__":
    app()     
 