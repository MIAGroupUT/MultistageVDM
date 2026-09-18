import glob
import os
import warnings
warnings.filterwarnings("ignore",
                        message="pkg_resources is deprecated as an API")


import typer

from patient_seeds import all_patient_seeds

from umc_wrapper_final import UMCTotalSegmentatorProcessorComplete
from src.sire.inference.inference_models import SegmentationInferenceModel, TrackerInferenceModel
from src.sire.inference.segmentator_tracker_1 import SegmentatorTrackerPipeline
from src.sire.models.sire_seg import SIRESegmentation
from src.sire.models.sire_tracker import SIRETracker

app = typer.Typer()

@app.command()
def inference(root_dir: str = typer.Option("", "-r", "--root-dir"),
              output_dir: str = typer.Option("results/test", "-o", "--output-dir"),
            
              patient: str = typer.Option("", "-p", "--patient"),
              
              use_skull: bool = typer.Option(True, "--use-skull/--no-skull"),
              use_vertebrae: bool = typer.Option(True, "--use-vertebrae/--no-vertebrae"),
              normal_ts: bool = typer.Option(False, "--normal-ts"),
              use_circularity: bool = typer.Option(True, "--circularity/--no-circularity"),

              device: str = typer.Option("cpu", "-d", "--device")):
    
    scales = [4, 6, 8, 12, 16, 20, 24, 28, 32, 36] # scales 1

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

    # Options dictionary to send to the ROI generator, tracker etc.
    options = {"use_skull" : use_skull,
               "use_vertebrae" : use_vertebrae,
               "normal_mode_ts" : normal_ts,
               "circularity_sc" : use_circularity}

    # Load tracking model
    tracker_model = TrackerInferenceModel(
        model=SIRETracker.load_from_checkpoint("src/sire/models/checkpoints/tracking_model.ckpt", strict=False),
        scales=scales, # maybe check this.
        npoints=32,
        subdivisions=3,
        device=device
    )

    # Load segmentation model
    segmentation_models = [
        SegmentationInferenceModel(
            model=SIRESegmentation.load_from_checkpoint("src/sire/models/checkpoints/segmentation_model.ckpt", strict=False),
            names=["lumen"],
            scales=scales,
            npoints=32,
            subdivisions=2,
            device=device
        ),
    ]

    image_path = root_dir
    output_path = output_dir

    # Patient 006 -> NV_NTS
    # for patient in patients:
    for study in sorted(os.listdir(os.path.join(image_path, patient))):
        # Now we check the series and the study.
        study_path = os.path.join(image_path, patient, study)
        for series in sorted(os.listdir(study_path)):
            
            try: 
                umc_preprocessor = UMCTotalSegmentatorProcessorComplete()
                tracker_pipeline = SegmentatorTrackerPipeline(tracker_model,
                                                              segmentation_models)

                path = os.path.join(image_path, patient, study, series, "DICOM", f"{patient}_{series}.mhd")
                out_path = os.path.join(output_path, patient, study, f"{series}{suffix}")

                # Checking if the dictionary has the patient_seeds, if not the loop continues.
                key = f"{patient}_{series}"
                entry = all_patient_seeds.get(key)
                if entry is None:
                    warnings.warn(f"No seeds for {key}")
                    continue

                left_seeds = entry.get("left_seeds")
                right_seeds = entry.get("right_seeds")
                if left_seeds is None or right_seeds is None:
                    warnings.warn(f"Missing left/right seeds for {key}")
                    continue

                vessel_configs, anatomical_indices = umc_preprocessor(path, 
                                                                      out_path,
                                                                      left_seeds,
                                                                      right_seeds,
                                                                    #   fast_ts_preprocessing = False,
                                                                      device = "cuda",
                                                                      **options)

                if len(vessel_configs) != 0:
                    tracker_pipeline.run(path, 
                                         output_dir = out_path, 
                                         vessel_configs = vessel_configs,
                                         anatomical_landmarks = anatomical_indices, 
                                         already_tracked_distance = 0,
                                         save_diameters = False)
                
                else:
                    warnings.warn("No suitable VesselConfigs were generated.")
            except Exception as e:
                warnings.warn(f"Patient {patient}/{study}/{series} failed: {e}")

if __name__ == "__main__":
    app()
