import json
from pathlib import Path
import uproot
from datacard_parser import Datacard

def crosscheck_sum_shape_var(datacard_path,validation_results_dir="/tmp/jwulff/inference/validation_results"):
    datacard = Datacard(datacard_path)
    validation_path = datacard.validate(validation_results_dir)
    if not validation_path:
        raise ValueError(f"Validation failed for datacard: {datacard_path}")
    else:
        with open(validation_path, 'r') as f:
            validation = json.load(f)
    small_shape_eff = validation.get("smallShapeEff", {})
    # reduce small_shape_eff dict by removing the cat entry
    small_shape_eff = {nuisance: small_shape_eff[nuisance][datacard.dirname] for nuisance in small_shape_eff}

    with uproot.open(datacard.shapes_file) as shapes_file_handle:
        for nuisance, processes in small_shape_eff.items():
            print(f"Checking nuisance: {nuisance}")
            for process, json_vals in processes.items():
                up_json, down_json = json_vals["diff_u"], json_vals["diff_d"]
                up_calc, down_calc = datacard._get_sum_shape_var(nuisance, process, shapes_file_handle)
                print(f"  Process: {process}")
                if not (abs(up_json - up_calc) < 1e-6 and abs(down_json - down_calc) < 1e-6):
                    print(f"    JSON:   up={up_json:.6f}, down={down_json:.6f}")
                    print(f"    Calc:   up={up_calc:.6f}, down={down_calc:.6f}")
                    print("    MISMATCH!")
                else:
                    print("    Match.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python crosscheck_sum_shape_var.py <datacard.txt>")
        sys.exit(1)
    crosscheck_sum_shape_var(sys.argv[1])