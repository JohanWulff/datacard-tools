# coding: utf-8

import argparse

from datacard_parser import Datacard

from concurrent.futures import ProcessPoolExecutor, as_completed

import uproot
import numpy as np
import click
import tqdm


def check_hists(datacard: Datacard,
                threshold: float = 100) -> dict:
                  
    """
    Checks for non-finite values (NaN) and yields below 1e-5 in the histograms of a datacard. 
    Args:
        datacard (Datacard): The datacard to check for NaN values.
    """
    problematic_shapes = {}
    with uproot.open(datacard.shapes_file) as f:
        for nuisance in datacard.shape_nuisances:
            print(f"[INFO] Checking nuisance: {nuisance}")
            for bkgd in datacard.background_processes:
                print(f"  [INFO] Checking background: {bkgd}")
                nominal, up, down = datacard.get_nom_up_down(nuisance, bkgd, shapes_file_handle=f)

                # nan / inf check
                for shape in [nominal, up, down]:
                    values = shape.values()
                    variances = shape.variances()
                    if ~np.isfinite(values).any() or ~np.isfinite(variances).any():
                        problematic_shapes[f"{nuisance}_{bkgd}"] = {
                            "values": values,
                            "variances": variances
                        }
                    if (values < 1e-6).any():
                        problematic_shapes[f"{nuisance}_{bkgd}"] = {
                            "values": values,
                            "variances": variances
                        }

                ratio_up = up.values() / nominal.values()
                ratio_down = down.values() / nominal.values()
                if ratio_up.max() > threshold or ratio_up.min() < 1/threshold or ratio_down.max() > threshold or ratio_down.min() < 1/threshold:
                    problematic_shapes[f"{nuisance}_{bkgd}"] = {
                        "values": nominal.values(),
                        "variances": nominal.variances(),
                        "ratio_up": ratio_up,
                        "ratio_down": ratio_down
                    }
    return problematic_shapes

    
# for now we just run one process

def make_parser():
    parser = argparse.ArgumentParser(description="Check datacards for NaN values and very small yields.")
    parser.add_argument("datacards", nargs="+", help="Paths to datacard files to check.")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    for datacard_path in args.datacards:
        print(f"[INFO] Checking datacard: {datacard_path}")
        datacard = Datacard(datacard_path, ignore_processes=["data_obs"])
        problematic_shapes = check_hists(datacard)
        if problematic_shapes:
            print(f"[WARNING] Found problematic shapes in datacard {datacard_path}:")
            for shape_key, shape_info in problematic_shapes.items():
                print(f"  - Shape key: {shape_key}")
                print(f"    Values: {shape_info['values']}")
                print(f"    Variances: {shape_info['variances']}")
                if "ratio_up" in shape_info and "ratio_down" in shape_info:
                    print(f"    Ratio up: {shape_info['ratio_up']}")
                    print(f"    Ratio down: {shape_info['ratio_down']}")
        else:
            print(f"[INFO] No problematic shapes found in datacard {datacard_path}.")
            
            
if __name__ == "__main__":
    main()

            
            
#def futures_wrapper(datacard_path: str) -> dict:
#    datacard = Datacard(datacard_path, ignore_processes=["data_obs"])
#    return check_hists(datacard)
#        
#    
#@click.command(help="Check for nonfinite values and yields < 1e-5 in the datacard.")
#@click.argument("datacards", nargs=-1, required=True)
#@click.option(
#    "--max-processes",
#    type=click.IntRange(min=1),
#    default=4,
#    show_default=True,
#    help="Maximum number of worker processes.",
#)
#def main(datacards: tuple[str, ...], max_processes: int):
#    click.echo("[INFO] Found {} datacards.".format(len(datacards)))
#    click.echo("[INFO] Using max_processes={}.".format(max_processes))
#
#    nan_shapes = {}
#    with ProcessPoolExecutor(max_workers=max_processes) as executor:
#        future_to_datacard = {
#            executor.submit(futures_wrapper, datacard_path): datacard_path
#            for datacard_path in datacards
#        }
#        for future in tqdm.tqdm(
#            as_completed(future_to_datacard),
#            total=len(future_to_datacard),
#            desc="Checking datacards",
#        ):
#            datacard_path = future_to_datacard[future]
#            try:
#                result = future.result()
#                if result:
#                    nan_shapes[datacard_path] = result
#            except Exception as e:
#                print("[ERROR] Error occurred while checking datacard {}: {}".format(datacard_path, e))
#                continue
#
#    if nan_shapes:
#        for datacard_path, datacard_nan_shapes in nan_shapes.items():
#            print("[WARNING] Found NaN values in the following shapes for datacard {}:".format(datacard_path))
#            for shape_key, shape_info in datacard_nan_shapes.items():
#                print("  - Shape key: {}".format(shape_key))
#                print("    Values: {}".format(shape_info["values"]))
#                print("    Variances: {}".format(shape_info["variances"]))
#
#
#if __name__ == "__main__":
#    main()
    
    

    