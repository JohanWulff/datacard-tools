from datacard_parser import Datacard

from typing import List, Dict, Tuple
from pathlib import Path
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import argparse
import json
import numpy as np
import re
import uproot
import subprocess
import os

from hist import Hist 
import hist 


def get_rate_string(up_rate, down_rate):
    if np.abs(1-up_rate) < 0.01 and np.abs(1-down_rate) < 0.01:
        return "-"
    # ignoring this case for now in order to avoid that shape/ bug with asym lnN's
    #elif np.abs(1-up_rate) > 0.01 and np.abs(1-down_rate) > 0.01:
    #   return f"{down_rate:.3f}/{up_rate:.3f}"
    else:
        return f"{np.max((down_rate, up_rate)):.3f}"


def get_non_genuine_shapes(datacard, threshold, shapes_file_handle=None) -> List[str]:
    """
    Get non-genuine shapes from the datacard.
    """
    
    shape_nuisances = [n for n in datacard.get_nuisance_types() if datacard.get_nuisance_types()[n] == "shape"]
    non_genuine_shapes = []
    if shapes_file_handle is None:
        with uproot.open(datacard.shapes_file) as f:
            for nuisance in shape_nuisances:
                val_up, val_down = datacard.get_sum_shape_var(nuisance, f)
                if val_up < threshold and val_down < threshold:
                    non_genuine_shapes.append(nuisance)
    else:
        for nuisance in shape_nuisances:
            val_up, val_down = datacard.get_sum_shape_var(nuisance, shapes_file_handle)
            if val_up < threshold and val_down < threshold:
                non_genuine_shapes.append(nuisance)
    return non_genuine_shapes


def replace_nuisance_lines(old_card: Datacard,
                           new_card: Path,
                           modifications: list[tuple[str, str, dict[str, str]]]) -> None:
    """
    Apply multiple nuisance line replacements in one go.
    Each modification is a tuple: (nuisance, nuisance_type, new_entries)
    """
    if not isinstance(new_card, Path):
        new_card = Path(new_card)
    if not new_card.exists():
        print(f"Replacement path {new_card} does not exist, creating it.")
        os.makedirs(new_card, exist_ok=True)
    with open(old_card.datacard, "r") as f:
        lines = f.readlines()

    for nuisance, nuisance_type, new_entries in modifications:
        nuisance_lines = [i for i, l in enumerate(lines) if l.startswith(nuisance+" ")]
        if len(nuisance_lines) == 0:
            raise ValueError(f"Nuisance {nuisance} not found in datacard {old_card.datacard}")
        elif len(nuisance_lines) > 1:
            raise ValueError(f"Found multiple lines for nuisance {nuisance} in datacard {old_card.datacard}")
        if len(new_entries) != len(old_card.processes):
            raise ValueError(f"Expected {len(old_card.processes)} entries for nuisance {nuisance}, but got {len(new_entries)}")
        line_index = [l.split()[0] for l in lines].index(nuisance)
        # check if all new entries are empty -> line can be removed
        if all(entry == "-" for entry in new_entries.values()):
            # remove the line
            lines.pop(line_index)
            continue
        new_line = [nuisance, nuisance_type]
        process_entries = ["-" for _ in old_card.all_processes] # ignore processes will get a "-" by default
        for process, entry in new_entries.items():
            if process not in old_card.processes:
                raise ValueError(f"Process {process} not found in datacard {old_card.datacard}")
            process_entries[old_card.processes.index(process)] = entry
        new_line.extend(process_entries)
        spaces = [old_card.positions[i+1] - (old_card.positions[i]+len(new_line[i])) for i in range(len(new_line)-1)]
        new_line = "".join([f"{new_line[i]}{' ' * spaces[i]}" for i in range(len(new_line)-1)]) + f"{new_line[-1]}\n"
        lines[line_index] = new_line
    new_datacard_path = new_card / old_card.datacard.name
    with open(new_datacard_path, "w") as f:
        f.writelines(lines)


def conservative_update(datacard: Datacard,
                    output_path: Path,
                    validation_results_dir: str,
                    only_remove: bool = False) -> None:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    If only_remove is True, only remove nuisances, do not convert to lnN.
    """
    def get_rate_string(up_rate, down_rate):
        if np.abs(1-up_rate) < 0.01 and np.abs(1-down_rate) < 0.01:
            return "-"
        # ignoring this case for now in order to avoid that shape/ bug with asym lnN's
        #elif np.abs(1-up_rate) > 0.01 and np.abs(1-down_rate) > 0.01:
        #   return f"{down_rate:.3f}/{up_rate:.3f}"
        else:
            return f"{np.max((down_rate, up_rate)):.3f}"

    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = set([re.sub(";\d", "", key) for key, cname in cnames.items() if key.startswith(datacard.dirname)
                         and cname == "TH1D"]) 
        validation_results = datacard.validate(validation_results_dir)
    
        # before doing any update, check if there's any signal in the datacard
        ## NOTE: Not copying datacards with no signal to the output directory is debatable. 
        ## the one bin clearly contributes to the background modelling so should probably be kept.
        ## I personally don't expect a large discprepancy on limit level however. 

        #signal_names = [p for p in datacard.processes if (("ggf" in p) or ("vbf" in p))]
        #assert len(signal_names) == 1
        #signal_name = signal_names[0]
        #nominal_signal = f[f"{datacard.dirname}/{signal_name}"].to_hist()
        #if nominal_signal.sum().value <= 1e-5:
        #    print(f"{signal_name} Integral <= 0.00001")
        #    print(f"{datacard.datacard.name} won't be copied as it contains no signal")
        #    return {}
            
        #from IPython import embed; embed()
        if not validation_results:
            raise ValueError(f"Validation failed for datacard {datacard.datacard.name}. Cannot proceed with update.")

        with open(validation_results, "r") as vf:
            validation_results_json = json.load(vf)

        if not "smallShapeEff" in validation_results_json:
            print(f"No small shape effects found for datacard {datacard.datacard.name}. No update necessary.")
            # copy the datacard and the shapes file to the output directory
            output_datacard_path = Path(output_path) / datacard.datacard.name
            shutil.copy(datacard.datacard, output_datacard_path)
            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            shutil.copy(datacard.shapes_file, output_shapes_file)
            return
        else:
            small_shape_effects = validation_results_json["smallShapeEff"]
            cat_name = next(iter(small_shape_effects[next(iter(small_shape_effects))]))
            small_shape_effects = {nuisance: small_shape_effects[nuisance][cat_name] for nuisance in small_shape_effects}

            modifications = []  # Collect all modifications here
            remove_unused_shapes = set()
            for nuisance in small_shape_effects:
                if len(small_shape_effects[nuisance]) == len(datacard.processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    rates = {process: get_rate_string(*rates[process]) for process in rates}
                    # Check if all entries are "-"
                    if all(rate == "-" for rate in rates.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "lnN"
                        if only_remove:
                            continue  # Skip conversion if only_remove is set
                    modifications.append((nuisance, nuisance_type, rates))
                elif len(small_shape_effects[nuisance]) < len(datacard.processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    keep_processes =  set(datacard.processes) - set(small_shape_effects[nuisance].keys())
                    rates = {process: get_rate_string(*rates[process]) for process in small_shape_effects[nuisance].keys()}
                    if all(rate == "-" for rate in rates.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "shape?"
                        continue # skip modification of mixed nuisances for now 
                        # remove_unused_shapes = True
                    rates.update({process: "1" for process in keep_processes})
                    modifications.append((nuisance, nuisance_type, rates))
                else:
                    raise ValueError(f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}")

                remove_unused_shapes = set([f"{datacard.dirname}/{p}_{nuisance}{ud}"
                                                for p in small_shape_effects[nuisance].keys()
                                                for ud in ["Up", "Down"]])
            keep_keys -= remove_unused_shapes

            # Apply all modifications at once
            if modifications:
                replace_nuisance_lines(datacard, output_path, modifications)
            
            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            shutil.copy(datacard.shapes_file, output_shapes_file)
            if remove_unused_shapes:
                # instead of removing unused shapes, we just create a new shapes file with only the used shapes
                with uproot.recreate(output_shapes_file) as new_shapes_file:
                    #print(f"Keeping {len(keep_keys)}/{len(f.keys())} keys in shapes file {output_shapes_file}")
                    hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
                    for key, hist in hists.items():
                        try:
                            hist = update_bugged_hist(hist)
                            new_shapes_file[key] = hist
                        except ValueError:
                            print(f"Warning: Found non-finite values in histogram {key} for datacard {datacard.datacard.name}. Keeping original histogram without update.")
                            print(f"Take a look at the shapes file {datacard.shapes_file} for more details.")
            else:
                # If no shapes were removed, just copy the file
                shutil.copy(datacard.shapes_file, output_shapes_file)


def update_bugged_hist(hist: Hist) -> None:
    hist_vals, hist_vars = hist.values(), hist.variances()
    if not np.all(np.isfinite(hist_vals)):
        # if there are non-finite values, set them to a small number and print a warning
        raise ValueError("Found non-finite values in histogram.")
    if not np.all(np.isfinite(hist_vars)):
        print(f"Warning: Found non-finite bin-errors in histogram {hist}")
        hist_vals[~np.isfinite(hist_vals)] = 1e-6
        hist_vars[~np.isfinite(hist_vars)] = 1e-5
    if np.any(mask:= (hist_vals < 1e-5)):
        hist_vals[mask] = 1e-5
        hist_vars[mask] = 1e-6
    elif np.any(mask:=(hist_vars < 1e-6)):
        hist_vars[mask] = 1e-6
    else:
        return hist  # No bug found, return original histogram
    new_hist = Hist(hist.axes[0], storage=hist.storage_type())
    new_hist.view().value = hist_vals
    new_hist.view().variance = hist_vars
    return new_hist


def loose_update(datacard: Datacard,
                 output_path: Path,
                 threshold: float = 0.01,) -> None:

    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    This is a more relaxed version of conservative_update, which does not check for small shape effects.
    """

    nuisance_types_before = datacard.get_nuisance_types()  # {nuisance: type}
    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = set([re.sub(";\d", "", key) for key, cname in cnames.items() if key.startswith(datacard.dirname)
                            and cname == "TH1D"])
        # Get all shape nuisances
        shape_nuisances = [n for n in nuisance_types_before if nuisance_types_before[n] == "shape"]
        modifications = []  # Collect all modifications here
        for nuisance in shape_nuisances:
            rates = datacard.get_rates(nuisance, shapes_file_handle=f)
            flagged = datacard.get_shape_vars(nuisance, threshold=threshold, shapes_file_handle=f)
            if len(flagged) == datacard.processes:
                rate_entries = {process: get_rate_string(*rates[process]) for process in rates}
                modifications.append((nuisance, "lnN", rate_entries))
            elif len(flagged) < len(datacard.processes):
                # Some processes are flagged, others are not
                keep_processes = set(datacard.processes) - set(flagged)
                rate_entries = {process: get_rate_string(*rates[process]) for process in flagged}
                if all(rate == "-" for rate in rate_entries.values()):
                    nuisance_type = "shape"
                else:
                    nuisance_type = "shape?"
                rate_entries.update({process: "1" for process in keep_processes})
                modifications.append((nuisance, nuisance_type, rate_entries))

            keep_keys -= set([f"{datacard.dirname}/{process}_{nuisance}{ud}"
                                for process in flagged for ud in ["Up", "Down"]])
    # Apply all modifications at once
    replace_nuisance_lines(datacard, output_path, modifications)
    # Copy the shapes file to the output path
    output_shapes_file = Path(output_path) / datacard.shapes_file.name
    shutil.copy(datacard.shapes_file, output_shapes_file)
    with uproot.recreate(output_shapes_file) as new_shapes_file:
        print(f"Keeping {len(keep_keys)}/{len(f.keys())} keys in shapes file {output_shapes_file}")
        hists = datacard.get_shape_hists(keys=list(keep_keys), shapes_file_handle=f)
        for key, hist in hists.items():
            new_shapes_file[key] = hist

# new function for checking large shape effects that are probably none-physical
def fix_large_shapes(datacard: Datacard,
                     output_path: Path):
    """
    Check each shape effect for large shape effects and adjust them to neighboring bins if necessary. 
    A shape effect is considered unphysical if it's larger than 100 times the nominal value. 
    """

    def smoothen(idx: int, variations: np.ndarray,) -> np.ndarray:
        """
        helper to smoothen the variations
        """
        # what to do if there's only one bin?
        if len(variations) == 1:
            # cannot smoothen based on other bins, so just
            return variations 
        # check if bin_idx is on the border 
        if idx == 0 or idx == len(variations)-1:
            variations[idx] == variations[idx+1] if idx == 0 else variations[idx-1]
        else:
            variations[idx] = (variations[idx+1]+variations[idx-1])/2.
        return variations

    changed = {}
    with uproot.open(datacard.shapes_file) as f:
        for nuisance in datacard.shape_nuisances:
            for bkgd in datacard.background_processes:
                up_var, down_var = datacard.get_bin_variations(nuisance, bkgd, shapes_file_handle=f)
                # print a warning if one of the variations is more than 100 times as large as the other one
                problematic_bins_up = up_var > 100
                problematic_bins_down = down_var < 1/100
                if any(problematic_bins_up) or any(problematic_bins_down):
                    # check if problematic bins are neighbors, if yes, raise a warning
                    bin_ids_up = np.where(problematic_bins_up)[0]
                    diff = bin_ids_up[1:] - bin_ids_up[:-1]
                    if np.any(diff == 1):
                        print(f"Warning: Found neighboring large shape effects for nuisance {nuisance} and process {bkgd} in datacard {datacard.datacard.name}")
                        print(f"Will not smoothen these bins.")
                        print(f"Up variations: {up_var}")
                    bin_ids_down = np.where(problematic_bins_down)[0]
                    diff = bin_ids_down[1:] - bin_ids_down[:-1]
                    if np.any(diff == 1):
                        print(f"Warning: Found neighboring large shape effects for nuisance {nuisance} and process {bkgd} in datacard {datacard.datacard.name}")
                        print(f"Will not smoothen these bins.")
                        print(f"Down variations: {down_var}")
                    print(f"Warning: Found large shape effect for nuisance {nuisance} and process {bkgd} in datacard {datacard.datacard.name}")
                    print(f"Up variations: {up_var}")
                    print(f"Down variations: {down_var}")
                    # smoothen the variations
                    corrected_up_var = np.copy(up_var)
                    corrected_down_var = np.copy(down_var)
                    for bins in problematic_bins_up:
                        corrected_up_var = smoothen(bins, corrected_up_var)
                    for bins in problematic_bins_down:
                        corrected_down_var = smoothen(bins, corrected_down_var)

                    up_shape = f[f"{datacard.dirname}/{bkgd}__{nuisance}Up"].to_hist()
                    down_shape = f[f"{datacard.dirname}/{bkgd}__{nuisance}Down"].to_hist()
                    nom_shape = f[f"{datacard.dirname}/{bkgd}"].to_hist()
                    # apply the corrected variations to the shapes
                    up_shape.view().value = nom_shape.values() * corrected_up_var
                    down_shape.view().value = nom_shape.values() * corrected_down_var
                    changed[f"{datacard.dirname}/{bkgd}_{nuisance}Up"] = up_shape
                    changed[f"{datacard.dirname}/{bkgd}_{nuisance}Down"] = down_shape
        # save changes
        if changed:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # first read all histograms that are not changed
            keep_keys = set([re.sub(";\d", "", key) for key, cname in f.classnames().items() if key.startswith(datacard.dirname)
                                and cname == "TH1D"]) - set(changed.keys())
            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
            hists.update(changed)
            with uproot.recreate(output_shapes_file) as new_shapes_file:
                for key, hist in hists.items():
                    new_shapes_file[key] = hist


def process_datacard_wrapper(
    datacard_path: str,
    ignore_processes: list[str],
    update_mode: str,
    output_path: str,
    validation_results_dir: str,
    only_remove: bool,
    threshold: float,
) -> None:
    datacard = Datacard(datacard=Path(datacard_path), ignore_processes=ignore_processes)
    if update_mode == "conservative":
        conservative_update(
            datacard,
            Path(output_path) / "conservative",
            validation_results_dir,
            only_remove=only_remove,
        )
    elif update_mode == "loose":
        loose_update(datacard, Path(output_path) / "loose", threshold=threshold)
    elif update_mode == "smoothen":
        fix_large_shapes(datacard, Path(output_path) / "smoothen")


def main():
    parser = argparse.ArgumentParser(description="Validate and update a datacard with non-genuine shape nuisances.")
    parser.add_argument("--mass", type=int, nargs="+", required=False,
                        choices=(MASSES:=[250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
                                 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1250,
                                 1500, 1750, 2000, 2500, 3000]),
                        default=MASSES,
                        help=f"Mass of the hypothetical particle in GeV. Default: {MASSES}.")
    parser.add_argument("--datacard-path", "-d", type=str, 
                        default=(dc_path:=("/data/dust/user/kramerto/taunn_data/store/WriteDatacards/"
                                 "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite"
                                 "-default_extended_pair_ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096"
                                 "_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_fi80_lbn_ft_lt20_lr1_LBdefault_"
                                 "daurot_fatjet_composite_FIx5_SDx5/prod9/flats_systs10/final/symtest/")),
                        help=f"Path to the datacards directory. Default: {dc_path}")
    parser.add_argument("--output_path", "-o", type=str, help="/output/path/for/updated/datacards/",
                        default="/data/dust/user/jwulff/inference/remodel_cards/")
    parser.add_argument("--validation_results_dir", type=str,
                        default="/tmp/jwulff/inference/validation_results/", help="Directory to store validation results.")
    parser.add_argument("--ignore-processes", nargs='*', default=["data_obs", "QCD"], help="Processes to ignore in the datacard.")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=4,
        help="Maximum number of worker processes.",
    )
    parser.add_argument("--only-remove", action="store_true", help="Only remove nuisances, do not convert shape to lnN.")
    parser.add_argument("--update-mode", choices=["conservative", "loose", "smoothen"], default="conservative",
                        help="Choose update mode: 'conservative' (default), 'loose' or 'smoothen'.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for non-genuine shape detection (only used in loose mode).")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # get the datacard paths
    datacard_paths = []
    for mass in args.mass:
        paths = list(Path(args.datacard_path).glob(f"datacard_cat_*_spin_*_mass_{mass}.txt"))
        if not paths:
            print(f"No datacards found for mass {mass} in {args.datacard_path}")
            continue
        datacard_paths.extend(paths)

    campaigns = sorted({re.search(r"datacard_cat_([^_]+)_", path.name).group(1) for path in datacard_paths})
    spins = sorted({re.search(r"spin_(\d+)", path.name).group(1) for path in datacard_paths})
    print(
        f"Found {len(datacard_paths)} datacards for campaigns {campaigns}, "
        f"spins {spins} and mass {args.mass} in {args.datacard_path}"
    )
    with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
        future_to_datacard = {
            executor.submit(
                process_datacard_wrapper,
                str(path),
                args.ignore_processes,
                args.update_mode,
                args.output_path,
                args.validation_results_dir,
                args.only_remove,
                args.threshold,
            ): str(path)
            for path in datacard_paths
        }
        for future in tqdm(
            as_completed(future_to_datacard),
            total=len(future_to_datacard),
            desc="Processing datacards",
        ):
            datacard_path = future_to_datacard[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Error while processing datacard {datacard_path}: {e}")
                continue

if __name__ == "__main__":
    main()
