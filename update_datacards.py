from datacard_parser import Datacard

from typing import List, Dict, Tuple
from pathlib import Path
import shutil

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import argparse
import collections
import json
import numpy as np
import re
import uproot
import subprocess
import os


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
                    only_remove: bool = False) -> dict:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    Also returns statistics about nuisance types before and after the update.
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

    # --- STATISTICS: BEFORE UPDATE ---
    nuisance_types_before = datacard.get_nuisance_types()  # {nuisance: type}
    stats_before = collections.Counter(nuisance_types_before.values())

    modified_nuisances = set()
    untouched_nuisances = set(nuisance_types_before.keys())

    # New counters
    n_removed = 0

    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = set([re.sub(";\d", "", key) for key, cname in cnames.items() if key.startswith(datacard.dirname)
                         and cname == "TH1D"]) 
        validation_results = datacard.validate(validation_results_dir)
        if not validation_results:
            print(f"Validation failed for {datacard.datacard}, skipping update.")
            return {} 

        with open(validation_results, "r") as vf:
            validation_results_json = json.load(vf)

        if not "smallShapeEff" in validation_results_json:
            print(f"No smallShapeEffect warnings found in validation results for {datacard.datacard}")
            return False
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
                        n_removed += 1
                    else:
                        nuisance_type = "lnN"
                        if only_remove:
                            continue  # Skip conversion if only_remove is set
                    modifications.append((nuisance, nuisance_type, rates))
                    modified_nuisances.add(nuisance)
                elif len(small_shape_effects[nuisance]) < len(datacard.processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    keep_processes =  set(datacard.processes) - set(small_shape_effects[nuisance].keys())
                    rates = {process: get_rate_string(*rates[process]) for process in small_shape_effects[nuisance].keys()}
                    if all(rate == "-" for rate in rates.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "shape?"
                        n_shapeq += 1
                        continue # skip modification of mixed nuisances for now 
                        # remove_unused_shapes = True
                    rates.update({process: "1" for process in keep_processes})
                    modifications.append((nuisance, nuisance_type, rates))
                    modified_nuisances.add(nuisance)
                else:
                    raise ValueError(f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}")

                remove_unused_shapes = set([f"{datacard.dirname}/{p}_{nuisance}{ud}"
                                                for p in small_shape_effects[nuisance].keys()
                                                for ud in ["Up", "Down"]])
                # Remove unused nuisances
                keep_keys -= remove_unused_shapes

            untouched_nuisances -= modified_nuisances
            keep_keys -= remove_unused_shapes

            # Apply all modifications at once
            if modifications:
                replace_nuisance_lines(datacard, output_path, modifications)

            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            shutil.copy(datacard.shapes_file, output_shapes_file)
            if remove_unused_shapes:
                #retcode = subprocess.run(["remove_unused_shapes.py", str(datacard.datacard), "*,*", "--inplace-shapes"],
                               #stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                #if retcode.returncode != 0:
                    #print(f"Error removing unused shapes: {retcode.stderr}")
                    #return False
                
                # instead of removing unused shapes, we just create a new shapes file with only the used shapes
                with uproot.recreate(output_shapes_file) as new_shapes_file:
                    print(f"Keeping {len(keep_keys)}/{len(f.keys())} keys in shapes file {output_shapes_file}")
                    hists = datacard.get_shape_hists(keys=list(keep_keys), shapes_file_handle=f)
                    for key, hist in hists.items():
                        new_shapes_file[key] = hist
            else:
                # If no shapes were removed, just copy the file
                shutil.copy(datacard.shapes_file, output_shapes_file)
                    

    # --- STATISTICS: AFTER UPDATE ---
    updated_datacard_path = Path(output_path) / datacard.datacard.name
    updated_datacard = Datacard(datacard=updated_datacard_path, ignore_processes=datacard.ignore_processes)
    nuisance_types_after = updated_datacard.get_nuisance_types()
    stats_after = collections.Counter(nuisance_types_after.values())

    # --- STATISTICS: UNTOUCHED ---
    untouched_types = [nuisance_types_before[n] for n in untouched_nuisances]
    stats_untouched = collections.Counter(untouched_types)

    # --- PRINT STATISTICS ---
    #print("\nNuisance statistics:")
    #print("Before update:", dict(stats_before))
    #print("After update: ", dict(stats_after))
    #print("Untouched:    ", dict(stats_untouched))
    #print(f"Modified:     {len(modified_nuisances)}")
    #print(f"Untouched:    {len(untouched_nuisances)}")
    #print(f'Updated to "shape?": {n_shapeq}')
    #print(f'Removed: {n_removed}\n')

    return {
        "before": dict(stats_before),
        "after": dict(stats_after),
        "untouched": dict(stats_untouched),
        "n_modified": len(modified_nuisances),
        "n_untouched": len(untouched_nuisances),
        "n_shapeq": n_shapeq,
        "n_removed": n_removed,
    }

    
def loose_update(datacard: Datacard,
                 output_path: Path,
                 threshold: float = 0.01,) -> dict:

    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    This is a more relaxed version of conservative_update, which does not check for small shape effects.
    """

    nuisance_types_before = datacard.get_nuisance_types()  # {nuisance: type}
    stats_before = collections.Counter(nuisance_types_before.values())

    modified_nuisances = set()
    removed_nuisances = set()
    modified_nuisances = set()
    mixed_nuisances = set()
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
                if all(rate == "-" for rate in rates.values()):
                    removed_nuisances.add(nuisance)
                else:
                    modified_nuisances.add(nuisance)
                rate_entries = {process: get_rate_string(*rates[process]) for process in rates}
                modifications.append((nuisance, "lnN", rate_entries))
            elif len(flagged) < len(datacard.processes):
                # Some processes are flagged, others are not
                keep_processes = set(datacard.processes) - set(flagged)
                rate_entries = {process: get_rate_string(*rates[process]) for process in flagged}
                if all(rate == "-" for rate in rate_entries.values()):
                    modified_nuisances.add(nuisance)
                    nuisance_type = "shape"
                else:
                    modified_nuisances.add(nuisance)
                    mixed_nuisances.add(nuisance)
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

    # --- STATISTICS ---
    updated_datacard_path = Path(output_path) / datacard.datacard.name
    updated_datacard = Datacard(datacard=updated_datacard_path, ignore_processes=datacard.ignore_processes)
    nuisance_types_after = updated_datacard.get_nuisance_types()
    stats_after = collections.Counter(nuisance_types_after.values())
    stats_before = collections.Counter(nuisance_types_before.values())
    untouched_nuisances = set(nuisance_types_before.keys()) - modified_nuisances - removed_nuisances
    stats_untouched = collections.Counter([nuisance_types_before[n] for n in untouched_nuisances])
    return {
        "before": dict(stats_before),
        "after": dict(stats_after),
        "unaltered": dict(stats_untouched),
        "n_unaltered": len(untouched_nuisances),
        "altered": dict(collections.Counter(nuisance_types_after[n] for n in modified_nuisances)),
        "n_altered": len(modified_nuisances)+ len(mixed_nuisances) + len(removed_nuisances),
        "n_converted": len(modified_nuisances),
        "n_mixed": len(mixed_nuisances),
        "n_removed": len(removed_nuisances),
    }



def main():
    parser = argparse.ArgumentParser(description="Validate and update a datacard with non-genuine shape nuisances.")
    parser.add_argument("datacard", type=str, help="Path(s) to the datacard(s) to be updated.", nargs='+')
    parser.add_argument("output_path", type=str, help="Path to the directory where replacements will be stored.")
    parser.add_argument("--validation_results_dir", type=str,
                        default="/tmp/jwulff/inference/validation_results/", help="Directory to store validation results.")
    parser.add_argument("--ignore-processes", nargs='*', default=["data_obs", "QCD"], help="Processes to ignore in the datacard.")
    parser.add_argument("--n-threads", type=int, default=4, help="Number of threads for parallel processing.")
    parser.add_argument("--only-remove", action="store_true", help="Only remove nuisances, do not convert shape to lnN.")
    parser.add_argument("--update-mode", choices=["conservative", "loose"], default="conservative",
                        help="Choose update mode: 'conservative' (default) or 'loose'.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for non-genuine shape detection (only used in loose mode).")
    args = parser.parse_args()

    all_stats = {}

    def process_one(datacard_path):
        datacard = Datacard(datacard=Path(datacard_path),
                            ignore_processes=args.ignore_processes)
        if args.update_mode == "conservative":
            stats = conservative_update(datacard,
                                        Path(args.output_path),
                                        args.validation_results_dir,
                                        only_remove=args.only_remove)
        else:
            stats = loose_update(datacard,
                                 Path(args.output_path),
                                 threshold=args.threshold)
        return str(datacard_path), stats

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        results = thread_map(process_one, args.datacard, max_workers=args.n_threads, desc="Processing datacards")

    for datacard_path, stats in results:
        all_stats[datacard_path] = stats

    # Write all stats to JSON
    with open(f"{args.output_path}/update_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nWrote statistics for {len(all_stats)} datacards to {args.output_path}/update_stats.json")

if __name__ == "__main__":
    main()
