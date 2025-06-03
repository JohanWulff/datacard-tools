from datacard_parser import Datacard

from typing import List, Dict, Tuple
from pathlib import Path
import shutil

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import collections
import json
import numpy as np
import uproot
import subprocess
import os


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
        original_line = lines[line_index]
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


def update_datacard(datacard: Datacard,
                    output_path: Path,
                    validation_results_dir: str) -> dict:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    Also returns statistics about nuisance types before and after the update.
    """
    def get_rate_string(up_rate, down_rate):
        if np.abs(1-up_rate) < 0.01 and np.abs(1-down_rate) < 0.01:
            return "-"
        elif np.abs(1-up_rate) > 0.01 and np.abs(1-down_rate) > 0.01:
            return f"{down_rate:.3f}/{up_rate:.3f}"
        else:
            return f"{np.max((down_rate, up_rate)):.3f}"

    # --- STATISTICS: BEFORE UPDATE ---
    nuisance_types_before = datacard.get_nuisance_types()  # {nuisance: type}
    stats_before = collections.Counter(nuisance_types_before.values())

    modified_nuisances = set()
    untouched_nuisances = set(nuisance_types_before.keys())

    # New counters
    n_shapeq = 0
    n_removed = 0

    with uproot.open(datacard.shapes_file) as f:
        shapes_keys = f.keys()
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
            remove_unused_shapes = False
            for nuisance in small_shape_effects:
                if len(small_shape_effects[nuisance]) == len(datacard.processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    rates = {process: get_rate_string(*rates[process]) for process in rates}
                    modifications.append((nuisance, "lnN", rates))
                    modified_nuisances.add(nuisance)
                    # Check if all entries are "-"
                    if all(rate == "-" for rate in rates.values()):
                        n_removed += 1
                elif len(small_shape_effects[nuisance]) < len(datacard.processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    keep_processes =  set(datacard.processes) - set(small_shape_effects[nuisance].keys())
                    rates = {process: get_rate_string(*rates[process]) for process in small_shape_effects[nuisance].keys()}
                    if all(rate == "-" for rate in rates.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "shape?"
                        n_shapeq += 1
                        # remove_unused_shapes = True
                        continue # skip modification of mixed nuisances for now 
                    rates.update({process: "1" for process in keep_processes})
                    modifications.append((nuisance, nuisance_type, rates))
                    modified_nuisances.add(nuisance)
                else:
                    raise ValueError(f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}")

            untouched_nuisances -= modified_nuisances

            # Apply all modifications at once
            if modifications:
                replace_nuisance_lines(datacard, output_path, modifications)

            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            shutil.copy(datacard.shapes_file, output_shapes_file)
            if remove_unused_shapes:
                retcode = subprocess.run(["remove_unused_shapes.py", str(datacard.datacard), "*,*", "--inplace-shapes"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if retcode.returncode != 0:
                    print(f"Error removing unused shapes: {retcode.stderr}")
                    return False

    # --- STATISTICS: AFTER UPDATE ---
    updated_datacard_path = Path(output_path) / datacard.datacard.name
    updated_datacard = Datacard(datacard=updated_datacard_path, ignore_processes=datacard.ignore_processes)
    nuisance_types_after = updated_datacard.get_nuisance_types()
    stats_after = collections.Counter(nuisance_types_after.values())

    # --- STATISTICS: UNTOUCHED ---
    untouched_types = [nuisance_types_before[n] for n in untouched_nuisances]
    stats_untouched = collections.Counter(untouched_types)

    # --- PRINT STATISTICS ---
    print("\nNuisance statistics:")
    print("Before update:", dict(stats_before))
    print("After update: ", dict(stats_after))
    print("Untouched:    ", dict(stats_untouched))
    print(f"Modified:     {len(modified_nuisances)}")
    print(f"Untouched:    {len(untouched_nuisances)}")
    print(f'Updated to "shape?": {n_shapeq}')
    print(f'Removed: {n_removed}\n')

    return {
        "before": dict(stats_before),
        "after": dict(stats_after),
        "untouched": dict(stats_untouched),
        "n_modified": len(modified_nuisances),
        "n_untouched": len(untouched_nuisances),
        "n_shapeq": n_shapeq,
        "n_removed": n_removed,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate and update a datacard with non-genuine shape nuisances.")
    parser.add_argument("datacard", type=str, help="Path(s) to the datacard(s) to be updated.", nargs='+')
    parser.add_argument("output_path", type=str, help="Path to the directory where replacements will be stored.")
    parser.add_argument("--validation_results_dir", type=str,
                        default="/tmp/jwulff/inference/validation_results/", help="Directory to store validation results.")
    parser.add_argument("--ignore_processes", nargs='*', default=["data_obs", "QCD"], help="Processes to ignore in the datacard.")
    parser.add_argument("--stats_json", type=str, default="update_stats.json", help="Output JSON file for statistics.")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of threads for parallel processing.")
    args = parser.parse_args()

    all_stats = {}

    def process_one(datacard_path):
        datacard = Datacard(datacard=Path(datacard_path),
                            ignore_processes=args.ignore_processes)
        stats = update_datacard(datacard,
                                Path(args.output_path),
                                args.validation_results_dir)
        return str(datacard_path), stats

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        results = list(tqdm(executor.map(process_one, args.datacard), total=len(args.datacard), desc="Processing datacards"))

    for datacard_path, stats in results:
        all_stats[datacard_path] = stats

    # Write all stats to JSON
    with open(args.stats_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nWrote statistics for {len(all_stats)} datacards to {args.stats_json}")

if __name__ == "__main__":
    main()