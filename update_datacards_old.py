import subprocess
import os
import shutil
import sys
from multiprocessing import Pool, Manager
import argparse

from pathlib import Path
import json
import re

import uproot
import hist
from hist import Hist
import numpy as np

from typing import List, Dict, Tuple
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from ROOT import TFile, gDirectory
from pprint import pp

from dataclasses import dataclass, field


# helper to delete histograms from a ROOT file
def del_histograms(filename: str, dirname: str, hist_names: List[str]) -> bool:
    """
    Importantly, the hist_names have to also include all storage cycles in order to be completely gone.
    """
    #tick = time.time()
    file = TFile(filename, "UPDATE")
    file.cd(dirname)
    for hist in hist_names:
        try:
            gDirectory.Delete(hist)
        except Exception as e:
            print(f"Error deleting histogram {hist} from {dirname} in {filename}: {e}")
            return False
    file.Close()
    #tock = time.time()
    #print(f"Deleted {len(hist_names)} histograms from {dirname} in {filename} in {tock-tick:.2f} seconds")
    return True


# alternative helper to delete histograms from a ROOT file using rmroot
def del_histograms_rmroot(filename: str, dirname: str, hist_names: List[str]) -> bool:
    """
    Delete histograms from a ROOT file using the rmroot command.
    """
    tick = time.time()
    for hist in hist_names:
        command = ["rootrm", f"{filename}:{dirname}/{hist}"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error deleting histograms from {filename}: {result.stderr}")
            return False
    tock = time.time()
    print(f"Deleted {len(hist_names)} histograms from {dirname} in {filename} in {tock-tick:.2f} seconds")
    return True
    

@dataclass
class Datacard:
    """
    Dataclass to hold information about a datacard.
    """
    datacard: Path
    replacement_path: Path
    ignore_processes: List[str] = field(default_factory=lambda: ["data_obs", "QCD"]) 
    _lines: list = field(init=False, default=None)
    _processes: list = field(init=False, default=None)
    _all_processes: list = field(init=False, default=None)
    _process_to_id: dict = field(init=False, default=None)
    _nuisance_types: dict = field(init=False, default=None)
    
    def __post_init__(self):
        if not isinstance(self.datacard, Path):
            self.datacard = Path(self.datacard)
        if not isinstance(self.replacement_path, Path):
            self.replacement_path = Path(self.replacement_path)
        if not self.replacement_path.exists():
            print(f"Replacement path {self.replacement_path} does not exist, creating it.")
            os.makedirs(self.replacement_path, exist_ok=True)
        if not self.datacard.exists():
            raise FileNotFoundError(f"Datacard {self.datacard} does not exist.")
        self.shapes_file = self.datacard.parent / str(self.datacard.name).replace("datacard", "shapes").replace("txt", "root")
        if not self.shapes_file.exists():
            raise FileNotFoundError(f"Shapes file {self.shapes_file} does not exist.")
        
        # match the year, channel, category, category_suffix, sign, isolation, spin, mass from the datacard name
        self.year = re.search(r"_(2016|2016APV|2017|2018)_", self.datacard.stem).group(0).strip("_")
        self.channel = re.search(r"_(etau|mutau|tautau)_", self.datacard.stem).group(0).strip("_")
        self.category = re.search(r"_(resolved1b|resolved2b|boosted)_", self.datacard.stem).group(0).strip("_")
        try:
            self.category_suffix = re.search(r"_(noak8|first|notres2b)_", self.datacard.stem).group(0).strip("_")
        except AttributeError:
            # no category suffix found, set to empty string 
            self.category_suffix = ""
        self.sign = re.search(r"_(os|ss)_", self.datacard.stem).group(0).strip("_")
        self.isolation = re.search(r"_(iso|noniso)_", self.datacard.stem).group(0).strip("_")
        self.spin = int(re.search(r"_(spin)_(0|2)_", self.datacard.stem).group(2))
        self.mass = int(re.search(r"_(mass)_(\d+)", self.datacard.stem).group(2))

        # Load lines once
        self._load_lines()
        # Initialize processes and process_to_id
        self.process_lines = [l for l in self._lines if l.startswith("process")]
        assert len(self.process_lines) == 2, f"Found {len(self.process_lines)} lines for process"
        self._all_processes = self.process_lines[0].strip().split()[1:]
        process_ids = self.process_lines[1].strip().split()[1:]
        self._processes = [p for p in self._all_processes if not any(ignored in p for ignored in self.ignore_processes)]
        self._process_to_id = {p: i for i, p in zip(process_ids, self._processes) if not any(ignored in p for ignored in self.ignore_processes)}

        # get a nuisance line
        br_line = [l for l in self._lines if l.startswith("BR_hbb")][0]
        self.positions = [0, br_line.index("lnN")]
        self.positions.extend([self.process_lines[0].index(p) for p in self._all_processes])


    @property
    def dirname(self) -> str:
        """
        Get the directory name for the datacard.
        """
        return "_".join([
            "cat",
            self.year,
            self.channel,
            self.category,
            self.category_suffix if self.category_suffix != "" else "",
            self.sign,
            self.isolation
        ])

    @property
    def n_nuisances(self) -> int:
        """
        Get the number of nuisances in the datacard.
        """
        return len(self.get_nuisance_types())
    
    @property
    def n_rate(self) -> int:
        """
        Get the number of rate nuisances in the datacard.
        """
        return sum(1 for l in self.get_nuisance_types().values() if l == "lnN")
    
    @property
    def n_shape(self) -> int:
        """
        Get the number of shape nuisances in the datacard.
        """
        return sum(1 for l in self.get_nuisance_types().values() if l == "shape")
    
    @property
    def n_mixed(self) -> int:
        """
        Get the number of mixed nuisances in the datacard.
        """
        return sum(1 for l in self.get_nuisance_types().values() if l == "?")

    def _load_lines(self):
        if self._lines is None:
            with open(self.datacard, "r") as f:
                self._lines = f.readlines()


    def get_processes(self) -> List[str]:
        return self._processes

    def get_all_processes(self) -> List[str]:
        return self._all_processes

    def get_process_to_id(self) -> Dict[str, int]:
        return self._process_to_id

    def get_nuisance_line(self, nuisance: str) -> dict[str, str]:
        """
        Get the line for a given nuisance in the datacard.
        """
        self._load_lines()
        nuisance_lines = [l for l in self._lines if l.startswith(nuisance)]
        if len(nuisance_lines) == 0:
            raise ValueError(f"Nuisance {nuisance} not found in datacard {self.datacard}")
        elif len(nuisance_lines) > 1:
            raise ValueError(f"Found multiple lines for nuisance {nuisance} in datacard {self.datacard}")
        # create a process to entry mapping
        entries = nuisance_lines[0].strip().split()
        entry_dict = {}
        for process, i in self.get_process_to_id().items():
            entry_dict[process] = entries[i + 2]  # +2 because first two entries are nuisance name and type
        return entry_dict

    def replace_nuisance_lines(self, modifications: list[tuple[str, str, dict[str, str]]]) -> None:
        """
        Apply multiple nuisance line replacements in one go.
        Each modification is a tuple: (nuisance, nuisance_type, new_entries)
        """
        with open(self.datacard, "r") as f:
            lines = f.readlines()

        for nuisance, nuisance_type, new_entries in modifications:
            nuisance_lines = [i for i, l in enumerate(lines) if l.startswith(nuisance+" ")]
            if len(nuisance_lines) == 0:
                raise ValueError(f"Nuisance {nuisance} not found in datacard {self.datacard}")
            elif len(nuisance_lines) > 1:
                raise ValueError(f"Found multiple lines for nuisance {nuisance} in datacard {self.datacard}")
            processes = self.get_processes()
            if len(new_entries) != len(processes):
                raise ValueError(f"Expected {len(processes)} entries for nuisance {nuisance}, but got {len(new_entries)}")
            line_index = [l.split()[0] for l in lines].index(nuisance)
            # check if all new entries are empty -> line can be removed
            #if all(entry == "-" for entry in new_entries.values()):
                ## remove the line
                #lines.pop(line_index)
                #continue
            original_line = lines[line_index]
            new_line = [nuisance, nuisance_type]
            process_entries = ["-" for _ in self._all_processes] # ignore processes will get a "-" by default
            for process, entry in new_entries.items():
                if process not in processes:
                    raise ValueError(f"Process {process} not found in datacard {self.datacard}")
                process_entries[processes.index(process)] = entry
            new_line.extend(process_entries)
            spaces = [self.positions[i+1] - (self.positions[i]+len(new_line[i])) for i in range(len(new_line)-1)]
            new_line = "".join([f"{new_line[i]}{' ' * spaces[i]}" for i in range(len(new_line)-1)]) + f"{new_line[-1]}\n"
            lines[line_index] = new_line
        new_datacard_path = self.replacement_path / self.datacard.name
        with open(new_datacard_path, "w") as f:
            f.writelines(lines)

    def get_nuisance_types(self) -> dict[str, str]:
        """
        get a dictionary of nuisances and their types from the datacard.
        """
        self._load_lines()
        if self._nuisance_types is None:
            nuisance_lines = [l for l in self._lines if any(t in l for t in [" shape ", " lnN ", " shape? "])]
            if len(nuisance_lines) == 0:
                raise ValueError(f"No nuisances found in datacard {self.datacard}")
            self._nuisance_types = {l.split()[0]: l.split()[1] for l in nuisance_lines}
        return self._nuisance_types

    def _extract_shape_rates(self, f, nuisance: str) -> dict[str, Tuple[float, float]]:
        """
        Helper to extract shape rates from an open uproot file handle.
        """
        processes = self.get_processes()
        rates = {}
        for process in processes:
            if process in self.ignore_processes:
                continue
            nominal_hist = f[self.dirname][f"{process}"].to_hist()
            up_hist = f[self.dirname][f"{process}__{nuisance}Up"].to_hist()
            down_hist = f[self.dirname][f"{process}__{nuisance}Down"].to_hist()
            # Vectorized calculation
            nominal_vals = nominal_hist.values()
            up_vals = up_hist.values()
            down_vals = down_hist.values()
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                up_rate = np.nanmean(np.where(nominal_vals != 0, up_vals / nominal_vals, 1.0))
                down_rate = np.nanmean(np.where(nominal_vals != 0, down_vals / nominal_vals, 1.0))
            rates[process] = (up_rate, down_rate)
        return rates

    def get_rates(self, nuisance: str, shapes_file_handle=None) -> dict[str, Tuple[float, float]]:
        """
        Get the rate for a nuisance from the datacard (in case it's a rate nuisance)
        or from the shapes file (in case it's a shape nuisance).
        Returns a dictionary with process names as keys and a tuple of (up_rate, down_rate) as values.
        Optionally, pass an open shapes_file_handle to avoid reopening.
        """
        nuisance_type = self.get_nuisance_types()[nuisance]
        if nuisance_type not in ["lnN", "shape", "?"]:
            raise ValueError(f"Nuisance {nuisance} is not a rate nuisance or a shape nuisance in datacard {self.datacard}")
        if nuisance_type == "lnN":
            nuisance_line = self.get_nuisance_line(nuisance)
            rates = {process: (nuisance_line.split()[1:][i]) for process, i in self.get_process_to_id().items()}
        elif nuisance_type == "shape":
            rates = {}
            # Use the provided file handle, or open if not provided
            if shapes_file_handle is None:
                with uproot.open(self.shapes_file) as f:
                    rates = self._extract_shape_rates(f, nuisance)
            else:
                rates = self._extract_shape_rates(shapes_file_handle, nuisance)
        elif nuisance_type == "?":
            raise NotImplementedError(f"Nuisance {nuisance} is a mixed nuisance, not implemented yet.")
        return rates
    
    def copy_shapes_file(self) -> Path:
        """
        Copy the shapes file to the output directory.
        Returns the path to the copied shapes file.
        """
        output_shapes_file = self.replacement_path / self.shapes_file.name
        shutil.copy(self.shapes_file, output_shapes_file)
        return output_shapes_file
    
    def validate(self, validation_results_dir: Path) -> str:
        """
        Validate the datacard using the ValidateDatacards.py script.
        Returns the path to the validation results JSON file.
        """
        if not isinstance(validation_results_dir, Path):
            validation_results_dir = Path(validation_results_dir)
        # Check if the validation results directory exists
        if not validation_results_dir.exists():
            print(f"Validation results directory {validation_results_dir} does not exist.")
            os.makedirs(validation_results_dir, exist_ok=True)

        # Construct the command to run the script
        command = ["ValidateDatacards.py", self.datacard, "--jsonFile", f"{validation_results_dir}/{self.datacard.stem}.json"]
        # Run the command and capture the output
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Check if the command was successful
            if result.returncode == 0:
                #print(f"Validation successful for {self.datacard}")
                return f"{validation_results_dir}/{self.datacard.stem}.json"
            else:
                print(f"Validation failed for {self.datacard}: {result.stderr}")
                return False 
        except Exception as e:
            print(f"Error running validation for {self.datacard}: {e}")
            return False

        
def update_datacard(datacard: Datacard,
                    validation_results_dir: str) -> dict:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    """
    def get_rate_string(up_rate, down_rate):
        if np.abs(1-up_rate) < 0.01 and np.abs(1-down_rate) < 0.01:
            return "-"
        elif np.abs(1-up_rate) > 0.01 and np.abs(1-down_rate) > 0.01:
            return f"{np.min((up_rate, down_rate)):.3f}/{np.max((up_rate, down_rate)):.3f}"
        else:
            rate = np.max((down_rate, up_rate))
            return f"{rate:.3f}"

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
            return {}
        else:
            small_shape_effects = validation_results_json["smallShapeEff"]
            cat_name = next(iter(small_shape_effects[next(iter(small_shape_effects))]))
            small_shape_effects = {nuisance: small_shape_effects[nuisance][cat_name] for nuisance in small_shape_effects}
            uncert_templ_same = validation_results_json.get("uncertTemplSame", {})
            overlapping = set(small_shape_effects.keys()).intersection(set(uncert_templ_same.keys())) 
            if overlapping:
                print(f"Found {len(overlapping)} nuisances with both smallShapeEffect and \
uncertTemplSame warnings in validation results for {datacard.datacard.name}")

            processes = datacard.get_processes()
            to_delete = []
            n_updated_rate = 0
            n_updated_mixed = 0
            modifications = []  # Collect all modifications here

            for nuisance in small_shape_effects:
                if len(small_shape_effects[nuisance]) == len(processes):
                    # can be remodelled as a rate nuisance
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    rates = {process: get_rate_string(*rates[process]) for process in rates}
                    modifications.append((nuisance, "lnN", rates))
                    n_updated_rate += 1
                elif len(small_shape_effects[nuisance]) < len(processes):
                    rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                    keep_processes =  set(processes) - set(small_shape_effects[nuisance].keys())
                    rates = {process: get_rate_string(*rates[process]) for process in small_shape_effects[nuisance].keys()}
                    if all(rate == "-" for rate in rates.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "shape?"
                        continue
                    rates.update({process: "1" for process in keep_processes})
                    modifications.append((nuisance, nuisance_type, rates))
                    hist_names = [f"{datacard.dirname}/{process}__{nuisance}Up" for process in small_shape_effects[nuisance].keys()]
                    hist_names.extend([f"{datacard.dirname}/{process}__{nuisance}Down" for process in small_shape_effects[nuisance].keys()])
                    delete_shapes = [k.split("/")[-1] for k in shapes_keys if any(hist_name in k for hist_name in hist_names)]
                    to_delete.extend(delete_shapes)
                    n_updated_mixed += 1
                else:
                    raise ValueError(f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}")
            print(f"updated {n_updated_rate} rate nuisances and {n_updated_mixed} mixed nuisances.")

            # Apply all modifications at once
            if modifications:
                datacard.replace_nuisance_lines(modifications)

            # now delete the shapes from the shapes file
            if to_delete:
                shapes_file = datacard.copy_shapes_file()
                del_ret = del_histograms(str(shapes_file), datacard.dirname, to_delete)
                #del_ret = del_histograms_rmroot(str(shapes_file), datacard.dirname, to_delete)
                if not del_ret:
                    print(f"Failed to delete histograms {to_delete} from shapes file {shapes_file}.")
                    return False
            else:
                shapes_file = datacard.copy_shapes_file()
        results = {
            "datacard": datacard.datacard.name,
            "n_nuisances": datacard.n_nuisances,
            "n_rate": datacard.n_rate,
            "n_shape": datacard.n_shape,
            "n_mixed": datacard.n_mixed,
            "n_updated": n_updated_rate + n_updated_mixed,
            "n_updated_rate": n_updated_rate,
            "n_updated_mixed": n_updated_mixed, 
        }
        return results

def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs ValidateDatacards.py and updates the datacards "
            "with non-genuine shape nuisances modelled as rate nuisances.")
    )
    parser.add_argument("--input_datacard", type=str, nargs="+", help="Path(s) to the input datacard(s)")
    parser.add_argument("--output_dir", type=str, required=False, help="Path to the output directory")
    parser.add_argument("--validation_results_dir",
                        type=str,
                        default="/tmp/jwulff/inference/validation_results",
                        help="Path to a tmp validation results directory")
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum number of parallel jobs (default: 1)")
    parser.add_argument("--stats-json", type=str, required=False, help="Output JSON file for remodelled nuisance statistics")
    return parser


def main(input_datacards: list,
         output_dir: Path,
         validation_results_dir: Path,
         max_jobs: int = 1,
         stats_json: str = None):
    """
    Main function to run the script.
    """
    import threading

    stats_dict = {}
    stats_lock = threading.Lock()

    def process_one(input_datacard):
        #print("Processing datacard:", input_datacard)
        input_datacard = Path(input_datacard)
        if not input_datacard.exists():
            print(f"Input datacard {input_datacard} does not exist. Skipping.")
            return

        this_output_dir = Path(output_dir)
        if not this_output_dir.exists():
            os.makedirs(this_output_dir, exist_ok=True)

        if not isinstance(validation_results_dir, Path):
            vrd = Path(validation_results_dir)
        else:
            vrd = validation_results_dir
        if not vrd.exists():
            os.makedirs(vrd, exist_ok=True)


        # Create Datacard instance
        datacard_obj = Datacard(datacard=input_datacard, replacement_path=this_output_dir)
        local_stats = update_datacard(datacard_obj, validation_results_dir=vrd)

        # Merge local_stats into shared stats_dict
        if local_stats:
            with stats_lock:
                # Optionally, add timings to the stats for this datacard/category
                stats_dict[datacard_obj.dirname] = local_stats
        

    if max_jobs > 1:
        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=max_jobs) as executor:
            list(tqdm(executor.map(process_one, input_datacards), total=len(input_datacards), desc="Processing datacards"))
    else:
        from tqdm import tqdm
        for input_datacard in tqdm(input_datacards, desc="Processing datacards"):
            process_one(input_datacard)

    # Write stats_dict to JSON if requested
    if stats_json:
        with open(stats_json, "w") as f:
            json.dump(stats_dict, f, indent=2)
        print(f"Remodelled nuisance statistics written to {stats_json}")

#import cProfile
#import pstats

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if not args.output_dir:
        raise ValueError("No output directory given")
    # Profile the main function
    #profiler = cProfile.Profile()
    #profiler.enable()
    main(
        args.input_datacard,
        args.output_dir,
        args.validation_results_dir,
        max_jobs=args.max_jobs,
        stats_json=args.stats_json
    )
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats(30)  # Show top 30 functions by cumulative time
