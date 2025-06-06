import subprocess
import os

from pathlib import Path
import json
import re

import uproot
from hist import Hist
import numpy as np

from typing import List, Dict, Tuple

from dataclasses import dataclass, field

@dataclass
class Datacard:
    """
    Dataclass to hold information about a datacard.
    """
    datacard: Path
    ignore_processes: List = field(default_factory=lambda: ["data_obs", "QCD"]) 

    processes: list = field(init=False, default=None)
    all_processes: list = field(init=False, default=None)
    process_to_id: dict = field(init=False, default=None)
    lines: list = field(init=False, default=None)
    nuisance_types: dict = field(init=False, default=None)

    _positions: list = field(init=False, default=None)
    
    def __post_init__(self):
        if not isinstance(self.datacard, Path):
            self.datacard = Path(self.datacard)
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
        self.process_lines = [l for l in self.lines if l.startswith("process")]
        assert len(self.process_lines) == 2, f"Found {len(self.process_lines)} lines for process"
        self.all_processes = self.process_lines[0].strip().split()[1:]
        process_ids = list(map(int, self.process_lines[1].strip().split()[1:]))
        self.processes = [p for p in self.all_processes if not any(ignored in p for ignored in self.ignore_processes)]
        self.process_to_id = {p: i for i, p in zip(process_ids, self.processes) if not any(ignored in p for ignored in self.ignore_processes)}

        # get a nuisance line
        # nuisance type position is one space after the name of the longest nuisance
        longest_nuisance = max(self.get_nuisance_types().keys(), key=len)
        self._positions = [0, len(longest_nuisance) + 2]  # start with the first two positions
        self._positions.extend([self.process_lines[0].index(p) for p in self.all_processes])


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
        return sum(1 for l in self.get_nuisance_types().values() if l == "shape?")
    
    @property
    def positions(self) -> list[int]:
        return self._positions

    def _load_lines(self):
        if self.lines is None:
            with open(self.datacard, "r") as f:
                self.lines = f.readlines()

    def get_nuisance_line(self, nuisance: str) -> dict[str, str]:
        """
        Get the line for a given nuisance in the datacard.
        """
        self._load_lines()
        nuisance_lines = [l for l in self.lines if l.startswith(f"{nuisance} ")]
        if len(nuisance_lines) == 0:
            raise ValueError(f"Nuisance {nuisance} not found in datacard {self.datacard}")
        elif len(nuisance_lines) > 1:
            raise ValueError(f"Found multiple lines for nuisance {nuisance} in datacard {self.datacard}")
        # create a process to entry mapping
        entries = nuisance_lines[0].strip().split()
        entry_dict = {}
        for process, i in self.process_to_id.items():
            entry_dict[process] = entries[i + 2]  # +2 because first two entries are nuisance name and type
        return entry_dict

    def get_nuisance_types(self) -> dict[str, str]:
        """
        get a dictionary of nuisances and their types from the datacard.
        """
        self._load_lines()
        if self.nuisance_types is None:
            nuisance_lines = [l for l in self.lines if any(t in l for t in [" shape ", " lnN ", " shape? "])]
            if len(nuisance_lines) == 0:
                raise ValueError(f"No nuisances found in datacard {self.datacard}")
            self.nuisance_types = {l.split()[0]: l.split()[1] for l in nuisance_lines}
        return self.nuisance_types

    def _extract_shape_rates(self, f, nuisance: str) -> dict[str, Tuple[float, float]]:
        """
        Helper to extract shape rates from an open uproot file handle.
        """
        rates = {}
        for process in self.processes:
            if process in self.ignore_processes:
                continue
            nominal_hist = f[self.dirname][f"{process}"].to_hist()
            up_hist = f[self.dirname][f"{process}__{nuisance}Up"].to_hist()
            down_hist = f[self.dirname][f"{process}__{nuisance}Down"].to_hist()

            # take the integral of the histograms
            nominal_sum = nominal_hist.sum().value
            if nominal_sum <= 1e-6:
                print(f"Warning: Nominal value for process {process} is too small ({nominal_sum}), skipping.")
            up_rate = up_hist.sum().value / nominal_sum
            down_rate = down_hist.sum().value / nominal_sum 
            
            #nominal_vals = nominal_hist.values()
            #up_vals = up_hist.values()
            #down_vals = down_hist.values()
            ## Avoid division by zero
            #with np.errstate(divide='ignore', invalid='ignore'):
                #up_rate = np.nanmean(np.where(nominal_vals != 0, up_vals / nominal_vals, 1.0))
                #down_rate = np.nanmean(np.where(nominal_vals != 0, down_vals / nominal_vals, 1.0))
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
        if nuisance_type not in ["lnN", "shape", "shape?"]:
            raise ValueError(f"Nuisance {nuisance} is not a rate nuisance or a shape nuisance in datacard {self.datacard}")
        if nuisance_type == "lnN":
            nuisance_line = self.get_nuisance_line(nuisance)
            rates = {process: nuisance_line[process] for process in self.process_to_id}
        elif nuisance_type == "shape":
            rates = {}
            # Use the provided file handle, or open if not provided
            if shapes_file_handle is None:
                with uproot.open(self.shapes_file) as f:
                    rates = self._extract_shape_rates(f, nuisance)
            else:
                rates = self._extract_shape_rates(shapes_file_handle, nuisance)
        elif nuisance_type == "shape?":
            raise NotImplementedError(f"Nuisance {nuisance} is a mixed nuisance, not implemented yet.")
        return rates
    
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


    def get_shape_var(self, nuisance, shapes_file_handle=None) -> float:
        """
        Get the max variation of a shape vs the nominal shape for a given nuisance.
        """
        if nuisance not in self.get_nuisance_types():
            raise ValueError(f"Nuisance {nuisance} not found in datacard {self.datacard}")
        nuisance_type = self.get_nuisance_types()[nuisance]
        if nuisance_type != "shape":
            raise ValueError(f"Nuisance {nuisance} is not a shape nuisance in datacard {self.datacard}")
        
        # Use the provided file handle, or open if not provided
        if shapes_file_handle is None:
            with uproot.open(self.shapes_file) as f:
                nominal_hist = f[self.dirname][f"{self.processes[0]}"].to_hist()
                up_hist = f[self.dirname][f"{self.processes[0]}__{nuisance}Up"].to_hist()
                down_hist = f[self.dirname][f"{self.processes[0]}__{nuisance}Down"].to_hist()
        else:
            nominal_hist = shapes_file_handle[self.dirname][f"{self.processes[0]}"].to_hist()
            up_hist = shapes_file_handle[self.dirname][f"{self.processes[0]}__{nuisance}Up"].to_hist()
            down_hist = shapes_file_handle[self.dirname][f"{self.processes[0]}__{nuisance}Down"].to_hist()

        # Calculate the max variation
        nominal_vals = nominal_hist.values()
        up_vals = up_hist.values()
        down_vals = down_hist.values()

        # get the largest relative variation
        up_var = np.nanmax(np.where(nominal_vals != 0, np.abs(up_vals - nominal_vals) / nominal_vals, 0))
        down_var = np.nanmax(np.where(nominal_vals != 0, np.abs(down_vals - nominal_vals) / nominal_vals, 0))
        return max(up_var, down_var)
    

    def _get_sum_shape_var(self, nuisance, process, shapes_file_handle) -> tuple[float, float]:
        """
        Get the sum of the differences between the nominal shape and the up/down variations
        (the same way as validateDatacards.py does).
        Returns a tuple of (up_sum, down_sum).
        """
        nominal_hist = shapes_file_handle[f"{self.dirname}/{process}"].to_hist()
        up_hist = shapes_file_handle[f"{self.dirname}/{process}__{nuisance}Up"].to_hist()
        down_hist = shapes_file_handle[f"{self.dirname}/{process}__{nuisance}Down"].to_hist()

        # Calculate the sum of the differences
        nominal_vals = nominal_hist.values()/nominal_hist.sum().value
        up_vals = up_hist.values()/up_hist.sum().value
        down_vals = down_hist.values()/down_hist.sum().value
    
        val_up = np.sum((2*np.abs(up_vals-nominal_vals))/(np.abs(nominal_vals)+np.abs(up_vals)))
        val_down = np.sum((2*np.abs(down_vals-nominal_vals))/(np.abs(nominal_vals)+np.abs(down_vals)))
        return val_up, val_down
    
    
    def get_shape_vars(self, nuisance: str, threshold: float, shapes_file_handle=None) -> list[str]:
        """
        Get the shape variations for a nuisance.
        Returns a dictionary with process names as keys and a tuple of (up_var, down_var) as values.
        Optionally, pass an open shapes_file_handle to avoid reopening.
        """
        nuisance_type = self.get_nuisance_types()[nuisance]
        if nuisance_type != "shape":
            raise ValueError(f"Nuisance {nuisance} is not a shape nuisance in datacard {self.datacard}")
        
        # Use the provided file handle, or open if not provided
        if shapes_file_handle is None:
            with uproot.open(self.shapes_file) as f:
                # just flag processes for which the sum up and sum down is below the threshold
                vars = {process: self._get_sum_shape_var(nuisance, process, f) for process in self.processes}
                return [process for process, (up_var, down_var) in vars.items() if up_var < threshold and down_var < threshold]
                
        else:
            vars = {process: self._get_sum_shape_var(nuisance, process, shapes_file_handle) for process in self.processes}
            return [process for process, (up_var, down_var) in vars.items() if up_var < threshold and down_var < threshold]

    
    def get_shape_hists(self, keys: list[str], shapes_file_handle=None) -> Dict[str, Hist]: 
        """
        Get the shape histograms for a list of nuisances.
        Returns a dictionary with nuisance names as keys and a dictionary of process histograms as values.
        """
        if shapes_file_handle is None:
            with uproot.open(self.shapes_file) as f:
                return {key: f[key].to_hist() for key in keys}
        else:
            return {key: shapes_file_handle[key].to_hist() for key in keys}