from datacard_parser import Datacard

from typing import List, Optional
from pathlib import Path
import shutil
import json
import subprocess
import tempfile

import numpy as np
import re
import uproot
import os

from hist import Hist
import hist


def get_rate_string(nominal_yield, up_rate, down_rate):
    if nominal_yield <= 1e-3:
        return "-"
    if np.abs(1 - up_rate) < 0.01 and np.abs(1 - down_rate) < 0.01:
        return "-"
    elif np.abs(up_rate - down_rate) < 0.01:
        return f"{np.max((down_rate, up_rate)):.3f}"
    else:
        return f"{down_rate:.3f}/{up_rate:.3f}"


def plot_variation(
    datacard: Datacard,
    nuisance: str,
    process: str,
    output_dir: Path,
    y_log: bool = True,
    binning: str = "numbers",
) -> bool:
    #plot_arg = f"{datacard.dirname},{process},{nuisance}"
    #plot_cmd = [
    #    "plot_datacard_shapes.py",
    #    str(datacard.datacard),
    #    "--y-log",
    #    "--binning",
    #    "numbers",
    #    "--directory",
    #    str(output_dir),
    #    plot_arg,
    #]
    ##print(f"Running plotting command: {' '.join(plot_cmd)}")
    #result = subprocess.run(plot_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #if result.returncode != 0:
    #    print(
    #        f"Warning: plotting failed for {plot_arg} in {datacard.datacard.name}: "
    #        f"{result.stderr.strip()}"
    #    )
    #    return False
    #return True

    def equal_width_hist(h: Hist) -> Hist:
        if np.all(h.axes[0].widths == h.axes[0].widths[0]):
            return h
        else:
            new_hist = Hist(hist.axis.Regular(h.axes[0].size,
                                    h.axes[0].edges[0],
                                    h.axes[0].edges[-1], name=h.axes[0].name), storage=h.storage_type)
            new_hist.view()[:] = h.view()
            return new_hist


    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    
    with uproot.open(datacard.shapes_file) as f:
        key_up = f"{datacard.dirname}/{process}__{nuisance}Up"
        key_down = f"{datacard.dirname}/{process}__{nuisance}Down"
        if key_up not in f or key_down not in f:
            print(f"Warning: shape keys {key_up} or {key_down} not found in shapes file for datacard {datacard.datacard.name}")
            return False
        hist_up = f[key_up].to_hist()
        hist_down = f[key_down].to_hist()
        nominal_hist = f[f"{datacard.dirname}/{process}"].to_hist()
    if binning == "numbers":
        hist_up = equal_width_hist(hist_up)
        hist_down = equal_width_hist(hist_down)
        nominal_hist = equal_width_hist(nominal_hist)
    change_up, change_down = hist_up.sum().value/nominal_hist.sum().value, hist_down.sum().value/nominal_hist.sum().value
    fig, ax = plt.subplots()
    label_factor_up = f"{change_up:.1f}" if np.abs(change_up - 1) > 0.01 else "1"
    label_factor_down = f"{change_down:.1f}" if np.abs(change_down - 1) > 0.01 else "1"
    hep.histplot(nominal_hist.values(), histtype="step", label=f"Nominal, Yield {nominal_hist.sum().value:.1e}", ax=ax, color="black")
    hep.histplot(hist_up.values(), histtype="step", label=f"{nuisance.replace('_', '-')} Up, x{label_factor_up}", ax=ax, color="C0")
    hep.histplot(hist_down.values(), histtype="step", label=f"{nuisance.replace('_', '-')} Down, x{label_factor_down}", ax=ax, color="C1")
    ax.set_title(f"{process.replace('_', '-')} - {nuisance.replace('_', '-')}")
    ax.legend()
    ax.set_ylabel("Events")
    ax.set_yscale("log" if y_log else "linear")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{datacard.dirname}__{process}__{nuisance}.pdf")
    plt.close() 

    
def can_remodel_as_rate(rates: tuple[float, float], nominal_yield: float, min_yield: float = 1e-3) -> bool:
    """
    Check if a nuisance can be safely remodeled as a rate nuisance.
    
    Guards:
    1. Nominal yield must be > min_yield (default 1e-3)
    2. Rate factors must be within valid lnN range (0.01 to 1.99)
    
    Returns: True if safe to remodel, False otherwise
    """
    up_rate, down_rate = rates
    
    # commented out becuase for now we switch off all nuisances for processes with nominal yield <= 1e-3, 
    # see get_rate_string()
    ## Guard 1: Check nominal yield
    #if nominal_yield <= min_yield:
    #    return False
    
    # Guard 2: Check if rate factors are within valid lnN range
    max_rate = max(up_rate, down_rate)
    
    # keep as a shape if beyond a factor of 1.99
    if max_rate > 1.99:
        return False
    
    return True


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


def replace_nuisance_lines(
    old_card: Datacard,
    new_card: Path,
    modifications: dict[str, tuple[str, dict[str, str]]], 
) -> None:
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

    for nuisance, (nuisance_type, new_entries) in modifications.items():
        nuisance_lines = [i for i, l in enumerate(lines) if l.startswith(nuisance + " ")]
        if len(nuisance_lines) == 0:
            raise ValueError(f"Nuisance {nuisance} not found in datacard {old_card.datacard}")
        elif len(nuisance_lines) > 1:
            raise ValueError(f"Found multiple lines for nuisance {nuisance} in datacard {old_card.datacard}")
        if len(new_entries) != len(old_card.processes):
            raise ValueError(
                f"Expected {len(old_card.processes)} entries for nuisance {nuisance}, but got {len(new_entries)}"
            )
        line_index = [l.split()[0] for l in lines].index(nuisance)
        # check if all new entries are empty -> line can be removed
        if all(entry == "-" for entry in new_entries.values()):
            # remove the line
            lines.pop(line_index)
            continue
        new_line = [nuisance, nuisance_type]
        process_entries = ["-" for _ in old_card.all_processes]  # ignored processes get "-" by default
        for process, entry in new_entries.items():
            if process not in old_card.processes:
                raise ValueError(f"Process {process} not found in datacard {old_card.datacard}")
            process_entries[old_card.processes.index(process)] = entry
        new_line.extend(process_entries)
        spaces = [old_card.positions[i + 1] - (old_card.positions[i] + len(new_line[i])) for i in range(len(new_line) - 1)]
        new_line = "".join([f"{new_line[i]}{' ' * spaces[i]}" for i in range(len(new_line) - 1)]) + f"{new_line[-1]}\n"
        lines[line_index] = new_line
    new_datacard_path = new_card / old_card.datacard.name
    with open(new_datacard_path, "w") as f:
        f.writelines(lines)


def conservative_update(
    datacard: Datacard,
    output_path: Path,
    validation_results_dir: Optional[str] = None,
    check_uncert_over: float = 2.0,
    plot_output_dir: Optional[str] = None,
) -> dict:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    If only_remove is True, only remove nuisances, do not convert to lnN.
    small shape effects are identified via ValidateDatacards.py smallShapeEff output.
    If plot_output_dir is provided, also plots large normalization effects from ValidateDatacards.py.
    Nuisances with a nominal yield <= 1e-3 in all affected processes are considered empty and are dropped 
    """

    def parse_validation_section(validation_json: dict, section_name: str) -> dict[str, dict]:
        """
        just removes the "cat_20xx_channel_category_suffix_os_iso" key from the dict
        """
        section = validation_json.get(section_name, {})
        if not section:
            return {}
        first_nuisance = next(iter(section))
        if not section[first_nuisance]:
            return {}
        first_cat = next(iter(section[first_nuisance]))
        return {nuisance: section[nuisance].get(first_cat, {}) for nuisance in section}

    # Initialize plot stats and dropped nuisances tracking
    n_updated_nuisances = 0
    dropped_nuisances = []
    validation_json_path = datacard.validate(validation_results_dir,
                                             check_uncert_over)
    with open(validation_json_path, "r") as vf:
        validation_results_json = json.load(vf)

    small_shape_effects = parse_validation_section(validation_results_json, "smallShapeEff")
    large_norm_effects = parse_validation_section(validation_results_json, "largeNormEff")

    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = {
            re.sub(";\d", "", key)
            for key, cname in cnames.items()
            if key.startswith(datacard.dirname) and cname == "TH1D"
        }

        # Get nominal yields for all processes once
        nominal_yields = datacard.get_nominal_yields(f)

        # considering a process as empty if the summed nominal yield is <= 1e-3
        empty_yields = {proc for proc, yield_ in nominal_yields.items() if yield_ <= 1e-3}

        if not small_shape_effects:
            if not large_norm_effects:
                if len(empty_yields) == 0:
                    print((f"No empty nominal processes and no large/small "
                          f"effects found for datacard {datacard.datacard.name}. No modifications applied."))
                    output_datacard_path = Path(output_path) / datacard.datacard.name
                    shutil.copy(datacard.datacard, output_datacard_path)
                    output_shapes_file = Path(output_path) / datacard.shapes_file.name
                    shutil.copy(datacard.shapes_file, output_shapes_file)
                    return {
                        "n_updated_nuisances": 0,
                        "removed_shape_keys": [],
                        "dropped_nuisances": [],
                    }

        modifications = {}
        remove_unused_shapes = set()
        for nuisance in small_shape_effects:
            # Check if all processes can be remodeled as rates
            rates = datacard.get_rates(nuisance, shapes_file_handle=f)
            processes_to_remodel = set([k for k in small_shape_effects[nuisance].keys() if can_remodel_as_rate(rates[k], nominal_yields.get(k, 0))])
            if processes_to_remodel:
                if len(processes_to_remodel) == len(datacard.processes):
                    rate_strings = {process: get_rate_string(nominal_yields[process], *rates[process]) for process in datacard.processes}
                    if all(rate == "-" for rate in rate_strings.values()):
                        dropped_nuisances.append(nuisance)
                    # all processes can be remodeled as rates, so we remodel the entire nuisance as lnN
                    modifications[nuisance] = ("lnN", rate_strings)
                    # Only remove shapes if we actually remodeled the nuisance
                    remove_unused_shapes.update({
                        f"{datacard.dirname}/{p}_{nuisance}{ud}"
                        for p in processes_to_remodel 
                        for ud in ["Up", "Down"]
                    })
                elif len(processes_to_remodel) < len(datacard.processes):
                    keep_processes = set(datacard.processes) - processes_to_remodel
                    rate_strings = {process: get_rate_string(nominal_yields[process], *rates[process]) for process in processes_to_remodel}
                    if all(rate == "-" for rate in rate_strings.values()):
                        nuisance_type = "shape"
                    else:
                        nuisance_type = "shape?"
                    rate_strings.update({process: "1" for process in keep_processes})
                    modifications[nuisance] = (nuisance_type, rate_strings)
                    # Only remove shapes if we actually remodeled the nuisance
                    remove_unused_shapes.update({
                        f"{datacard.dirname}/{p}_{nuisance}{ud}"
                        for p in processes_to_remodel 
                        for ud in ["Up", "Down"]
                    })
                else:
                    raise ValueError(
                        f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}"
                    )
                n_updated_nuisances += 1
        keep_keys -= remove_unused_shapes
    
        #for nuisance in large_norm_effects:
        #    processes = large_norm_effects[nuisance].keys()
        #    processes_to_drop = set(processes) & empty_yields
        #    if processes_to_drop:
        #        if nuisance in modifications:
        #            modifications[nuisance][1].update({process: "-" for process in processes_to_drop})
        #        else:
        #            modifications[nuisance] = ("shape", {process: "-" for process in processes_to_drop})
        #        n_updated_nuisances += 1

        if empty_yields:
            for nuisance in datacard.get_nuisance_types():
                if nuisance in modifications:
                    modifications[nuisance][1].update({process: "-" for process in empty_yields})
                else:
                    process_entries = datacard.get_nuisance_line(nuisance)
                    process_entries.update({process: "-" for process in empty_yields})
                    modifications[nuisance] = ("shape", process_entries)
                n_updated_nuisances += 1


        if modifications:
            replace_nuisance_lines(datacard, output_path, modifications)

        output_shapes_file = Path(output_path) / datacard.shapes_file.name
        shutil.copy(datacard.shapes_file, output_shapes_file)
        if remove_unused_shapes:
            with uproot.recreate(output_shapes_file) as new_shapes_file:
                hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
                for key, hist in hists.items():
                    try:
                        #hist = update_bugged_hist(hist)
                        new_shapes_file[key] = hist
                    except ValueError:
                        print(
                            f"Warning: Found non-finite values in histogram {key} for datacard {datacard.datacard.name}. "
                            "Keeping original histogram without update."
                        )
                        print(f"Take a look at the shapes file {datacard.shapes_file} for more details.")
        else:
            shutil.copy(datacard.shapes_file, output_shapes_file)

    # In conservative mode, plot largeNormEff entries to be able to cross-check later 
    if large_norm_effects and plot_output_dir:
        plot_dir_for_large_norm = Path(plot_output_dir) / f"spin_{datacard.spin}_mass_{datacard.mass}"
        # Plot only largeNormEff entries that affect empty-nominal bins
        for nuisance in large_norm_effects:
            for process in large_norm_effects[nuisance].keys():
                #print(
                #    f"Plotting largeNormEff nuisance {nuisance} for process {process} in datacard {datacard.datacard.name}..."
                #)
                plot_variation(datacard, nuisance, process, plot_dir_for_large_norm)

    return {
        "n_updated_nuisances": n_updated_nuisances, 
        "removed_shape_keys": sorted(remove_unused_shapes),
        "dropped_nuisances": dropped_nuisances,
    }

