from datacard_parser import Datacard

from typing import List
from pathlib import Path
import shutil

import json
import numpy as np
import re
import uproot
import os

from hist import Hist


def get_rate_string(up_rate, down_rate):
    if np.abs(1 - up_rate) < 0.01 and np.abs(1 - down_rate) < 0.01:
        return "-"
    # ignoring this case for now in order to avoid that shape/ bug with asym lnN's
    # elif np.abs(1-up_rate) > 0.01 and np.abs(1-down_rate) > 0.01:
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


def replace_nuisance_lines(
    old_card: Datacard,
    new_card: Path,
    modifications: list[tuple[str, str, dict[str, str]]],
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

    for nuisance, nuisance_type, new_entries in modifications:
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
    validation_results_dir: str,
    only_remove: bool = False,
) -> None:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    If only_remove is True, only remove nuisances, do not convert to lnN.
    """

    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = {
            re.sub(";\d", "", key)
            for key, cname in cnames.items()
            if key.startswith(datacard.dirname) and cname == "TH1D"
        }
        validation_results = datacard.validate(validation_results_dir)

        if not validation_results:
            raise ValueError(f"Validation failed for datacard {datacard.datacard.name}. Cannot proceed with update.")

        with open(validation_results, "r") as vf:
            validation_results_json = json.load(vf)

        if "smallShapeEff" not in validation_results_json:
            print(f"No small shape effects found for datacard {datacard.datacard.name}. No update necessary.")
            output_datacard_path = Path(output_path) / datacard.datacard.name
            shutil.copy(datacard.datacard, output_datacard_path)
            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            shutil.copy(datacard.shapes_file, output_shapes_file)
            return {
                "updated_nuisances": {},
                "n_updated_nuisances": 0,
                "removed_shape_keys": [],
            }

        small_shape_effects = validation_results_json["smallShapeEff"]
        cat_name = next(iter(small_shape_effects[next(iter(small_shape_effects))]))
        small_shape_effects = {nuisance: small_shape_effects[nuisance][cat_name] for nuisance in small_shape_effects}

        modifications = []
        remove_unused_shapes = set()
        for nuisance in small_shape_effects:
            if len(small_shape_effects[nuisance]) == len(datacard.processes):
                rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                rates = {process: get_rate_string(*rates[process]) for process in rates}
                if all(rate == "-" for rate in rates.values()):
                    nuisance_type = "shape"
                else:
                    nuisance_type = "lnN"
                    if only_remove:
                        continue
                modifications.append((nuisance, nuisance_type, rates))
            elif len(small_shape_effects[nuisance]) < len(datacard.processes):
                rates = datacard.get_rates(nuisance, shapes_file_handle=f)
                keep_processes = set(datacard.processes) - set(small_shape_effects[nuisance].keys())
                rates = {process: get_rate_string(*rates[process]) for process in small_shape_effects[nuisance].keys()}
                if all(rate == "-" for rate in rates.values()):
                    nuisance_type = "shape"
                else:
                    nuisance_type = "shape?"
                    continue
                rates.update({process: "1" for process in keep_processes})
                modifications.append((nuisance, nuisance_type, rates))
            else:
                raise ValueError(
                    f"Unexpected number of processes for nuisance {nuisance} in datacard {datacard.datacard.name}"
                )

            remove_unused_shapes = {
                f"{datacard.dirname}/{p}_{nuisance}{ud}"
                for p in small_shape_effects[nuisance].keys()
                for ud in ["Up", "Down"]
            }
        keep_keys -= remove_unused_shapes

        if modifications:
            replace_nuisance_lines(datacard, output_path, modifications)

        updated_nuisances = {
            nuisance: {
                "type": nuisance_type,
                "entries": entries,
            }
            for nuisance, nuisance_type, entries in modifications
        }

        output_shapes_file = Path(output_path) / datacard.shapes_file.name
        shutil.copy(datacard.shapes_file, output_shapes_file)
        if remove_unused_shapes:
            with uproot.recreate(output_shapes_file) as new_shapes_file:
                hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
                for key, hist in hists.items():
                    try:
                        hist = update_bugged_hist(hist)
                        new_shapes_file[key] = hist
                    except ValueError:
                        print(
                            f"Warning: Found non-finite values in histogram {key} for datacard {datacard.datacard.name}. "
                            "Keeping original histogram without update."
                        )
                        print(f"Take a look at the shapes file {datacard.shapes_file} for more details.")
        else:
            shutil.copy(datacard.shapes_file, output_shapes_file)

        return {
            "updated_nuisances": updated_nuisances,
            "n_updated_nuisances": len(updated_nuisances),
            "removed_shape_keys": sorted(remove_unused_shapes),
        }


def update_bugged_hist(hist: Hist) -> None:
    hist_vals, hist_vars = hist.values(), hist.variances()
    if not np.all(np.isfinite(hist_vals)):
        raise ValueError("Found non-finite values in histogram.")
    if not np.all(np.isfinite(hist_vars)):
        print(f"Warning: Found non-finite bin-errors in histogram {hist}")
        hist_vals[~np.isfinite(hist_vals)] = 1e-5
        hist_vars[~np.isfinite(hist_vars)] = 1e-6
    if np.any(mask := (hist_vals < 1e-5)):
        hist_vals[mask] = 1e-5
        hist_vars[mask] = 1e-6
    elif np.any(mask := (hist_vars < 1e-6)):
        hist_vars[mask] = 1e-6
    else:
        return hist
    new_hist = Hist(hist.axes[0], storage=hist.storage_type())
    new_hist.view().value = hist_vals
    new_hist.view().variance = hist_vars
    return new_hist


def loose_update(
    datacard: Datacard,
    output_path: Path,
    threshold: float = 0.01,
) -> None:
    """
    Update the datacard with non-genuine shape nuisances modelled as rate nuisances.
    This is a more relaxed version of conservative_update, which does not check for small shape effects.
    """

    nuisance_types_before = datacard.get_nuisance_types()
    with uproot.open(datacard.shapes_file) as f:
        cnames = f.classnames()
        keep_keys = {
            re.sub(";\d", "", key)
            for key, cname in cnames.items()
            if key.startswith(datacard.dirname) and cname == "TH1D"
        }
        shape_nuisances = [n for n in nuisance_types_before if nuisance_types_before[n] == "shape"]
        modifications = []
        for nuisance in shape_nuisances:
            rates = datacard.get_rates(nuisance, shapes_file_handle=f)
            flagged = datacard.get_shape_vars(nuisance, threshold=threshold, shapes_file_handle=f)
            if len(flagged) == datacard.processes:
                rate_entries = {process: get_rate_string(*rates[process]) for process in rates}
                modifications.append((nuisance, "lnN", rate_entries))
            elif len(flagged) < len(datacard.processes):
                keep_processes = set(datacard.processes) - set(flagged)
                rate_entries = {process: get_rate_string(*rates[process]) for process in flagged}
                if all(rate == "-" for rate in rate_entries.values()):
                    nuisance_type = "shape"
                else:
                    nuisance_type = "shape?"
                rate_entries.update({process: "1" for process in keep_processes})
                modifications.append((nuisance, nuisance_type, rate_entries))

            keep_keys -= {f"{datacard.dirname}/{process}_{nuisance}{ud}" for process in flagged for ud in ["Up", "Down"]}
    replace_nuisance_lines(datacard, output_path, modifications)
    output_shapes_file = Path(output_path) / datacard.shapes_file.name
    shutil.copy(datacard.shapes_file, output_shapes_file)
    with uproot.recreate(output_shapes_file) as new_shapes_file:
        print(f"Keeping {len(keep_keys)}/{len(f.keys())} keys in shapes file {output_shapes_file}")
        hists = datacard.get_shape_hists(keys=list(keep_keys), shapes_file_handle=f)
        for key, hist in hists.items():
            new_shapes_file[key] = hist

    updated_nuisances = {
        nuisance: {
            "type": nuisance_type,
            "entries": entries,
        }
        for nuisance, nuisance_type, entries in modifications
    }
    return {
        "updated_nuisances": updated_nuisances,
        "n_updated_nuisances": len(updated_nuisances),
    }


def fix_large_shapes(
    datacard: Datacard,
    output_path: Path,
):
    """
    Check each shape effect for large shape effects and adjust them to neighboring bins if necessary.
    A shape effect is considered unphysical if it's larger than 100 times the nominal value.
    """

    def smoothen(idx: int, variations: np.ndarray) -> np.ndarray:
        """
        helper to smoothen the variations
        """
        if len(variations) == 1:
            return variations
        if idx == 0 or idx == len(variations) - 1:
            variations[idx] == variations[idx + 1] if idx == 0 else variations[idx - 1]
        else:
            variations[idx] = (variations[idx + 1] + variations[idx - 1]) / 2.0
        return variations

    changed = {}
    with uproot.open(datacard.shapes_file) as f:
        for nuisance in datacard.shape_nuisances:
            for bkgd in datacard.background_processes:
                up_var, down_var = datacard.get_bin_variations(nuisance, bkgd, shapes_file_handle=f)
                problematic_bins_up = up_var > 100
                problematic_bins_down = down_var < 1 / 100
                if any(problematic_bins_up) or any(problematic_bins_down):
                    bin_ids_up = np.where(problematic_bins_up)[0]
                    diff = bin_ids_up[1:] - bin_ids_up[:-1]
                    if np.any(diff == 1):
                        print(
                            f"Warning: Found neighboring large shape effects for nuisance {nuisance} and process {bkgd} "
                            f"in datacard {datacard.datacard.name}"
                        )
                        print("Will not smoothen these bins.")
                        print(f"Up variations: {up_var}")
                        print(f"Down variations: {down_var}")
                    bin_ids_down = np.where(problematic_bins_down)[0]
                    diff = bin_ids_down[1:] - bin_ids_down[:-1]
                    if np.any(diff == 1):
                        print(
                            f"Warning: Found neighboring large shape effects for nuisance {nuisance} and process {bkgd} "
                            f"in datacard {datacard.datacard.name}"
                        )
                        print("Will not smoothen these bins.")
                        print(f"Up variations: {up_var}")
                        print(f"Down variations: {down_var}")
                    corrected_up_var = np.copy(up_var)
                    corrected_down_var = np.copy(down_var)
                    for bins in problematic_bins_up:
                        corrected_up_var = smoothen(bins, corrected_up_var)
                    for bins in problematic_bins_down:
                        corrected_down_var = smoothen(bins, corrected_down_var)

                    up_shape = f[f"{datacard.dirname}/{bkgd}__{nuisance}Up"].to_hist()
                    down_shape = f[f"{datacard.dirname}/{bkgd}__{nuisance}Down"].to_hist()
                    nom_shape = f[f"{datacard.dirname}/{bkgd}"].to_hist()
                    up_shape.view().value = nom_shape.values() * corrected_up_var
                    down_shape.view().value = nom_shape.values() * corrected_down_var
                    changed[f"{datacard.dirname}/{bkgd}_{nuisance}Up"] = up_shape
                    changed[f"{datacard.dirname}/{bkgd}_{nuisance}Down"] = down_shape
        if changed:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            keep_keys = {
                re.sub(";\d", "", key)
                for key, cname in f.classnames().items()
                if key.startswith(datacard.dirname) and cname == "TH1D"
            } - set(changed.keys())
            output_shapes_file = Path(output_path) / datacard.shapes_file.name
            hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
            hists.update(changed)
            with uproot.recreate(output_shapes_file) as new_shapes_file:
                for key, hist in hists.items():
                    new_shapes_file[key] = hist
    return changed


def smoothen_large_shape_effects(
    datacard: Datacard,
    output_path: Path,
):
    """
    Check each shape effect for large shape effects and symmetrize them in case one variation hits the lower bound of 1e-5.
    Also smoothens variations if they are larger than 1000 times the nominal value and no other bin variations
    exceed this threshold. This is then likely just low statistics and one MC event.
    """

    def smoothen(nominal, up, down):
        """
        expecting nominal, up and down to be just one bin
        """
        assert isinstance(nominal, (int, float)) and isinstance(up, (int, float)) and isinstance(down, (int, float))

        symmetric = (up > nominal) and (nominal > down)
        symmetric_inverted = (down > nominal) and (nominal > up)

        if symmetric:
            alpha = up / nominal - 1
            if alpha < 1:
                down = max(1e-5, nominal * (1 - alpha))
            else:
                down = max(1e-5, nominal / alpha)
        elif symmetric_inverted:
            alpha = down / nominal - 1
            if alpha < 1:
                up = max(1e-5, nominal * (1 - alpha))
            else:
                up = max(1e-5, nominal / alpha)
        else:
            diff_up = np.abs(up - nominal)
            diff_down = np.abs(down - nominal)
            if diff_up < diff_down:
                alpha = up / nominal - 1
                if alpha < 1:
                    down = max(1e-5, nominal * (1 - alpha))
                else:
                    down = max(1e-5, nominal / alpha)
            else:
                alpha = down / nominal - 1
                if alpha < 1:
                    up = max(1e-5, nominal * (1 - alpha))
                else:
                    up = max(1e-5, nominal / alpha)
        return up, down

    changed = {}
    with uproot.open(datacard.shapes_file) as f:
        for nuisance in datacard.shape_nuisances:
            for bkgd in datacard.background_processes:
                nominal, up, down = datacard.get_nom_up_down(nuisance, bkgd, shapes_file_handle=f)
                nominal_vals = nominal.values()
                up_vals = up.values()
                down_vals = down.values()
                if np.any(mask := down_vals <= 1e-5):
                    for idx in np.where(mask)[0]:
                        up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                if np.any(mask := up_vals <= 1e-5):
                    for idx in np.where(mask)[0]:
                        up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                mask = up_vals > 1000 * nominal_vals
                for idx in np.where(mask)[0]:
                    up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                mask = up_vals < nominal_vals / 1000
                for idx in np.where(mask)[0]:
                    up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                mask = down_vals > 1000 * nominal_vals
                for idx in np.where(mask)[0]:
                    up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                mask = down_vals < nominal_vals / 1000
                for idx in np.where(mask)[0]:
                    up_vals[idx], down_vals[idx] = smoothen(nominal_vals[idx], up_vals[idx], down_vals[idx])
                up.view().value = up_vals
                down.view().value = down_vals
                changed[f"{datacard.dirname}/{bkgd}_{nuisance}Up"] = up
                changed[f"{datacard.dirname}/{bkgd}_{nuisance}Down"] = down
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        keep_keys = {
            re.sub(";\d", "", key)
            for key, cname in f.classnames().items()
            if key.startswith(datacard.dirname) and cname == "TH1D"
        } - set(changed.keys())
        output_shapes_file = Path(output_path) / datacard.shapes_file.name
        hists = datacard.get_shape_hists(nuisances=list(keep_keys), shapes_file_handle=f)
        hists.update(changed)
        with uproot.recreate(output_shapes_file) as new_shapes_file:
            for key, hist in hists.items():
                new_shapes_file[key] = hist
        output_datacard_path = Path(output_path) / datacard.datacard.name
        shutil.copy(datacard.datacard, output_datacard_path)
        return changed
