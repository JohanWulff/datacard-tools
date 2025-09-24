import argparse
from pathlib import Path
import uproot
import numpy as np
import shutil
import hist
from hist import Hist

from datacard_parser import Datacard

def merge_bins(h, bin_idx, direction):
    values = h.values()
    variances = h.variances() if h.variances() is not None else np.zeros_like(values)
    n_bins = len(values)

    keep = [i for i in range(n_bins) if i != bin_idx]
    new_values = values[keep]
    if direction == "left":
        if bin_idx == 0:
            raise ValueError("Cannot merge leftmost bin to the left.")
        idx_to_merge_into = bin_idx - 1
    elif direction == "right":
        if bin_idx == n_bins - 1:
            raise ValueError("Cannot merge rightmost bin to the right.")
        idx_to_merge_into = bin_idx
    else:
        raise ValueError("Direction must be 'left' or 'right'.")
    new_variances = variances[keep]
    new_values[idx_to_merge_into] += values[bin_idx]
    new_variances[idx_to_merge_into] += variances[bin_idx]

    axes = h.axes
    edges = axes[0].edges
    if direction == "left":
        new_edges = np.delete(edges, bin_idx)
    else:
        new_edges = np.delete(edges, bin_idx+1)
    new_hist = Hist(hist.axis.Variable(new_edges, name=axes[0].name), storage=h.storage_type())
    new_hist.view().value = new_values
    if hasattr(new_hist, "variances") and new_hist.variances() is not None:
        new_hist.view().variance = new_variances
    else:
        new_hist.view().variance = np.zeros_like(new_values)
    return new_hist

def main():
    parser = argparse.ArgumentParser(description="Merge a bin in a datacard's shapes file.")
    parser.add_argument("datacard", type=str, help="Path to the datacard file")
    parser.add_argument("bin", type=int, help="Bin number to merge (0-based index)")
    parser.add_argument("direction", choices=["left", "right"], help="Direction to merge: left or right")
    parser.add_argument("--output-dir", type=str, default="merged_output", help="Output directory")
    parser.add_argument("--in-place", action="store_true", help="If set, change the datacard in place")
    args = parser.parse_args()

    datacard_path = Path(args.datacard)
    if args.in_place:
        output_dir = datacard_path.parent 
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    out_datacard = output_dir / datacard_path.name
    dc = Datacard(datacard_path)
    if not args.in_place:
        shutil.copy(datacard_path, out_datacard)

    shapes_in = dc.shapes_file
    shapes_out = output_dir / shapes_in.name

    hists = {}
    with uproot.open(shapes_in) as fin:
        for k in dc.get_hist_keys():
            old_hist = fin[f"{dc.dirname}/{k}"].to_hist()
            new_hist = merge_bins(old_hist, args.bin, args.direction)
            hists[f"{dc.dirname}/{k}"] = new_hist
    with uproot.recreate(shapes_out) as fout:
        for key, new_hist in hists.items():
            fout[key] = new_hist

    print(f"New datacard written to: {out_datacard}")
    print(f"New shapes file written to: {shapes_out}")

if __name__ == "__main__":
    main()
