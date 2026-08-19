# coding: utf-8

from pathlib import Path
import numpy as np
import uproot
from hist import Hist
import hist
import matplotlib.pyplot as plt
import mplhep as hep

from datacard_parser import Datacard


def plot_variation(
    datacard: Datacard,
    nuisance: str,
    process: str,
    output_dir: Path,
    y_log: bool = True,
    binning: str = "numbers",
) -> None:

    def equal_width_hist(h: Hist) -> Hist:
        if np.all(h.axes[0].widths == h.axes[0].widths[0]):
            return h
        else:
            new_hist = Hist(hist.axis.Regular(h.axes[0].size,
                                    h.axes[0].edges[0],
                                    h.axes[0].edges[-1], name=h.axes[0].name), storage=h.storage_type)
            new_hist.view()[:] = h.view()
            return new_hist

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

    # add a guard to avoid division by zero in case the nominal histogram is empty
    if nominal_hist.sum().value == 0:
        print(f"Warning: nominal histogram for process {process} is empty in datacard {datacard.datacard.name}")
        return False

    change_up, change_down = hist_up.sum().value/nominal_hist.sum().value, hist_down.sum().value/nominal_hist.sum().value

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    label_factor_up = f"{change_up:.1f}" if np.abs(change_up - 1) > 0.01 else "1"
    label_factor_down = f"{change_down:.1f}" if np.abs(change_down - 1) > 0.01 else "1"
    # we don't care about the correct bin edges but just number bins from 0 to n_bins, so we can use the hist.values() method to get the bin contents
    hep.histplot(nominal_hist.values(), histtype="step", label=f"Nominal, Yield {nominal_hist.sum().value:.1e}", ax=ax, color="black")
    hep.histplot(hist_up.values(), histtype="step", label=f"{nuisance.replace('_', '-')} Up, x{label_factor_up}", ax=ax, color="C0")
    hep.histplot(hist_down.values(), histtype="step", label=f"{nuisance.replace('_', '-')} Down, x{label_factor_down}", ax=ax, color="C1")
    ax.set_title(f"{process.replace('_', '-')} - {nuisance.replace('_', '-')}")
    ax.legend()
    ax.set_ylabel("Events")
    ax.set_yscale("log" if y_log else "linear")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    yccsm = "_".join([datacard.year, datacard.channel, datacard.category, str(datacard.spin), str(datacard.mass)])
    plt.savefig(output_dir / f"{yccsm}__{process}_{nuisance}.pdf")
    plt.close() 

    
def main(datacard_path: Path,
         output_dir: Path,
         process: str = None,
         nuisances: list[str] = None,
         y_log: bool = True,
         binning: str = "numbers") -> None:
    datacard = Datacard(datacard_path)

    if process and nuisances:
        print(f"Plotting shape variations for datacard {datacard.datacard.name} in directory {datacard.dirname}")
        print(f"Process: {process}")
        for nuisance in nuisances:
            print(f"Nuisance: {nuisance}")
            plot_variation(datacard, nuisance, process, output_dir, y_log, binning)
         


    for process in processes:
        for nuisance in nuisances:
            plot_variation(datacard, nuisance, process, output_dir, y_log, binning)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot shape variations for a datacard")
    parser.add_argument("datacard_path", type=Path, help="Path to the datacard file")
    parser.add_argument("output_dir", type=Path, help="Directory to save the output plots")
    parser.add_argument("--process", type=str, help="Process to plot (default: all processes)")
    parser.add_argument("--nuisances", type=str, nargs="+", help="Nuisances to plot (default: all nuisances)")
    parser.add_argument("--y-log", action="store_true", help="Use logarithmic scale for y-axis")
    parser.add_argument("--binning", type=str, choices=["numbers", "edges"], default="numbers", help="Binning method for histograms (default: numbers)")

    args = parser.parse_args()
    main(args.datacard_path, args.output_dir, args.process, args.nuisances, args.y_log, args.binning)   