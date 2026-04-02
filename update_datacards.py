from datacard_parser import Datacard
from datacard_updates import conservative_update

from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import argparse
import json
import re
import os
from datetime import datetime


def process_datacard_wrapper(
    datacard_path: str,
    ignore_processes: list[str],
    output_path: str,
    validation_results_dir: str,
    check_uncert_over: float,
    plot_output_dir: Optional[str],
):
    datacard = Datacard(datacard=Path(datacard_path), ignore_processes=ignore_processes)
    result = conservative_update(
        datacard,
        Path(output_path) / "conservative",
        validation_results_dir=validation_results_dir,
        check_uncert_over=check_uncert_over,
        plot_output_dir=plot_output_dir,
    )

    return datacard_path, result


def main():
    parser = argparse.ArgumentParser(description="Validate and update a datacard with non-genuine shape nuisances.")
    parser.add_argument(
        "--mass",
        type=int,
        nargs="+",
        required=False,
        choices=(MASSES := [ 250, 260, 270, 280, 300, 320, 350,
                             400, 450, 500, 550, 600, 650, 700, 750,
                             800, 850, 900, 1000, 1250, 1500, 1750, 2000,
                             2500, 3000]),
        default=MASSES,
        help=f"Mass of the hypothetical particle in GeV. Default: {MASSES}.",
    )
    parser.add_argument(
        "--datacard-path",
        "-d",
        type=str,
        default=(
            dc_path := (
                "/data/dust/user/kramerto/taunn_data/store/WriteDatacards/"
                "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite"
                "-default_extended_pair_ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096"
                "_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_fi80_lbn_ft_lt20_lr1_LBdefault_"
                "daurot_fatjet_composite_FIx5_SDx5/prod9/flats_systs10/final/symtest/"
            )
        ),
        help=f"Path to the datacards directory. Default: {dc_path}",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="/output/path/for/updated/datacards/",
        default="/data/dust/user/jwulff/inference/remodel_cards/",
    )
    parser.add_argument(
        "--validation-results-dir",
        type=str,
        default="/tmp/jwulff/inference/validation_results/",
        help="Directory to store ValidateDatacards.py JSON output.",
    )
    parser.add_argument("--ignore-processes", nargs="*", default=["data_obs", "QCD"], help="Processes to ignore in the datacard.")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=4,
        help="Maximum number of worker processes.",
    )
    parser.add_argument(
        "--check-uncert-over",
        type=float,
        default=2.0,
        help="Normalization cap factor passed to ValidateDatacards.py and datacard_update mode.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="If set, generate largeNormEff shape plots into this directory.",
    )
    parser.add_argument(
        "--updated-shifts-json",
        type=str,
        default=None,
        help="Optional output path for a JSON summary of updated shifts.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    datacard_paths = []
    missing_masses = []
    for mass in args.mass:
        paths = list(Path(args.datacard_path).glob(f"datacard_cat_*_spin_*_mass_{mass}.txt"))
        if not paths:
            missing_masses.append(mass)
            continue
        datacard_paths.extend(paths)

    if missing_masses:
        print(
            f"No datacards found for {len(missing_masses)} requested masses in {args.datacard_path}: "
            f"{missing_masses}"
        )

    campaigns = sorted({re.search(r"datacard_cat_([^_]+)_", path.name).group(1) for path in datacard_paths})
    spins = sorted({re.search(r"spin_(\d+)", path.name).group(1) for path in datacard_paths})
    matched_masses = sorted({int(re.search(r"mass_(\d+)", path.name).group(1)) for path in datacard_paths})
    print(
        f"Found {len(datacard_paths)} datacards for campaigns {campaigns}, "
        f"spins {spins} and masses {matched_masses} in {args.datacard_path}"
    )
    if args.plot_output_dir:
        print(f"LargeNormEff plotting: enabled, output directory: {args.plot_output_dir}")
    else:
        print("LargeNormEff plotting: disabled (set --plot-output-dir to enable)")
    update_stats = {}

    with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
        future_to_datacard = {
            executor.submit(
                process_datacard_wrapper,
                str(path),
                args.ignore_processes,
                args.output_path,
                args.validation_results_dir,
                args.check_uncert_over,
                args.plot_output_dir,
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
                returned_datacard_path, result = future.result()
                update_stats[returned_datacard_path] = result
            except Exception as e:
                print(f"[ERROR] Error while processing datacard {datacard_path}: {e}")
                continue

    #result = conservative_update(
    #    Datacard(datacard=Path(datacard_paths[0]), ignore_processes=args.ignore_processes),
    #    Path(args.output_path) / "conservative",
    #    validation_results_dir=args.validation_results_dir,
    #    check_uncert_over=args.check_uncert_over,
    #    plot_output_dir=args.plot_output_dir,
    #)

    output_json = args.updated_shifts_json
    if output_json is None:
        output_json = str(Path(args.output_path) / "updated_shifts_conservative.json")

    output_stats = update_stats

    with open(output_json, "w") as f:
        json.dump(
            {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "update_mode": "conservative",
                    "check_uncert_over": args.check_uncert_over,
                    "plot_output_dir": args.plot_output_dir,
                    "n_datacards_processed": len(update_stats),
                    "n_datacards_in_output": len(output_stats),
                },
                "updated_shifts": output_stats,
            },
            f,
            indent=2,
        )
    print(f"Wrote updated shifts JSON to: {output_json}")


if __name__ == "__main__":
    main()
