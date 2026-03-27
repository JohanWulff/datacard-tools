from datacard_parser import Datacard
from datacard_updates import conservative_update, loose_update, smoothen_large_shape_effects

from pathlib import Path
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
    update_mode: str,
    output_path: str,
    validation_results_dir: str,
    only_remove: bool,
    threshold: float,
):
    datacard = Datacard(datacard=Path(datacard_path), ignore_processes=ignore_processes)
    if update_mode == "conservative":
        result = conservative_update(
            datacard,
            Path(output_path) / "conservative",
            validation_results_dir,
            only_remove=only_remove,
        )
    elif update_mode == "loose":
        result = loose_update(datacard, Path(output_path) / "loose", threshold=threshold)
    elif update_mode == "smoothen":
        changed = smoothen_large_shape_effects(datacard, Path(output_path) / "smoothen")
        result = {
            "updated_shape_keys": sorted(changed.keys()),
            "n_updated_shape_keys": len(changed),
        }
    else:
        raise ValueError(f"Unsupported update mode: {update_mode}")

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
        help="Directory to store validation results.",
    )
    parser.add_argument("--ignore-processes", nargs="*", default=["data_obs", "QCD"], help="Processes to ignore in the datacard.")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=4,
        help="Maximum number of worker processes.",
    )
    parser.add_argument("--only-remove", action="store_true", help="Only remove nuisances, do not convert shape to lnN.")
    parser.add_argument(
        "--update-mode",
        choices=["conservative", "loose", "smoothen"],
        default="conservative",
        help="Choose update mode: 'conservative' (default), 'loose' or 'smoothen'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for non-genuine shape detection (only used in loose mode).",
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
    update_stats = {}
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
                returned_datacard_path, result = future.result()
                update_stats[returned_datacard_path] = result
            except Exception as e:
                print(f"[ERROR] Error while processing datacard {datacard_path}: {e}")
                continue

    output_json = args.updated_shifts_json
    if output_json is None:
        output_json = str(Path(args.output_path) / f"updated_shifts_{args.update_mode}.json")

    with open(output_json, "w") as f:
        json.dump(
            {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "update_mode": args.update_mode,
                    "threshold": args.threshold,
                    "only_remove": args.only_remove,
                    "n_datacards": len(update_stats),
                },
                "updated_shifts": update_stats,
            },
            f,
            indent=2,
        )
    print(f"Wrote updated shifts JSON to: {output_json}")


if __name__ == "__main__":
    main()
