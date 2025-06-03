import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_datacard(input_path, output_folder, only_shape=False):
    with open(input_path) as f:
        lines = f.readlines()

    # Match lines starting with nuisance name, then whitespace, then lnN or shape
    nuisance_line_regex = re.compile(r"^[A-Za-z0-9_]+(\s+)(lnN|shape)\s")

    new_lines = []
    for line in lines:
        match = nuisance_line_regex.match(line)
        if match:
            nuisance_type = match.group(2)
            # If only_shape is set, skip lnN lines
            if only_shape and nuisance_type != "shape":
                new_lines.append(line)
                continue
            # Split into columns and spaces
            parts = re.split(r'(\s+)', line.rstrip('\n'))
            # Find indices of actual entries (skip spaces)
            entry_indices = [i for i in range(0, len(parts), 2)]
            # Replace all entries after the first two with '-'
            for idx in entry_indices[2:]:
                parts[idx] = '-'
            # Reconstruct the line
            new_line = ''.join(parts) + '\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    output_path = Path(output_folder) / Path(input_path).name
    with open(output_path, "w") as f:
        f.writelines(new_lines)
    print(f"Wrote cleaned datacard to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace all nuisance values in datacard(s) with '-'. Optionally, only remove shape nuisances.")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input datacard file(s)")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder for cleaned datacards")
    parser.add_argument("--parallel", action="store_true", help="Process datacards in parallel")
    parser.add_argument("--only-shape", action="store_true", help="Only remove shape nuisances")
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_datacard, input_datacard, output_folder, args.only_shape)
                for input_datacard in args.input
            ]
            for future in futures:
                future.result()
    else:
        for input_datacard in args.input:
            process_datacard(input_datacard, output_folder, args.only_shape)