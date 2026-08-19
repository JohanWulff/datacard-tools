# coding: utf-8

import re

import numpy as np
from datacard_parser import Datacard

import argparse
import pandas as pd
import os

MERGE_PROCESSES = {"VV": ["WW*", "WZ*", "ZZ*"],
                   "SM H": ["ggH*", "qqH*", "WH*", "ZH*"],
                   "ttH": ["ttH*",]}

def match_process(pattern: str, process: str) -> bool:
    """
    Checks if a process matches a pattern. The pattern can contain wildcards (*).
    """
    from fnmatch import fnmatch
    return fnmatch(process, pattern)

def make_parser():
    parser = argparse.ArgumentParser(
        description="Check that the nominal yields in the datacard match the yields in the shapes file.")
    parser.add_argument("datacard", type=str,
                        nargs="+", help="Path to the datacard file(s).")
    parser.add_argument("--export", type=str, default=None,
                        help="Export the yield table to a file (CSV or LaTeX).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the output file. If not specified, the current directory is used.")
    return parser

def collect_yields(datacard: Datacard, 
                   merge_processes: dict[str, list[str]]) -> dict[str, float]:
    """
    Collects the nominal yields from the datacard and returns them as a dictionary.
    """
    nominal_yields = datacard.nominal_yields

    # check if all processes to merge are present in the datacard
    for merged_process, processes_to_merge in merge_processes.items():
        for process in processes_to_merge:
            matches = [p for p in nominal_yields if match_process(process, p)]
            if not matches:
                raise ValueError(f"Process {process} to merge into {merged_process} not found in datacard.")
            else:
                for match in matches:
                    nominal_yields[merged_process] = nominal_yields.get(merged_process, 0) + nominal_yields[match]
                    del nominal_yields[match]
    
    # format 
    nominal_yields = {re.sub(r"_20\d+(?:APV)?", "", k): np.round(v, decimals=3) for k,v in nominal_yields.items()}
    nominal_yields = {k: v for k,v in sorted(nominal_yields.items(), key=lambda item: item[1], reverse=True)}

    return nominal_yields

#def signal_scan(datacards: list[str],) -> dict[str, dict[str, float]]:
#    """
#    """
#    # find the set of years in the datacards
#    if not all([years, spins, masses]): 
#        raise ValueError("Could not find years, spins, or masses in the datacard paths. Please check the datacard paths.")
#    for year in set(years):
#        signal_yields = {}
#        for spin in set(spins):
#            cards = [c for c in datacards if year in c and f"spin_{spin}" in c]
#            for card in cards:
#                datacard = Datacard(card)
#                datacard.cross_check_nominal_yields()
#                nominal_yields = collect_yields(datacard, merge_processes={})
#                signal_yields[(datacard.channel, datacard.category, datacard.mass)] = nominal_yields[datacard.signal_process]
#        signal_yield_df = pd.DataFrame.from_dict(signal_yields, orient="index", columns=["Yield"])
#        signal_yield_df.index.names = ["Channel", "Category", "Mass"]
#        signal_yield_df = signal_yield_df.sort_index()
#        print(f"Signal yields for year {year} and spin {spin}:")
#        print(signal_yield_df)
        

def export_yields_to_file(nominal_yields: dict[str, float], output_file: str):
    """
    Collects the nominal yields in a pandas DataFrame and exports them to either a 
    CSV or LaTeX file, depending on the file extension of the output_file.
    """

    df = pd.DataFrame(list(nominal_yields.items()), columns=["Process", "Yield"])
    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    elif output_file.endswith(".tex"):
        df.to_latex(output_file, index=False)
    else:
        raise ValueError("Output file must be either .csv or .tex")
    

def main():

    parser = make_parser()
    args = parser.parse_args()

    # background yields are the same irrespective of the spin and mass
    # if multiple spin and mass hypotheses are passed, print an info and 
    # filter out all but one hypothesis for the yield table
    if len(args.datacard) > 1:
        years = set(re.compile(r"_(2016APV|2016|2017|2018)_").findall(" ".join(args.datacard)))
        spins = set(re.compile(r"spin_(\d+)").findall(" ".join(args.datacard)))
        masses = set(re.compile(r"mass_(\d+)").findall(" ".join(args.datacard)))
        categories = set(re.compile(r"_(resolved1b|resolved2b|boosted)_").findall(" ".join(args.datacard)))
        channels = set(re.compile(r"_(etau|mutau|tautau)_").findall(" ".join(args.datacard)))
        if len(spins) > 1 or len(masses) > 1:
            print(f"Multiple hypotheses found in the datacards")
            print(f"Generating a signal yield table for all hypotheses, but only the first hypothesis will be used for the background yields.") 

            for year in years:
                signal_yields = {}
                for s in spins:
                    for channel in channels:
                        for cat in categories:
                            cards = [p for p in args.datacard if f"spin_{s}_" in p and f"_{year}_" in p and f"_{channel}_" in p and f"_{cat}_" in p]
                            for card_path in cards:
                                mass = int(re.compile(r"mass_(\d+)").search(card_path).group(1))
                                signal_yields[(mass, int(s), channel, cat)] = Datacard(card_path).get_signal_yield()

                signal_yield_df = pd.Series(signal_yields).unstack([1, 2, 3])
                signal_yield_df.index.name = "Mass"
                signal_yield_df.columns.names = ["Spin", "Channel", "Category"]
                signal_yield_df = signal_yield_df.sort_index().sort_index(axis=1)
                # export the signal yield table to a tex file
                if args.export and args.export.endswith("tex"):
                    if args.output_dir: 
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        output_file = os.path.join(args.output_dir, f"signal_yields_{year}.tex")
                    else:
                        output_file = f"signal_yields_{year}.tex"
                    signal_yield_df.to_latex(output_file, index=True)
                    print(f"Signal yield table exported to {output_file}")
            

            spin, mass = spins.pop(), masses.pop()
            background_cards = [p for p in args.datacard if f"spin_{spin}_mass_{mass}" in p]
    else:
        background_cards = args.datacard

    if len(channels) > 1 or len(categories) > 1:
        # Collect process yields for every year, channel, and category.
        background_yields = {}
        for datacard_path in background_cards: 
            if not os.path.isfile(datacard_path):
                raise FileNotFoundError(f"Datacard file {datacard_path} does not exist.")
            datacard = Datacard(datacard_path)
            datacard.cross_check_nominal_yields()
            nominal_yields = collect_yields(datacard, merge_processes=MERGE_PROCESSES)
            background_yields[(datacard.year, datacard.channel, datacard.category)] = nominal_yields
        if args.export and args.export.endswith("tex"):
            for year in sorted({year for year, _, _ in background_yields}):
                year_yields = {
                    (channel, category): yields
                    for (entry_year, channel, category), yields in background_yields.items()
                    if entry_year == year
                }
                background_yield_df = pd.DataFrame(year_yields)
                background_yield_df.index.name = "Process"
                background_yield_df.columns.names = ["Channel", "Category"]
                background_yield_df = background_yield_df.sort_index(axis=1)
                total_column = ("Total", "")
                background_yield_df[total_column] = background_yield_df.sum(axis=1)
                background_yield_df = background_yield_df.sort_values(by=total_column, ascending=False)

                if args.output_dir:
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    output_file = os.path.join(args.output_dir, f"background_yields_{year}.{args.export}")
                else:
                    output_file = f"background_yields_{year}.{args.export}"
                background_yield_df.to_latex(output_file, index=True)
                print(f"Yield table exported to {output_file}")


    else:
        datacard_path = background_cards[0]
        if not os.path.isfile(datacard_path):
            raise FileNotFoundError(f"Datacard file {datacard_path} does not exist.")
        datacard = Datacard(datacard_path)
        print(f'Yield table for datacard: \n'
            f'{datacard.year} {datacard.channel} {datacard.category}\n'
            f'Signal: spin {datacard.spin}, mass {datacard.mass} \n')
        datacard.cross_check_nominal_yields()
        nominal_yields = collect_yields(datacard, merge_processes=MERGE_PROCESSES)
        for process, yield_value in nominal_yields.items():
            print(f" {process},  {yield_value}")
            print(f" -------------------------------")

        print("           ")

        if args.export:
            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                output_file = os.path.join(args.output_dir, f"{datacard.datacard.stem}.{args.export}")
            else:
                output_file = f"{datacard.datacard.stem}.{args.export}"
            export_yields_to_file(nominal_yields, output_file)
            print(f"Yield table exported to {output_file}")

if __name__ == "__main__":
    main()