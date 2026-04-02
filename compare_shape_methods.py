"""
Script to compare two methods for testing smallShapeEffects:
1. ValidateDatacards.py (subprocess method) 
2. Datacard.get_shape_vars() (in-class method)

This script runs both methods and compares the numerical shape variation values.
"""

import json
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Union
from tqdm import tqdm


def _json_default_serializer(obj):
    """Convert numpy scalar objects (e.g. bool_, float64) to native Python types."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ShapeMethodComparison:
    """Compare two smallShapeEffects testing methods"""
    
    DEFAULT_THRESHOLD = 0.001  # ValidateDatacards.py default
    TOLERANCE = 1e-6
    
    def __init__(self, validation_results_dir: Path = None):
        """
        Initialize the comparison object.
        
        Args:
            validation_results_dir: Directory to store ValidateDatacards.py output.
                                   Defaults to /tmp/jwulff/inference/validation_results
        """
        if validation_results_dir is None:
            validation_results_dir = Path("/tmp/jwulff/inference/validation_results")
        self.validation_results_dir = Path(validation_results_dir)
        self.validation_results_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_datacard(self, datacard_path: Path) -> Dict:
        """
        Compare both methods for a single datacard.
        
        Args:
            datacard_path: Path to the datacard file
            
        Returns:
            Dictionary with comparison results for all nuisances
        """
        import uproot
        from datacard_parser import Datacard

        datacard = Datacard(datacard_path)
        results = {
            "datacard": str(datacard_path),
            "year": datacard.year,
            "channel": datacard.channel,
            "category": datacard.category,
            "spin": datacard.spin,
            "mass": datacard.mass,
            "nuisances": {}
        }
        
        # Run ValidateDatacards.py
        validation_path = datacard.validate(self.validation_results_dir)
        if not validation_path:
            results["error"] = "ValidateDatacards.py failed"
            return results
        
        # Parse ValidateDatacards.py output
        with open(validation_path, 'r') as f:
            validation_output = json.load(f)
        
        # Mirror conservative_update() parsing logic for ValidateDatacards output:
        # smallShapeEff is structured as nuisance -> category -> process.
        if "smallShapeEff" in validation_output and validation_output["smallShapeEff"]:
            small_shape_eff = validation_output["smallShapeEff"]
            first_nuisance = next(iter(small_shape_eff))
            cat_name = next(iter(small_shape_eff[first_nuisance]))
            small_shape_eff = {
                nuisance: small_shape_eff[nuisance].get(cat_name, {})
                for nuisance in small_shape_eff
            }
        else:
            small_shape_eff = {}
            results["info"] = "No small shape effects found by ValidateDatacards.py"
        
        # Use the default threshold from ValidateDatacards.py
        threshold = self.DEFAULT_THRESHOLD
        
        # Compare methods for each shape nuisance in the datacard
        with uproot.open(datacard.shapes_file) as shapes_file_handle:
            for nuisance in datacard.shape_nuisances:
                processes_data = small_shape_eff.get(nuisance, {})
                nuisance_result = {
                    "threshold": threshold,
                    "processes": {}
                }
                
                # Get the list of processes with small shape effects from get_shape_vars()
                try:
                    small_processes_from_method2 = datacard.get_shape_vars(
                        nuisance, threshold, shapes_file_handle
                    )
                except Exception as e:
                    nuisance_result["error"] = f"get_shape_vars() failed: {str(e)}"
                    results["nuisances"][nuisance] = nuisance_result
                    continue
                
                # Compare all processes flagged by either method for this nuisance
                processes_to_compare = sorted(set(processes_data.keys()) | set(small_processes_from_method2))

                # For each process, get the actual shape var values from both methods
                for process in processes_to_compare:
                    json_vals = processes_data.get(process)
                    comparison = {
                        "method1_validateDatacards": (
                            {
                                "diff_u": json_vals["diff_u"],
                                "diff_d": json_vals["diff_d"],
                            }
                            if json_vals is not None
                            else None
                        ),
                        "is_in_method1_results": json_vals is not None,
                        "is_in_method2_results": process in small_processes_from_method2,
                    }
                    
                    # Get values from method 2 (_get_sum_shape_var)
                    try:
                        up_var, down_var = datacard._get_sum_shape_var(
                            nuisance, process, shapes_file_handle
                        )
                        comparison["method2_get_shape_vars"] = {
                            "up_var": up_var,
                            "down_var": down_var
                        }
                        
                        if json_vals is not None:
                            # Check if values match when method1 has this process
                            up_match = abs(json_vals["diff_u"] - up_var) < self.TOLERANCE
                            down_match = abs(json_vals["diff_d"] - down_var) < self.TOLERANCE
                            comparison["match"] = up_match and down_match

                            if not comparison["match"]:
                                comparison["diff_u_error"] = json_vals["diff_u"] - up_var
                                comparison["diff_d_error"] = json_vals["diff_d"] - down_var
                        else:
                            # Process flagged only by method2 for this nuisance
                            comparison["match"] = False
                        
                    except Exception as e:
                        comparison["method2_error"] = str(e)
                        comparison["match"] = False
                    
                    nuisance_result["processes"][process] = comparison
                
                results["nuisances"][nuisance] = nuisance_result
        
        return results
    
    def compare_datacards_batch(self, directory: Path, max_processes: int = 1) -> List[Dict]:
        """
        Compare both methods for all datacards in a directory.
        
        Args:
            directory: Directory containing datacard files
            max_processes: Maximum number of worker processes for parallel batch mode
            
        Returns:
            List of comparison results for each datacard
        """
        directory = Path(directory)
        datacards = sorted(directory.glob("datacard_*.txt"))
        
        if not datacards:
            raise ValueError(f"No datacard files found in {directory}")
        
        if max_processes <= 1:
            results = []
            for i, datacard_path in enumerate(
                tqdm(datacards, desc="Processing datacards", total=len(datacards)),
                1,
            ):
                result = self.compare_datacard(datacard_path)
                results.append(result)
            return results

        print(
            f"Running batch comparison with {max_processes} processes "
            f"for {len(datacards)} datacards"
        )

        results_by_path = {}
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            future_to_path = {
                executor.submit(
                    compare_datacard_wrapper,
                    str(datacard_path),
                    str(self.validation_results_dir),
                ): str(datacard_path)
                for datacard_path in datacards
            }
            for future in tqdm(
                as_completed(future_to_path),
                total=len(future_to_path),
                desc="Processing datacards",
            ):
                datacard_path = future_to_path[future]
                try:
                    returned_datacard_path, result = future.result()
                    results_by_path[returned_datacard_path] = result
                except Exception as e:
                    results_by_path[datacard_path] = {
                        "datacard": datacard_path,
                        "error": f"Batch worker failed: {e}",
                    }
                    print(f"[ERROR] Failed: {Path(datacard_path).name}: {e}")

        results = [results_by_path[str(path)] for path in datacards]
        
        return results
    
    def print_console_report(self, results: Union[List[Dict], Dict]):
        """
        Print a human-readable comparison report to console.
        
        Args:
            results: Single comparison result or list of results
        """
        if isinstance(results, dict):
            results = [results]
        
        print("\n" + "="*100)
        print("SHAPE EFFECTS COMPARISON REPORT: ValidateDatacards.py vs get_shape_vars()")
        print("="*100)
        
        total_matches = 0
        total_mismatches = 0
        total_errors = 0
        
        for result in results:
            print(f"\n{'-'*100}")
            print(f"Datacard: {Path(result['datacard']).name}")
            print(f"Metadata: {result['year']} | {result['channel']} | {result['category']} | spin {result['spin']} | mass {result['mass']}")
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
                total_errors += 1
                continue
            
            if "info" in result:
                print(f"INFO: {result['info']}")
                continue
            
            if not result.get("nuisances"):
                print("No nuisances to compare")
                continue
            
            for nuisance, nuisance_result in result["nuisances"].items():
                print(f"\n  Nuisance: {nuisance}  (threshold: {nuisance_result['threshold']:.6f})")
                
                if "error" in nuisance_result:
                    print(f"  ERROR: {nuisance_result['error']}")
                    total_errors += 1
                    continue
                
                processes = nuisance_result["processes"]
                
                for process, comparison in processes.items():
                    if "method2_error" in comparison:
                        print(f"    {process:30s} | ERROR: {comparison['method2_error']}")
                        total_errors += 1
                        continue
                    
                    m1_vals = comparison["method1_validateDatacards"]
                    m2_vals = comparison.get("method2_get_shape_vars", {})
                    is_match = comparison["match"]
                    in_method2 = comparison["is_in_method2_results"]
                    in_method1 = comparison.get("is_in_method1_results", False)
                    
                    status = "✓ MATCH" if is_match else "✗ DIFF"
                    method2_status = "flagged" if in_method2 else "not flagged"
                    
                    print(f"    {process:30s} | {status:15s} | method2: {method2_status}")
                    if m1_vals is not None:
                        print(f"      Method1 (ValidateDatacards): diff_u={m1_vals['diff_u']:.8f}, diff_d={m1_vals['diff_d']:.8f}")
                    else:
                        print(f"      Method1 (ValidateDatacards): not flagged")
                    
                    if m2_vals:
                        print(f"      Method2 (get_shape_vars):    up_var ={m2_vals['up_var']:.8f}, down_var={m2_vals['down_var']:.8f}")

                    if not in_method1 and in_method2:
                        print("      Flag mismatch: present only in method2 results")

                    if not is_match:
                        if m1_vals is not None and m2_vals:
                            errors = comparison.get("diff_u_error", 0), comparison.get("diff_d_error", 0)
                            print(f"      Error: Δdiff_u={errors[0]:.2e}, Δdiff_d={errors[1]:.2e}")
                        total_mismatches += 1
                    else:
                        total_matches += 1
        
        print(f"\n{'='*100}")
        print(f"SUMMARY: {total_matches} matches | {total_mismatches} mismatches | {total_errors} errors")
        print("="*100 + "\n")
    
    def export_json_report(self, results: Union[List[Dict], Dict], output_path: Path = None) -> Path:
        """
        Export comparison results as JSON report.
        
        Args:
            results: Single comparison result or list of results
            output_path: Path for output file. If None, generated with timestamp.
            
        Returns:
            Path to the created JSON file
        """
        if isinstance(results, dict):
            results = [results]
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"comparison_report_{timestamp}.json")
        else:
            output_path = Path(output_path)
        
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "tolerance": self.TOLERANCE,
            "default_threshold": self.DEFAULT_THRESHOLD,
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=_json_default_serializer)
        
        print(f"JSON report exported to: {output_path}")
        return output_path


def main():
    """Main entry point for the comparison script"""
    parser = argparse.ArgumentParser(
        description=(
            "Compare smallShapeEffects results from ValidateDatacards.py "
            "with Datacard.get_shape_vars()."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a datacard file or a directory containing datacard_*.txt files.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("/tmp/jwulff/inference/validation_results"),
        help="Directory for ValidateDatacards.py JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for JSON report output. Default: comparison_report_<timestamp>.json",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=1,
        help="Maximum number of worker processes for directory mode.",
    )

    args = parser.parse_args()
    input_path = args.input_path
    validation_dir = args.validation_dir
    output_file = args.output
    
    # Initialize comparison object
    comparator = ShapeMethodComparison(validation_dir)
    
    # Determine if input is file or directory
    if input_path.is_file():
        print(f"Comparing single datacard: {input_path}")
        result = comparator.compare_datacard(input_path)
        comparator.print_console_report(result)
        comparator.export_json_report(result, output_file)
    
    elif input_path.is_dir():
        print(f"Comparing all datacards in: {input_path}")
        results = comparator.compare_datacards_batch(input_path, max_processes=args.max_processes)
        comparator.print_console_report(results)
        comparator.export_json_report(results, output_file)
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


def compare_datacard_wrapper(datacard_path: str, validation_results_dir: str) -> Tuple[str, Dict]:
    """Top-level wrapper for parallel worker execution."""
    comparator = ShapeMethodComparison(Path(validation_results_dir))
    result = comparator.compare_datacard(Path(datacard_path))
    return datacard_path, result


if __name__ == "__main__":
    main()
