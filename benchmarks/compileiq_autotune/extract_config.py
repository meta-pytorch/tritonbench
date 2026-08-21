#!/usr/bin/env python
"""
Extract the best Advanced Control Files (ACFs) out of a CompileIQ search.

CompileIQ records every evaluation in a `SearchResult` dataframe (also dumped to
CSV via `Search.dump_results`). For a file-backed search space (e.g. the PTXAS
search space), the `params` column of each row is the hex-encoded ACF that was
handed to the objective function, so extracting the best configs is just a
matter of sorting by score and writing those blobs back out as binary files.
"""

import argparse
import os
import sys
from typing import List, Optional

from compileiq.results import SearchResult
from compileiq.utils.helpers import save_compiler_config


def extract_best_configs(
    results: SearchResult | str,
    output_dir: str,
    best_number: int = 3,
    file_prefix: str = "best",
    higher_is_better: bool = True,
) -> List[str]:
    """Write the top `best_number` ACFs of a search to `output_dir`.

    Args:
        results: A `SearchResult`, or a path to a CompileIQ results CSV.
        output_dir: Directory the ACF files are written to.
        best_number: How many configs to extract.
        file_prefix: Prefix of the generated file names.
        higher_is_better: True for maximization searches (e.g. tflops),
            False for minimization searches (e.g. latency).

    Returns:
        The paths of the written ACF files, best first.
    """
    problem_type = "max" if higher_is_better else "min"
    if isinstance(results, str):
        results = SearchResult.from_csv(results, problem_type=problem_type)

    df = results.df_results
    score_column = results.score_columns[0]
    scores = df[score_column].apply(_to_float)
    ranked = df.assign(**{"_score": scores})
    ranked = ranked[ranked["_score"].notna()]
    ranked = ranked.sort_values(by="_score", ascending=not higher_is_better)

    os.makedirs(output_dir, exist_ok=True)
    config_files = []
    for rank, (_, row) in enumerate(ranked.head(best_number).iterrows()):
        params = row["params"]
        if not isinstance(params, str):
            # Not a file-backed search space: nothing to write out as an ACF.
            continue
        config_file = os.path.join(
            output_dir, f"{file_prefix}-{rank + 1}-{row['_score']}.acf"
        )
        save_compiler_config(config_file, params)
        config_files.append(config_file)
    return config_files


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        # CompileIQ marks failed evaluations with INVALID_SCORE ("*").
        return None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract the best ACFs from a CompileIQ results CSV."
    )
    parser.add_argument("results_csv", help="Path to the CompileIQ results CSV.")
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write the ACF files to."
    )
    parser.add_argument(
        "--best-number", type=int, default=3, help="Number of configs to extract."
    )
    parser.add_argument("--file-prefix", default="best", help="Output file prefix.")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Set for minimization searches (e.g. latency).",
    )
    return parser


def main(args: Optional[List[str]] = None):
    args = get_parser().parse_args(args)
    config_files = extract_best_configs(
        results=args.results_csv,
        output_dir=args.output_dir,
        best_number=args.best_number,
        file_prefix=args.file_prefix,
        higher_is_better=not args.lower_is_better,
    )
    for config_file in config_files:
        print(config_file)


if __name__ == "__main__":
    main(sys.argv[1:])
