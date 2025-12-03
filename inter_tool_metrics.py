from pathlib import Path
import pandas as pd
import sys
import argparse
from scipy.stats import spearmanr
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import numpy as np
import os, re

def load_and_process_files(all_files):
    all_results = []

    # FDR adjust and make ranks:
    for f in all_files:
        # Load file
        result = pd.read_csv(f)

        # Workaround for non-equal 
        if 'P.Value' in result.columns:
            result = result.rename({'P.Value': 'p_value'}, axis = 1)
        if 'Unnamed: 0' in result.columns:
            result = result.set_index('Unnamed: 0')

        # FDR-adjust p-values
        _, p_adjusted, _, _ = multipletests(result['p_value'], method='fdr_bh')
        result['p_adjusted'] = p_adjusted

        # Make importance score
        result['importance'] = np.abs(result['effect_size']) * -np.log10(result['p_adjusted'] + 1e-10)

        # Rank
        result['rank'] = result['importance'].rank(method='min', ascending=False)
        all_results.append(result)

    return all_results

def calculate_metrics(all_results, file_names, pairs, top_n = 50):

    rank_correlation = []

    for i, j in pairs:
        df1 = all_results[i].sort_values('rank')
        df2 = all_results[j].sort_values('rank')
        
        # Rank correlation
        rho, pval = spearmanr(df1['rank'], df2['rank'])

        # Concordance: number of shared proteins in top
        top_n = 50
        top1 = set(df1.sort_values('rank').head(top_n)['ID'])
        top2 = set(df2.sort_values('rank').head(top_n)['ID'])

        c_score = len(top1 & top2) / top_n

        
        rank_correlation.append({
            'method1': file_names[i],
            'method2': file_names[j],
            f'c_score_top{top_n}': c_score, 
            'rank_cor_rho': rho,
            'rank_cor_pval': pval
            })
        
    concordance_df = pd.DataFrame(rank_correlation)

    return concordance_df


def main():
    parser = argparse.ArgumentParser(description='Welch t-test benchmark runner')

    # parser.add_argument('--results', type=str,
    #                     help='path to all results', required = True)
    parser.add_argument('--output_dir', type=str,
                        help='output directory to store results of metrics.')
    
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    results_dir = Path(os.path.join(os.getcwd(), 'out'))
    output_dir = Path(args.output_dir)

    print(f"Input directory:  {results_dir}")
    print(f"Output directory: {output_dir}")

    # gather _results.csv files
    all_files = sorted(results_dir.rglob("*_results.csv"))

    if not all_files:
        print("ERROR: No '_results.csv' files found!")
        sys.exit(1)

    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f)

    # save file names
    file_names = [re.search(r'/methods/([^/]+)/default/', str(f)).group(1) for f in all_files]
    print(file_names)

    # Make unique pairs of the runs:
    indices = list(range(len(all_files)))
    pairs = list(combinations(indices, 2))
    print(pairs)

    # Load files and calculate FDR-adjusted p-value and rank
    all_results = load_and_process_files(all_files)
    
    # Calculate metrics
    concordance_df = calculate_metrics(all_results, file_names=file_names, pairs=pairs)

    print(concordance_df)

    concordance_df.to_csv(os.path.join(args.output_dir, 'concordance_scores.csv'))

if __name__ == "__main__":
    main()