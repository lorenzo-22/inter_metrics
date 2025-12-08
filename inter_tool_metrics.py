from pathlib import Path
import pandas as pd
import sys
import argparse
from scipy.stats import spearmanr
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import numpy as np
import os, re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product



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

        # Make it find any/ corrrect proteincolumn name
        protein_column_names = ['ID', 'Protein', 'protein', 'id', 'prot', 'gene', 'Gene']
        df1_id = (col for col in df1.columns if col in protein_column_names)
        df2_id = (col for col in df2.columns if col in protein_column_names)

        # Rank correlation
        rho, pval = spearmanr(df1['rank'], df2['rank'])

        # Concordance: number of shared proteins in top
        top_n = 50
        top1 = set(df1.sort_values('rank').head(top_n)[df1_id])
        top2 = set(df2.sort_values('rank').head(top_n)[df2_id])

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



def plot_method_heatmap(df, value_col='rank_cor_rho', title=None, cmap='coolwarm'):
    """
    Plots a heatmap of method1 vs method2 colored by value_col.
    
    Parameters:
        df: pandas DataFrame with columns ['method1', 'method2', value_col]
        value_col: column name to color by
        title: optional plot title
        cmap: colormap for heatmap
    """
    pivot_df = df.pivot(index='method1', columns='method2', values=value_col)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'label': 'Rank correlation'}, vmin=0, vmax=1)
    plt.title(title if title else 'Heatmap of rank correlations')
    plt.xlabel('Tool 1')
    plt.ylabel('Tool 2')
    plt.tight_layout()

    return plt


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
    
    # If first omnibench run, y
    if len(all_files) < 2:
        # write minimal empty CSV so Snakemake is happy
        pd.DataFrame().to_csv(os.path.join(args.output_dir, 'concordance_scores.csv'), index=False)
        print("Only one tool found. Skipping pairwise metrics.")
        sys.exit(0)

    if not all_files:
        print("ERROR: No '_results.csv' files found!")
        sys.exit(1)

    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f)

    # extract dataset names
    datasets = list(set(
        re.search(r'/out/data/([^/]+)/', str(f)).group(1)
        for f in all_files
        if re.search(r'/out/data/([^/]+)/', str(f))
    ))

    # map dataset -> list of files
    results_files = defaultdict(list)

    for f in all_files:
        match = re.search(r'/out/data/([^/]+)/', str(f))
        if match:
            dataset = match.group(1)
            results_files[dataset].append(f)

    results_files = dict(results_files)  # if you don't want defaultdict
    print(results_files)

    concordance_scores = []

    for dataset in datasets:
        print(f'Now running {dataset}')
        files = results_files[dataset]

        # Make unique pairs:
        indices = list(range(len(files)))
        pairs = list(product(indices, indices))

        # Load files and calculate FDR-adjusted p-value and rank
        all_results = load_and_process_files(files)
        file_names = [re.search(r'/methods/([^/]+)/default/', str(f)).group(1) for f in files]
        
        # Calculate metrics
        concordance_df = calculate_metrics(all_results, file_names=file_names, pairs=pairs)

        concordance_scores.append(concordance_df)
    
    print(concordance_scores)

    for idx in range(len(datasets)):
        dataset = datasets[idx]
        plot = plot_method_heatmap(concordance_scores[idx], title=f'Heatmap of rank correlations, dataset: {dataset}')
        plot.savefig(os.path.join(output_dir, f'heatmap_rankcorrelations_{dataset}.png'), dpi = 300)


if __name__ == "__main__":
    main()