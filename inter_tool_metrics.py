from pathlib import Path
import pandas as pd
import sys
import argparse
from scipy.stats import spearmanr, kendalltau
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
        result = result.sort_values('importance')

        # Rank
        result['rank'] = result['importance'].rank(method='min', ascending=False)
        all_results.append(result)

    return all_results

def calculate_metrics(all_results, file_names, pairs, top_n = 50):

    rank_correlation = []

    for i, j in pairs:
        df1 = all_results[i]#.sort_values('rank')
        df2 = all_results[j]#.sort_values('rank')

        # Make it find any/ corrrect proteincolumn name
        protein_column_names = ['ID', 'Protein', 'protein', 'id', 'prot', 'gene', 'Gene']
        # Find the first column in df1 that matches
        df1_id = next((col for col in df1.columns if col in protein_column_names), None)
        df2_id = next((col for col in df2.columns if col in protein_column_names), None)

        if df1_id is None or df2_id is None:
            raise ValueError("No valid protein ID column found in one of the datasets")

        # Merge the two dataframes on protein ID
        merged = pd.merge(
            df1, 
            df2, 
            left_on=df1_id, right_on=df2_id,
            suffixes=('_A', '_B')
        )

        # Compute correlations on aligned importance values
        # Special case: if comparing method to itself, correlation is exactly 1.0
        if i == j:
            rho = 1.0
            pval = 0.0
            kendall_corr = 1.0
            kendall_pval = 0.0
        else:
            rho, pval = spearmanr(merged['importance_A'], merged['importance_B'], nan_policy='omit')
            kendall_corr, kendall_pval = kendalltau(merged['importance_A'], merged['importance_B'], nan_policy='omit')

        # Concordance: number of shared proteins in top
        top_n = 50
        top1 = set(df1.sort_values('rank').head(top_n)[df1_id])
        top2 = set(df2.sort_values('rank').head(top_n)[df2_id])

        c_score = len(top1 & top2) / top_n

        rank_correlation.append({
            'method1': file_names[i],
            'method2': file_names[j],
            f'c_score_top{top_n}': c_score, 
            'spearmanr_corr_rho': rho,
            'spearmanr_pval': pval, 
            'kendall_corr': kendall_corr, 
            'kendall_pval': kendall_pval
            })
        
    concordance_df = pd.DataFrame(rank_correlation)

    return concordance_df


def plot_method_heatmap(df, value_col='spearmanr_corr_rho', title=None, cmap='coolwarm'):
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
    # -----------------------------------------------------------------------
    # Load and parse
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Inter tool metrics')

    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--results', type=str, nargs='+', help='Results files', required=False)
    parser.add_argument('--input_dir', type=str, help='Input directory', required=False)
    parser.add_argument('--name', type=str, help='name', required=False)

    # Use parse_known_args to safely ignore any other args
    args, unknown = parser.parse_known_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(os.getcwd()) / 'out' / 'inter_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # -----------------------------------------------------------------------
    # Get results files - either from arguments or search
    # -----------------------------------------------------------------------
    if args.results:
        # Files passed as arguments (from OmniBenchmark metric_collector)
        all_files = [Path(f) for f in args.results]
        # Filter to only include cancer dataset
        all_files = [f for f in all_files if '/data/cancer/' in str(f)]
        all_files = sorted(all_files)
    else:
        # Search for files (for standalone execution)
        results_dir = Path(os.path.join(os.getcwd(), 'out'))
        print(f"Input directory:  {results_dir}")
        all_files = sorted(results_dir.rglob("*_results.csv"))
        # Filter to only include cancer dataset
        # all_files = [f for f in all_files if '/data/cancer/' in str(f)]
    
    # If first omnibench run, create empty outputs
    if len(all_files) < 2:
        # write minimal empty CSV so Snakemake is happy
        pd.DataFrame().to_csv(os.path.join(args.output_dir, 'concordance_scores.csv'), index=False)
        print("Only one tool found. Skipping pairwise metrics.")
        sys.exit(0)

    if not all_files:
        print("ERROR: No '_results.csv' files found!")
        sys.exit(1)

    # extract dataset names
    dataset_names = list(set(
        re.search(r'/data/([^/]+)/', str(f)).group(1)
        for f in all_files
        if re.search(r'/data/([^/]+)/', str(f))
    ))

    # Make dictionary of which files belong to which dataset
    results_files = defaultdict(list)

    for f in all_files:
        match = re.search(r'/data/([^/]+)/', str(f))
        if match:
            dataset = match.group(1)
            results_files[dataset].append(f)

    results_files = dict(results_files)

    concordance_scores = []

    # -----------------------------------------------------------------------------------
    # Load datasets and calculate metrics
    # -----------------------------------------------------------------------------------
    for dataset in dataset_names:
        files = results_files[dataset]

        # Make unique pairs:
        indices = list(range(len(files)))
        pairs = list(product(indices, indices))

        # Load files and calculate FDR-adjusted p-value and rank
        all_results = load_and_process_files(files)

        file_names = [re.search(r'/methods/([^/]+)/default/', str(f)).group(1) for f in files]
        
        # Calculate metrics
        print(f"Running dataset: {dataset}")
        concordance_df = calculate_metrics(all_results, file_names=file_names, pairs=pairs)

        concordance_scores.append(concordance_df)
        print(concordance_df)
    
    # Save concordance scores to CSV
    all_concordance = pd.concat(concordance_scores, ignore_index=True)
    concordance_path = os.path.join(output_dir, 'concordance_scores.csv')
    all_concordance.to_csv(concordance_path, index=False)
    print(f"Concordance scores saved to: {concordance_path}")
    
    # -----------------------------------------------------------------------------------
    # Make heatmaps and save in HTML
    # -----------------------------------------------------------------------------------

    plots = []
    plots_kendall = []

    for idx, dataset in enumerate(dataset_names):
        fig = plot_method_heatmap(
            concordance_scores[idx],
            title=f'Heatmap of rank correlations (Spearman), dataset: {dataset}'
        )

        fname = f"heatmap_rankcorrelations_spearman_{dataset}.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=300)
        plots.append((dataset, fname))

        fig = plot_method_heatmap(
            concordance_scores[idx],
            title=f'Heatmap of rank correlations (Kendall Tau), dataset: {dataset}', 
            value_col='kendall_corr'
        )

        fname_kendall = f"heatmap_rankcorrelations_kendall_{dataset}.png"
        fpath_kendall = os.path.join(output_dir, fname_kendall)
        fig.savefig(fpath_kendall, dpi=300)
        plots_kendall.append((dataset, fname_kendall))
    
    report_path = os.path.join(output_dir, "plotting_report.html")

    with open(report_path, "w") as f:
        f.write("<html><head><title>Report</title></head><body>\n")

        f.write("<h1>Comparisons of results (inter tool metrics)</h1>\n")

        for dataset, fname in plots:
            f.write(f"<h2>{dataset}</h2>\n")
            df = concordance_scores[idx]  # assumes concordance_scores[idx] is a DataFrame
            f.write(df.to_html(index=False, border=1))  # render HTML table
            f.write(f'<img src="{fname}" style="max-width:800px;"><br><br>\n')
            f.write(f'<img src="{fname_kendall}" style="max-width:800px;"><br><br>\n')

        f.write("</body></html>\n")

if __name__ == "__main__":
    main()