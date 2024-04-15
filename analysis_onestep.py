import os
import tempfile
import sh
import uuid
import numpy as np
import pandas as pd
from plotly import express as px
from math import ceil
import numpy as np
from statsmodels.stats import multitest
from scipy.stats import ttest_ind

config = {
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        },
    }


def run_rscript(r_script_contents:list, r_script_data: pd.DataFrame, replace_name: list):
    with tempfile.NamedTemporaryFile() as datafile:
        repwith = datafile.name
        r_script_contents = [line.replace(replace_name,repwith) for line in r_script_contents]
        r_script_data.to_csv(datafile, sep='\t')
        with tempfile.NamedTemporaryFile() as scriptfile:
            scriptfile.write('\n'.join(r_script_contents).encode('utf-8'))
            scriptfile.flush()
            try:
                sh.Rscript(scriptfile.name)
            except Exception as e:
                print('Error while running R script!')
                raise e
            with open(datafile.name, "r") as f:
                out = f.read().split('\n')
                out = [o.split('\t')[1:] for o in out[1:] if len(o)>0] # skip empty rows
    script_output_df = pd.DataFrame(data = out)
    script_output_df = script_output_df.replace('NA',np.nan).replace('',np.nan).astype(float)
    script_output_df.columns = r_script_data.columns # Restore columns and index in case R renames anything from either.
    script_output_df.index = r_script_data.index
    return script_output_df

def impute_qrilc(dataframe: pd.DataFrame, random_seed: int = 12) -> pd.DataFrame:
    tempname: uuid.UUID = str(uuid.uuid4())
    script: list = [
        'library("imputeLCMD")',
        f'set.seed({random_seed})',
        f'df <- read.csv("{tempname}",sep="\\t",row.names=1)',
        f'write.table(data.frame(impute.QRILC(df,tune.sigma=1)[1]),file="{tempname}",sep="\\t")'
    ]
    return run_rscript(script, dataframe, tempname)

def filter_missing(data_table: pd.DataFrame, sample_groups: dict, threshold: int = 60) -> pd.DataFrame:
    """Discards rows with more than threshold percent of missing values in all sample groups"""
    threshold: float = float(threshold)/100
    keeps: list = []
    for _, row in data_table.iterrows():
        keep: bool = False
        for _, sample_columns in sample_groups.items():
            keep = keep | (row[sample_columns].notna().sum()
                           >= ceil(threshold*len(sample_columns)))
            if keep:
                break
        keeps.append(keep)
    return data_table[keeps].copy()

def count_per_sample(data_table: pd.DataFrame, rev_sample_groups: dict) -> pd.Series:
    """Counts non-zero values per sample (sample names from rev_sample_groups.keys()) and returns a series with sample names in index and counts as values."""
    index: list = list(rev_sample_groups.keys())
    retser: pd.Series = pd.Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    )
    return retser

def median_normalize(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Median-normalizes a dataframe by dividing each column by its median.

    Args:
        df (pandas.DataFrame): The dataframe to median-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The median-normalized dataframe.
    """
    # Calculating the medians prior to looping is about 2-3 times more efficient,
    # than calculating the median of each column inside of the loop.
    medians: pd.Series = data_frame.median(axis=0)
    mean_of_medians: float = medians.mean()
    newdf: pd.DataFrame = pd.DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        newdf[col] = (data_frame[col] / medians[col]) * mean_of_medians
    return newdf 

def differential(data_table: pd.DataFrame, sample_groups: dict, comparisons: list, adj_p_thr: float = 0.01, fc_thr:float = 1.0) -> pd.DataFrame:
    sig_data: list = []
    for sample, control in comparisons:
        sample_columns: list = sample_groups[sample]
        control_columns: list = sample_groups[control]
        log2_fold_change: pd.Series = data_table[sample_columns].mean(
            axis=1) - data_table[control_columns].mean(axis=1)
        sample_mean_val: pd.Series = data_table[sample_columns].mean(axis=1)
        control_mean_val: pd.Series = data_table[control_columns].mean(axis=1)
        # Calculate the p-value for each protein using a two-sample t-test
        p_value: float = data_table.apply(lambda x: ttest_ind(x[sample_columns], x[control_columns])[1], axis=1)

        # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
        _: Any
        p_value_adj: np.ndarray
        _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

        # Create a new dataframe containing the fold change and adjusted p-value for each protein
        result: pd.DataFrame = pd.DataFrame(
            {
                'fold_change': log2_fold_change, 
                'p_value_adj': p_value_adj,
                'p_value_adj_neg_log10': -np.log10(p_value_adj),
                'p_value': p_value,
                'sample_mean_value': sample_mean_val,
                'control_mean_value': control_mean_val})
        
        result['Name'] = data_table.index.values
        result['Sample'] = sample
        result['Control'] = control
        result['Significant'] = ((result['p_value_adj']<adj_p_thr) & (result['fold_change'].abs() > fc_thr))
        result.sort_values(by='Significant',ascending=True,inplace=True)
        sig_data.append(result)
        
    return pd.concat(sig_data,ignore_index=True)[
        ['Sample',
         'Control',
         'Name',
         'Significant',
         'fold_change',
         'p_value',
         'p_value_adj',
         'p_value_adj_neg_log10',
         'sample_mean_value',
         'control_mean_value'
         ]]

def volcano_plot(
    data_table, title: str = None, fc_axis_min_max: float = 2, highlight_only: list = None,
    adj_p_threshold: float = 0.01, fc_threshold: float = 1.0
) -> tuple:
    """Draws a Volcano plot of the given data_table

    :param data_table: data table from stats.differential. Should only contain one comparison.
    :param title: Figure title
    :param fc_axis_min_max: minimum for the maximum value of fold change axis. Default of 2 is used to keep the plot from becoming ridiculously narrow
    :param adj_p_threshold: threshold of significance for the calculated adjusted p value (Default 0.01)
    :param fc_threshold: threshold of significance for the log2 fold change. Proteins with fold change of <-fc_threshold or >fc_threshold are considered significant (Default 1)
    :param highlight_only: only highlight significant ones that are also in this list

    :returns: volcano_plot: go.Figure
    """

    data_table['Highlight'] = [row['Name'] if row['Significant'] else '' for _, row in data_table.iterrows()]
    fig: go.Figure = px.scatter(
        data_table,
        x='fold_change',
        y='p_value_adj_neg_log10',
        title=title,
        color='Significant',
        text='Highlight',
        height=800,
        width=800,
        render_mode='svg',
        hover_data=['Name','Significant','p_value_adj_neg_log10','fold_change']
    )

    # Set yaxis properties
    p_thresh_val: float = -np.log10(adj_p_threshold)
    pmax: float = max(
        data_table['p_value_adj_neg_log10'].max(), p_thresh_val)+0.5
    fig.update_yaxes(title_text='-log10 (q-value)', range=[0, pmax])
    # Set the x-axis properties
    fcrange: float = max(abs(data_table['fold_change']).max(), fc_threshold)
    if fcrange < fc_axis_min_max:
        fcrange = fc_axis_min_max
    fcrange += 0.25
    fig.update_xaxes(title_text='Log2 fold change', range=[-fcrange, fcrange])
    # Add vertical lines indicating the significance thresholds
    fig.add_shape(type='line', x0=-fc_threshold, y0=0, x1=-
                  fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    fig.add_shape(type='line', x0=fc_threshold, y0=0,
                  x1=fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    # And horizontal line:
    fig.add_shape(type='line', x0=-fcrange, y0=p_thresh_val,
                  x1=fcrange, y1=p_thresh_val, line=dict(width=2, dash='dot'))

    # Return the plot
    return fig

data_table = pd.read_csv('example_report.pg_matrix.tsv',sep='\t',index_col = 'Protein.Group')
sample_table = pd.read_csv('example_sample_table.tsv',sep='\t')
sample_groups = {}
for _,row in sample_table.iterrows():
    sg = row['Sample group']
    if sg not in sample_groups:
        sample_groups[sg] = []
    sample_groups[sg].append(row['Sample name'])
    
sample_groups_rev = {}
for samplegroup,v in sample_groups.items():
    for samplename in v:
        sample_groups_rev[samplename] = samplegroup
data_table.drop(columns=[c for c in data_table.columns if c not in sample_groups_rev],inplace=True)
data_table = filter_missing(data_table, sample_groups)
data_table = np.log2(data_table)
data_table = median_normalize(data_table)
data_table = impute_qrilc(data_table)
data_table.to_csv('Fully processed data.tsv',sep='\t')

comparisons = []
with open('example_comparisons.tsv') as fil:
    for line in fil:
        comparisons.append(line.strip().split('\t'))
comparisons = [c for c in comparisons if ((c[0] in sample_groups) and (c[1] in sample_groups))]
    
fc_thr = 1
p_thr = 0.01
significant_data: pd.DataFrame = differential(
    data_table, sample_groups, comparisons, fc_thr=fc_thr, adj_p_thr=p_thr)
significant_data.to_csv('Produced comparisons.tsv',sep='\t',index=False)

for _, row in significant_data[['Sample', 'Control']].drop_duplicates().iterrows():
    sample: str = row['Sample']
    control: str = row['Control']
    volcano_plot(
        significant_data[(significant_data['Sample'] == sample) & (
            significant_data['Control'] == control)],
        adj_p_threshold=p_thr, fc_threshold=fc_thr
    ).write_html(f'Volcano plot {sample} vs {control}',config=config)
    