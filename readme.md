## Analysis script for phosphoproteomics data from DIA-NN
Script is available either as a jupyter notebook, or as a standalone python file.
- The script expects three files to exist:
    1. example_report.pg_matrix.tsv : default pg_matrix.tsv generated by DIA-NN
    2. example_sample_table.tsv : Annotation for which columns in the report correspond to which sample groups
    3. example_comparisons.tsv : File describing which sample groups should be compared.
Refer to the original files for reference of what these should look like. Feel free to change these names in the code files to correspond to your files.
Data provided in the example files has been generated from a DIA-NN report pg matrix by random substitution of protein IDs, sample names, and some values. 


### Usage:
- Run from the command line: python analysis_onestep.py (this expects files to be named example_report.pg_matrix.tsv, example_sample_table.tsv, and example_comparisons.tsv).
- Run using jupyter notebook

### Requirements:
- Python 3.11 or newer
  - see requirements.txt for python packages. i.e. pip install -r requirements.txt
- R version 4.3.3 or newer
  - see R_requirements.R for packages. i.e. Rscript R_requirements.R

### Outputs:
1. Fully processed data.tsv
- This file contains filtered, median normalized and QRILC-imputated data
2. Produced comparisons.tsv
- This file contains fold changes and significance values for the comparisons performed according to the comparisons file
3. Volcano plots of each comparison.