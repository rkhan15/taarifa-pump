import pandas as pd
import yaml
import numpy as np
import os


def get_config(config_path):
    return yaml.safe_load(open(config_path))


def taarifa_working_dir():
    return os.getcwd()[-12:] == 'taarifa-pump'


def get_values(config, val_type):
    """Returns training values if val_type = train, or test set values if val_type = test. """
    assert taarifa_working_dir() is True
    parse_dts, remove_cols, data_types = get_config_columns(config['column_dtypes'])
    path_pref = ''
    if val_type == 'train':
        path_pref = 'training'
    elif val_type == 'test':
        path_pref = 'test-set'
    return pd.read_csv(path_pref + '-values.csv', low_memory=False,
                       parse_dates=parse_dts, usecols=lambda x: x not in remove_cols,
                       dtype=data_types)


def get_training_labels():
    assert taarifa_working_dir() is True
    return pd.read_csv('training-labels.csv', low_memory=False)


def get_merged_training_data(config):
    training_values = get_values(config=config, val_type='train')
    training_labels = get_training_labels()
    return pd.merge(training_values, training_labels, how='inner', on='id')


def sort_by_nunique(df, asc=True):
    return pd.DataFrame(df.nunique()).reset_index().rename(columns={'index': 'attribute', 0: 'count'}).sort_values(
        by='count', ascending=asc)


def print_value_counts(df, cols=None):
    columns = df.columns.to_list()[1:] if cols is None else cols
    for col in columns:
        print(col)
        print('-------')
        print(df[col].value_counts())
        print()


def get_correlations(df):
    corrs = df.iloc[:, 1:].corr().abs()
    sorted_corrs = corrs.unstack()
    sorted_corrs = sorted_corrs.sort_values(kind="quicksort", ascending=False)
    sorted_corrs = pd.DataFrame(sorted_corrs).reset_index()
    sorted_corrs = sorted_corrs[sorted_corrs['level_0'] != sorted_corrs['level_1']].iloc[::2]
    return sorted_corrs


def get_config_columns(column_dtypes):
    """Reads in columns based on specified types.
    Types were specified during initial data cleaning processes. """
    parse_dts, remove_cols = [], []
    data_types = dict()
    dtype_map = {'float_cols': np.float32, 'int_cols': np.int,
                 'str_cols': str, 'boolean_cols': bool}
    for data_type, col_lst in column_dtypes.items():
        if data_type == 'date_cols':
            parse_dts = col_lst
        elif data_type == 'remove_cols':
            remove_cols = col_lst
        else:
            data_types.update({col: dtype_map[data_type] for col in col_lst})
    return parse_dts, remove_cols, data_types


def get_missing_data_info(df):
    """Used during initial data cleaning processes. """
    if not df.isnull().values.any():
        print('No missing values!')
    print('Total count of missing values:', df.isnull().sum().sum())
    print()
    print('Total count by column:')
    print(df.isnull().sum())


def get_basin_region_combos(df, variance_column):
    """Returns variance of a column based on combinations of basin and region combos. """
    var_col_name = str(variance_column + '_var')
    basin_region_combo_vars = pd.DataFrame(columns={'basin', 'region', 'num_waterpoints', var_col_name})
    for basin in list(df['basin'].unique()):
        df_basin = df[df['basin'] == basin]
        for region in list(df['region'].unique()):
            df_basin_region = df_basin[df_basin['region'] == region]
            num_waterpoints = df_basin_region.shape[0]
            if num_waterpoints == 0:
                continue
            basin_region_combo_vars = basin_region_combo_vars.append(
                {'basin': basin,
                 'region': region,
                 'num_waterpoints': num_waterpoints,
                 var_col_name: np.var(df_basin_region[[variance_column]]).values[0]},
                ignore_index=True)
    basin_region_combo_vars = basin_region_combo_vars[['basin', 'region', var_col_name]]
    basin_region_combo_vars.sort_values(by=var_col_name, ascending=False, inplace=True)
    return basin_region_combo_vars
