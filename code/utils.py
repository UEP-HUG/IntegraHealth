import os
import sys
os.environ['USE_PYGEOS'] = '0'  

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def read_data(file_path, index_col=None, geo=False, lon=None, lat=None):
    """
    Read a dataframe from a parquet file and optionally set it as a GeoDataFrame.

    Parameters:
    file_path (str or Path): The file path of the data to read.
    index_col (str): The column to set as the dataframe index.
    geo (bool): If True, set the dataframe as a GeoDataFrame.
    lon (str): The longitude column name, needed if geo is True.
    lat (str): The latitude column name, needed if geo is True.

    Returns:
    DataFrame or GeoDataFrame: The loaded dataframe.
    """
    df = pd.read_parquet(file_path)
    if index_col is not None:
        df.set_index(index_col, inplace=True)
    if geo:
        if lon is None or lat is None:
            raise ValueError("Both 'lon' and 'lat' must be specified if 'geo' is True.")
        df = gpd.GeoDataFrame(df, crs=4326, geometry=gpd.points_from_xy(df[lon], df[lat]))
    return df


def optimize_memory_df(df):
    """
    Convert each column of a pandas DataFrame to the datatype that takes the lowest memory.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to convert.

    Returns:
    --------
    pandas DataFrame
        The converted DataFrame with lowest memory datatypes for each column.
    """

    # First, convert all object columns to category type
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype('category')

    # Next, loop through all numeric columns and downcast the data types
    for col in df.select_dtypes(include=['int', 'float']).columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == 'int':
            # Use smallest integer type possible
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
    return df

def feature_map(df, col, cmap, title, lakes, cantons_ch, out_image, ch_extent, result_folder, _reversed=True):
    """
    Function to generate and save a map based on provided dataframe and column.
    
    Parameters:
    df (GeoDataFrame): Input geopandas DataFrame with geographic data.
    col (str): The column to plot.
    cmap (str): Colormap to use.
    title (str): Title for the plot and filename for the saved image.
    lakes (GeoDataFrame): Geopandas DataFrame with lakes data.
    cantons_ch (GeoDataFrame): Geopandas DataFrame with cantons data.
    out_image (ndarray): Image data for the overlay.
    ch_extent (list): The extent of the map (xmin, ymin, xmax, ymax).
    result_folder (str or Path): Path to the folder where the result will be saved.
    _reversed (bool): Whether to reverse the colormap. Default is True.
    
    Returns:
    AxesSubplot: The generated map plot.
    """
    filename = f"{title}.png"
    map_file_path = result_folder / 'Maps features' / filename

    if os.path.isfile(map_file_path):
        print('Map already generated')
        return

    if _reversed:
        cmap += '_r'

    ax = df.to_crs(21781).plot(col, markersize=0.5, linewidth=0.1, cmap=cmap, 
                               legend=True, figsize=(8, 8), legend_kwds={'shrink':0.5})

    lakes.plot(color='lightblue', ax=ax)
    cantons_ch.geometry.boundary.plot(ax=ax, edgecolor='k', color=None, linewidth=0.1)

    plt.imshow(out_image.squeeze(), extent=ch_extent, cmap='Greys_r', alpha=0.4)
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, color='grey')
    plt.savefig(map_file_path, dpi=300, bbox_inches='tight')
    
    return ax

def show_values(axs, orient="v", digits=2, fontsize=8, space=.05):
    """
    Add data values on top of bars in bar plots.
    
    Parameters:
    axs (AxesSubplot or ndarray): Single axes instance or a numpy array of axes.
    orient (str): The orientation of the bar plot, either 'v' for vertical or 'h' for horizontal.
    digits (int): The number of digits after the decimal point for the value to be shown. Default is 2.
    fontsize (int): The font size for the value to be shown. Default is 8.
    space (float): The space between the end of the bar and the value to be shown. Default is 0.05.
    
    Returns:
    None. The function adds text to the axes in-place.
    """
    if orient not in ['v', 'h']:
        raise ValueError("orient must be 'v' or 'h'.")

    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.02)
                value = f'{p.get_height():.{digits}f}'
                ax.text(_x, _y, value, size=fontsize, ha="center") 
        else:  # orient == "h"
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = f'{p.get_width():.{digits}f}'
                ax.text(_x, _y, value, size=fontsize, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
def find_intersection(row, cols):
    """
    Finds the intersection of sets in specified columns of a DataFrame row.

    Parameters:
    row (Series): A row of a DataFrame.
    cols (list of str): A list of column names in the DataFrame.

    Returns:
    list: A list containing the elements that are common to all specified columns.
    """
    sets = (set(row[col]) for col in cols)
    return list(set.intersection(*sets))

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

## Helpers function for PSM
def plot_covariate_distributions(df_sample, df_matched, covariates, figure_title):
    C_COLOUR = 'grey'
    T_COLOUR = 'green'
    C_LABEL = 'Control'
    T_LABEL = 'Treatment'

    for var in covariates:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Visualize original distribution
        sns.kdeplot(data=df_sample[df_sample['treatment'] == 0], x=var, fill=True,
                    color=C_COLOUR, label=C_LABEL, ax=ax[0])
        sns.kdeplot(data=df_sample[df_sample['treatment'] == 1], x=var, fill=True,
                    color=T_COLOUR, label=T_LABEL, ax=ax[0])
        ax[0].set_title('Before matching')

        # Visualize new distribution
        sns.kdeplot(data=df_matched[df_matched['treatment'] == 0], x=var,
                    fill=True, color=C_COLOUR, label=C_LABEL, ax=ax[1])
        sns.kdeplot(data=df_matched[df_matched['treatment'] == 1], x=var,
                    fill=True, color=T_COLOUR, label=T_LABEL, ax=ax[1])
        ax[1].set_title('After matching')
        ax[1].set_ylabel("")
        
        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], color=C_COLOUR, lw=4, label=C_LABEL),
                           plt.Line2D([0], [0], color=T_COLOUR, lw=4, label=T_LABEL)]

        plt.tight_layout()
        
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=2)
    fig.suptitle(figure_title, fontsize=16, y=1.05)
    plt.show()
    
def plot_match(control_data, treatment_data, matched_entity='propensity_score', Title='Side by side matched controls', Ylabel='Number of patients', Xlabel='Propensity score', names=['treatment', 'control'], colors=['#E69F00', '#56B4E9'], save=False):
        """
        knn_matched -- Match data using k-nn algorithm
        Parameters
        ----------
        matched_entity : str
           string that will used to match - propensity_score or proppensity_logit
        Title : str
           Title of plot
        Ylabel : str
           Label for y axis
        Xlabel : str
           Label for x axis
        names  : list
           List of 2 groups
        colors : str
           string of hex code for group 1 and group 2
        save   : Bool
            Whether to save the figure in pwd (default = False)
        Returns
        grpahic
        """

        x1 = treatment_data[matched_entity]
        x2 = control_data[matched_entity]
        # Assign colors for each airline and the names
        colors = colors
        names = names
        sns.set_style("white")
        # Make the histogram using a list of lists
        # Normalize the flights and assign colors and names
        plt.hist([x1, x2], color=colors, label=names)
        # Plot formatting
        plt.legend()
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title)
        if save == True:
            plt.savefig('propensity_match.png', dpi=250)
        else:
            pass
# Compare the covariate balance before and after matching
def compare_balance(data, matched_data, covariate_columns, treatment_var):
    pre_matching_balance = data.groupby(treatment_var)[covariate_columns].mean().T
    post_matching_balance = matched_data.groupby(treatment_var)[covariate_columns].mean().T
    
    balance_comparison = pd.concat([pre_matching_balance, post_matching_balance], axis=1)
    balance_comparison.columns = ['Pre_matching_control', 'Pre_matching_treatment', 'Post_matching_control', 'Post_matching_treatment']
    
    return balance_comparison

def plot_categorical_proportional_diff(df_sample, df_matched, categorical_column, treatment_var):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Calculate proportions in the original dataset
    original_treatment_prop = df_sample[df_sample[treatment_var] == 1][categorical_column].value_counts(normalize=True)
    original_control_prop = df_sample[df_sample[treatment_var] == 0][categorical_column].value_counts(normalize=True)

    # Calculate proportions in the matched dataset
    matched_treatment_prop = df_matched[df_matched[treatment_var] == 1][categorical_column].value_counts(normalize=True)
    matched_control_prop = df_matched[df_matched[treatment_var] == 0][categorical_column].value_counts(normalize=True)

    # Calculate proportional differences before and after matching
    original_diff = original_treatment_prop - original_control_prop
    matched_diff = matched_treatment_prop - matched_control_prop

    # Merge original and matched differences into a single dataframe
    diff_df = pd.concat([original_diff, matched_diff], axis=1)
    diff_df.columns = ['Before matching', 'After matching']
    diff_df.fillna(0, inplace=True)

    # Plot proportional differences
#     diff_df.plot(kind='bar', ax=ax, legend=True)
    sns.barplot(data=diff_df, dodge=False, ax=ax)

    ax.set_title(f'Proportional Differences for {categorical_column}')
    ax.set_ylabel('Difference')
    ax.set_xlabel('')
    ax.grid(axis='y') 
    plt.tight_layout()
    plt.show()
def compute_mean_differences_and_proportions(df_before, df_after, variable_names, treatment_var):
    results = []
    for old_name, new_name in zip(variable_names['old'], variable_names['new']):
        mean_diff_before = df_before.loc[df_before[treatment_var] == 1, old_name].mean() - df_before.loc[df_before[treatment_var] == 0, old_name].mean()
        mean_diff_after = df_after.loc[df_after[treatment_var] == 1, old_name].mean() - df_after.loc[df_after[treatment_var] == 0, old_name].mean()

        if 'cat' in old_name:
            prop_diff_before = df_before.loc[df_before[treatment_var] == 1, old_name].value_counts(normalize=True).max() - df_before.loc[df_before[treatment_var] == 0, old_name].value_counts(normalize=True).max()
            prop_diff_after = df_after.loc[df_after[treatment_var] == 1, old_name].value_counts(normalize=True).max() - df_after.loc[df_after[treatment_var] == 0, old_name].value_counts(normalize=True).max()
            results.append((new_name, prop_diff_before, prop_diff_after))
        else:
            results.append((new_name, mean_diff_before, mean_diff_after))
    return pd.DataFrame(results, columns=['Variable', 'Before', 'After'])

def love_plot(df_mean_diffs_and_props, threshold = 0.1, xlim=100):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axvline(0, color='grey', linestyle='--')
    sns.scatterplot(data=df_mean_diffs_and_props, x='Before', y='Variable', ax=ax, color='red', label='Before')
    sns.scatterplot(data=df_mean_diffs_and_props, x='After', y='Variable', ax=ax, color='blue', label='After')
    ax.set_title('Covariate balance')
    ax.set_xlabel('Mean Differences / Proportions')
    ax.set_ylabel('')
    # Add two horizontal dotted lines at a defined threshold
    ax.axvline(x=threshold, color='black', linestyle=':')
    ax.axvline(x=0, color='black')
    ax.axvline(x=-threshold, color='black', linestyle=':')
#     ax.set_xscale('log')
    ax.set_xlim(-1,xlim)
    ax.legend()
    plt.show()

## Sensitivity analysis for k number of neighbors

def sensitivity_analysis_k_neighbors(data, treatment_data, control_data, covariate_columns, treatment_var, outcome_var, k_values):
    treatment_effects = []
    
    for k in k_values:
        # Perform k-nearest neighbors matching
        matcher = NearestNeighbors(n_neighbors=k)
        matcher.fit(control_data['propensity_score'].values.reshape(-1, 1))
        distances, indices = matcher.kneighbors(treatment_data['propensity_score'].values.reshape(-1, 1))

        # Create a matched dataset for each k
        matched_control_data = control_data.iloc[np.concatenate(indices)]
        df_matched = pd.concat([treatment_data, matched_control_data]).reset_index(drop=True)
        # df_matched = pd.merge(df_matched, data[['uuid',outcome_var]].sample(20000, random_state=42), on = 'uuid')

        # Calculate the ATE for each k
        ate = df_matched.groupby(treatment_var)[outcome_var].mean().diff().iloc[-1]
        treatment_effects.append(ate)
    
    return pd.DataFrame({'k_neighbors': k_values, 'ATE': treatment_effects})
