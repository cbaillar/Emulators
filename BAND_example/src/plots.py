import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D
import pandas as pd

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image

from scipy.stats.qmc import Sobol

            
def plot_design_points(train_points, validation_points):
    num_params = train_points.shape[1]
    param_name = ['A', 'B', 'C', 'D', 'E']

    fig, axes = plt.subplots(num_params, num_params, figsize=(15, 15))

    axes = axes.flatten()

    # Initialize variables to collect handles and labels for the legend
    handles, labels = None, None

    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i * num_params + j]
            
            if j > i:
                # Turn off the upper off-diagonal plots
                ax.axis('off')
                continue
            
            if i == j:
                # If the same parameter, create a histogram
                hist_train = ax.hist(train_points[:, i], bins=20, alpha=0.5, label='Train', color='blue')
                hist_validation = ax.hist(validation_points[:, i], bins=20, alpha=0.5, label='Validation', color='orange')
                ax.set_ylabel('Frequency')

                # Collect handles and labels for the legend only once
                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
            else:
                # Scatter plot of train and validation points for lower triangular part
                scatter_train = ax.scatter(train_points[:, j], train_points[:, i], alpha=0.5, label='Train', color='blue')
                scatter_validation = ax.scatter(validation_points[:, j], validation_points[:, i], alpha=0.5, label='Validation', color='orange')
                
            # Set labels for x-axis and y-axis with correct rotation
            if i == num_params - 1:  # Set x-axis label for the bottom row
                ax.set_xlabel(f'{param_name[j]}', fontsize=20)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

            if j == 0:  # Set y-axis label for the first column
                ax.set_ylabel(f'{param_name[i]}', fontsize=20)
                plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set a global legend for the figure
    fig.legend(handles, labels, fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Add spacing between subplots for readability
    plt.show()


def plot_data(pT, RAA, sigma, cent, system, ax=None):
    color = 'red' if cent == 'C0' else 'green'

    if ax is None:
        ax = plt.gca()
    
    ax.scatter(pT, RAA, color=color, label=f'{system} {cent}')
    ax.errorbar(pT, RAA, yerr=sigma, color=color, fmt='o', markersize=5, capsize=3, label=f'{system} {cent} stat + sys error')

    ax.set_xlabel('$p_T$ (GeV/c)', fontsize=18)
    ax.set_ylabel('$R_{AA}$', fontsize=18)
    ax.set_title(f'Nuclear Modification Factor $R_{{AA}}$ for {system} Collisions', fontsize=18)

    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))


def plot_combined_box_rmse(ax, y_true, surmise_pred, smooth_pred, sckit_pred, label, plot_type="Validation"):
    y_true = np.asarray(y_true)
    surmise_pred = np.asarray(surmise_pred)
    smooth_pred = np.asarray(smooth_pred)
    sckit_pred = np.asarray(sckit_pred)

    # Compute root mean square errors
    surmise_rmse = np.sqrt((surmise_pred - y_true) ** 2)
    smooth_rmse = np.sqrt((smooth_pred - y_true) ** 2)
    sckit_rmse = np.sqrt((sckit_pred - y_true) ** 2)

    # Combine errors for the box plot
    combined_errors = [surmise_rmse.flatten(), smooth_rmse.flatten(), sckit_rmse.flatten()]

    # Plotting
    ax.boxplot(combined_errors, tick_labels=['Surmise', 'Smooth', 'Scikit-learn'])
    ax.set_title(f'{label}', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)


def mean_rmse(ax, y_true, surmise_pred, smooth_pred, sckit_pred, label, plot_type="Validation"):
    y_true = np.asarray(y_true)
    surmise_pred = np.asarray(surmise_pred)
    smooth_pred = np.asarray(smooth_pred)
    sckit_pred = np.asarray(sckit_pred)

    # Compute mean RMSE for each model
    surmise_rmse = np.sqrt(np.mean((surmise_pred - y_true) ** 2))  # Mean RMSE for Surmise
    smooth_rmse = np.sqrt(np.mean((smooth_pred - y_true) ** 2))    # Mean RMSE for Smooth
    sckit_rmse = np.sqrt(np.mean((sckit_pred - y_true) ** 2))      # Mean RMSE for Scikit-learn

    return {'Surmise': surmise_rmse, 'Smooth': smooth_rmse, 'Scikit-learn': sckit_rmse}
    

def plot_trace(samples, parameter_names, title):
    num_params = samples.shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), sharex=True)
    
    for i, param in enumerate(parameter_names):
        axes[i].plot(samples[:, i])
        axes[i].set_ylabel(param)
        axes[i].set_xlabel('Iteration')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])