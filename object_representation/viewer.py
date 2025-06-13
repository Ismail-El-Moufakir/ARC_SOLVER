import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap

import numpy as np
# Custom color palette for values 0-9
custom_colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
]
cmap = ListedColormap(custom_colors)

def display_grid_pairs(grid_pairs, title):
    n_rows = len(grid_pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]  
    for i, pair in enumerate(grid_pairs):
        axes[i][0].imshow(pair["input"], cmap=cmap, vmin=0, vmax=9)
        axes[i][0].set_title("Input")
        axes[i][1].imshow(pair["output"], cmap=cmap, vmin=0, vmax=9)
        axes[i][1].set_title("Output")
        for ax in axes[i]:
            ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def Display_Task(Filename: str):
    with open(Filename, 'r') as file:
        Grid_dict = json.load(file)
    #train_fig = display_grid_pairs(Grid_dict["train"], f"Training Examples - {Filename}")
    test_fig = display_grid_pairs(Grid_dict["test"], f"Testing Examples - {Filename}")
    
def display_grid(grid):
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    plt.axis('off')
    plt.show()


