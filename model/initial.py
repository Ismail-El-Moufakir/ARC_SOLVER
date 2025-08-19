import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from GP import Genetic_Prog
from multiprocessing import Pool
import numpy as np

# ----- Config -----
POPULATION_SIZE = 3000
GENERATION_SIZE = 20
NUM_SIMULATIONS = 30
TASK_FILE = "../data/training/6fa7a44f.json"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----- Custom colormap -----
custom_colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
]
cmap = ListedColormap(custom_colors)

# ----- Load Task Data -----
with open(TASK_FILE, 'r') as f:
    grid_dict = json.load(f)
train_grids = grid_dict["train"]
test_grids = grid_dict["test"]
input_task = train_grids[0]["input"]

# ----- Accuracy Function -----

def compute_accuracy(pred_grid, true_out):
   #check how many pixels are the same
    if np.array(true_out).shape != np.array(pred_grid).shape:
        return -1
    correct_pixels = 0
    for i in range(len(true_out)):
        for j in range(len(true_out[i])):
            if true_out[i][j] == pred_grid[i][j] and true_out[i][j] != 0:
                correct_pixels += 1
    total_pixels = np.array(true_out).size
    accuracy = correct_pixels / total_pixels
    return accuracy

# ----- Simulation Worker -----
def run_simulation(run_id):
    print(f"[Run {run_id}] Starting simulation...")

    # Initialize GP
    gp = Genetic_Prog(
        POPULATION_SIZE,
        GENERATION_SIZE,
        max_depth=5,
        input_task=input_task
    )

    # Train on all training grids
    for task in train_grids:
        gp.evaluate_population(task["input"], task["output"])

    # Select best individual from population
    best_individual = max(gp.Population, key=lambda ind: gp.scores[ind.Id])

    # Evaluate on test grid (assume only one test sample)
    test_input = test_grids[0]["input"]
    test_output = test_grids[0]["output"]

    predicted_output = best_individual.execute(test_input).construct_grid()
    accuracy = compute_accuracy(predicted_output, test_output)
    print(f"[Run {run_id}] Test Accuracy: {accuracy:.2%}")

    # Save output image
    img_path = os.path.join(RESULTS_DIR, f"run_{run_id:02d}_output.png")
    plt.imshow(predicted_output, cmap=cmap)
    plt.title(f"Run {run_id} - Accuracy: {accuracy:.2%}")
    plt.axis("off")
    plt.savefig(img_path)
    plt.close()
    print(f"[Run {run_id}] Output image saved: {img_path}")

    # Save accuracy as JSON
    score_path = os.path.join(RESULTS_DIR, f"run_{run_id:02d}_accuracy.json")
    with open(score_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "predicted_shape": np.array(predicted_output).shape,
            "target_shape": np.array(test_output).shape
        }, f, indent=2)
    print(f"[Run {run_id}] Accuracy saved: {score_path}\n")

    return f"Run {run_id} completed with accuracy {accuracy:.2%}"

# ----- Main Execution -----
if __name__ == "__main__":
    print("Starting 30 parallel simulations...\n")
    with Pool() as pool:
        results = pool.map(run_simulation, range(1, NUM_SIMULATIONS + 1))

    print("\nAll simulations completed.\n")
    for res in results:
        print(res)
