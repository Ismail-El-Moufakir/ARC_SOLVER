import os
import json
import logging
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime
from GP import *  


def Test_Output(true_out, pred_grid):
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
# Custom color palette
custom_colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
]
cmap = ListedColormap(custom_colors)

# Single simulation function
def simulate_gp_run(task_filename, result_dir, run_id):
    run_folder = os.path.join(result_dir, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    log_file = os.path.join(run_folder, "run.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logger = logging.getLogger()

    try:
        logger.info(f"Run {run_id} - Loading task file {task_filename}")
        with open(task_filename) as f:
            grid_dict = json.load(f)
        input_task = grid_dict["train"][0]["input"]
        gp = Genetic_Prog(POPPULATION_SIZE, GENERATIOON_SIZE, max_depth=3,input_task=input_task)
        best_scores = []

        for i, task in enumerate(grid_dict["train"]):
            in_task, out_task = task["input"], task["output"]
            logger.info(f"Training on train example {i}")
            scores = gp.train(in_task, out_task)
            logger.info(f"trying best Individual")
            gp.evaluate_population(in_task,out_task)
            Individual = max(gp.Population, key=lambda ind: gp.scores[ind.Id])
            O = Individual.execute(in_task).construct_grid()
            logger.info(f"comparaison {O} with {np.array(out_task)}")
            for layer in Individual.execute(in_task).Layers:
                logger.info(f"Object at {layer.Position} with color {layer.color} and shape matrix:\n{layer.Shape_Mtx}")
            best_scores.extend(scores)
            

        # Best individual
        best_ind = max(gp.Population, key=lambda ind: gp.scores[ind.Id])
        best_ind.Show_Tree().render(os.path.join(run_folder, "best_tree"), format='pdf', cleanup=True)
        # Test set application
        in_test = grid_dict["test"][0]["input"]
        true_out = grid_dict["test"][0]["output"]
        out_pred = best_ind.execute(in_test)
        pred_grid = out_pred.construct_grid()

        # Plotting
        fig = plt.figure(figsize=(12, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(best_scores)
        ax1.set_title("Best Score Over Generations")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Score")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(in_test, cmap=cmap)
        ax2.set_title("Test Input")
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(true_out, cmap=cmap)
        ax3.set_title("Expected Output")
        ax3.axis("off")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(pred_grid, cmap=cmap)
        ax4.set_title("Predicted Output")
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "results.png"))
        plt.close()
        logger.info("Plot saved.")
        # save output grid and prediction in json
        with open(os.path.join(run_folder, "output.json"), 'w') as f:
            output_data = {
                "expected_output": list(true_out),
                "predicted_output": pred_grid.tolist() if pred_grid is not None else [],
            }
            json.dump(output_data, f)

    except Exception as e:
        logger.exception(f"Run {run_id} failed: {e}")

# Run N simulations on the same task file
def run_simulations_on_task(task_file, result_dir_path, num_simulations=4):
    os.makedirs(result_dir_path, exist_ok=True)
    # calculate how accuracy of simulation in each runs
    scores_over_runs = []


    processes = []
    for run_id in range(num_simulations):
        p = multiprocessing.Process(
            target=simulate_gp_run,
            args=(task_file, result_dir_path, run_id)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# Example usage
if __name__ == "__main__":
    task_file = "../data/training/1f642eb9.json"
    result_dir = "../results"
    run_simulations_on_task(task_file, result_dir, num_simulations=5)
    #compute accuracy of each runs
    accuracy_over_runs = []
    for run_id in range(20):
        run_folder = os.path.join(result_dir, f"run_{run_id}")
        with open(os.path.join(run_folder, "output.json"), 'r') as f:
            output_data = json.load(f)
            true_out = output_data["expected_output"]
            pred_grid = output_data["predicted_output"]
            accuracy = Test_Output(true_out, pred_grid)
            print(accuracy)
            accuracy_over_runs.append(accuracy)
    plt.plot(accuracy_over_runs)
    plt.title("Accuracy Over Runs")
    plt.xlabel("Run ID")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_dir, "accuracy_over_runs.png"))

