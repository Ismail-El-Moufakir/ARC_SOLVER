from GP import *
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Custom color palette for values 0-9
custom_colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
]
cmap = ListedColormap(custom_colors)

def run_GP(Task_Filename):
    gp = Genetic_Prog(POPPULATION_SIZE, GENERATIOON_SIZE, max_depth=5)
    with open(Task_Filename) as file:
        Grid_dict = json.load(file)
    Best_Score_Over_Generation = []
    for task in Grid_dict["train"]:
        in_Task, out_Task = task["input"], task["output"]
        scores = gp.train(in_Task, out_Task)
        Best_Score_Over_Generation.extend(scores)
    Best_Ind = max(gp.Population, key=lambda ind: gp.scores[ind.Id])
    Best_Ind.Show_Tree().render('Best_Ind', format='pdf', cleanup=True, view=True)
    # Apply best individual to test input
    in_Test = Grid_dict["test"][0]["input"]
    true_out = Grid_dict["test"][0]["output"]
    out_pred = Best_Ind.execute(in_Test)
    input_grid = in_Test
    output_grid = true_out
    pred_grid = out_pred.construct_grid()  

    # Plot results
    fig = plt.figure(figsize=(12, 12))

    # 1. Best score evolution
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(Best_Score_Over_Generation)
    ax1.set_title('Best Score Over Generations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')

    # 2. Input grid
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(input_grid, cmap=cmap)
    ax2.set_title('Test Input')
    ax2.axis('off')

    # 3. Ground truth output
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(output_grid, cmap=cmap)
    ax3.set_title('Expected Output')
    ax3.axis('off')

    # 4. Best individual prediction
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(pred_grid, cmap=cmap)
    ax4.set_title('Best Individual Output')
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

run_GP(r"..\data\training\6fa7a44f.json")
