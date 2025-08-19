import os
import json
import matplotlib.pyplot as plt

# Directory containing individual accuracy JSONs
RESULTS_DIR = "results"
SUMMARY_FILE = "all_accuracies.json"

# Collect and sort accuracy files
accuracy_files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.endswith("_accuracy.json")]
)

accuracies = []
summary_data = {}

for file in accuracy_files:
    path = os.path.join(RESULTS_DIR, file)
    with open(path, 'r') as f:
        data = json.load(f)
        acc = data.get("accuracy", None)
        if acc is not None:
            run_number = int(file.split('_')[1])
            accuracies.append((run_number, acc))
            summary_data[f"run_{run_number:02d}"] = acc

# Sort by run number
accuracies.sort(key=lambda x: x[0])
run_ids, accuracy_values = zip(*accuracies)

# Plot accuracies over runs
plt.figure(figsize=(10, 5))
plt.plot(run_ids, accuracy_values, marker='o')
plt.title("Test Accuracy over Runs")
plt.xlabel("Run Number")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()

# Save summary JSON
with open(SUMMARY_FILE, "w") as out_file:
    json.dump(summary_data, out_file, indent=2)

print(f"Saved accuracy plot as 'accuracy_plot.png'")
print(f"Saved accuracy summary JSON as '{SUMMARY_FILE}'")
