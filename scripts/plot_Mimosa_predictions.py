"""
This script reads the data from Mimosa_dataset/zero_shot_prediction_best_composite_0.9911_0.9977_epoch11.csv 
and Mimosa_dataset/zero_shot_prediction_best_composite_1.0000_1.0000_epoch10.csv and plots the binding probabilities 
predictions in box plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from Global_parameters import PROJ_HOME

# read the data from Mimosa_dataset/zero_shot_prediction_best_composite_0.9911_0.9977_epoch11.csv
data = pd.read_csv(os.path.join(PROJ_HOME, "Mimosa_dataset/zero_shot_prediction_best_composite_0.9911_0.9977_epoch11.csv"))

# read the data from Mimosa_dataset/zero_shot_prediction_best_composite_1.0000_1.0000_epoch10.csv
data2 = pd.read_csv(os.path.join(PROJ_HOME, "Mimosa_dataset/zero_shot_prediction_best_composite_1.0000_1.0000_epoch10.csv"))

# plot the binding probabilities predictions in box plots
plt.figure(figsize=(3, 4))
box_colors = ["#1f77b4", "#ff7f0e"]
boxplots = plt.boxplot([data["binding_probs"], data2["binding_probs"]], patch_artist=True)
for box, color in zip(boxplots["boxes"], box_colors):
    box.set_facecolor(color)

plt.xlabel("Model")
plt.ylabel("Binding Probability")
# plt.title("Binding Probability Prediction\n(Mimosa dataset)")
plt.legend(
    boxplots["boxes"],
    ["Before TS finetuning", "After TS finetuning"],
    loc="center",
    bbox_to_anchor=(0.5, 1.08),
    frameon=False,
)
# leave enough headroom so tight_layout doesn't squeeze the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
# create save path
save_path = os.path.join(PROJ_HOME, "Performance/Mimosa_dataset")
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, "binding_probability_predictions.png"))
print(f"Binding probability predictions saved to {os.path.join(save_path, 'binding_probability_predictions.png')}")
