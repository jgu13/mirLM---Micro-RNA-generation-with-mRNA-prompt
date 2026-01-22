"""
Randomly select samples for train, validation and test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from Global_parameters import PROJ_HOME
datadir = os.path.join(PROJ_HOME, "TargetScan_dataset")

data_path_train = os.path.join(PROJ_HOME, datadir, "Positive_primates_validation_500_randomized_start_random_samples.csv")
df = pd.read_csv(data_path_train, sep="\t")
# df_train, df_rem = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
# df_val, df_test = train_test_split(df_rem, test_size=0.2, random_state=42, shuffle=False)
df_train = pd.read_csv(data_path_train, sep=",")
df_train_subset = df_train.sample(n=100, random_state=42)

df_train_subset.to_csv(
    os.path.join(
        os.path.dirname(data_path_train), f"Positive_primates_validation_500_randomized_start_mini.csv"
    ),
    sep=",",
    index=False
)

# df_val.to_csv(
#     os.path.join(
#         os.path.dirname(data_path_train), f"your_data_path_validation.csv"
#     ),
#     sep=",",
#     index=False
# )

# df_test.to_csv(
#     os.path.join(
#         os.path.dirname(data_path_train), f"your_data_path_test.csv"
#     ),
#     sep=",",
#     index=False
# )