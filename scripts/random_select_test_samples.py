import os
import pandas as pd
from sklearn.model_selection import train_test_split
from Global_parameters import PROJ_HOME

mRNA_length = 30

data_path = os.path.join(PROJ_HOME, "TargetScan_dataset/positive_samples_30_randomized_start.csv")
df = pd.read_csv(data_path, sep=",")
df = df.sample(n=50000, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True)

train_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"positive_samples_30_randomized_start_train_random_samples.csv"
    ),
    sep=",",
    index=False
)

test_df.to_csv(
    os.path.join(
        os.path.dirname(data_path), f"positive_samples_30_randomized_start_validation_random_samples.csv"
    ),
    sep=",",
    index=False
)