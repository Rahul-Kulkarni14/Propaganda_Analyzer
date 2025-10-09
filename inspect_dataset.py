import pandas as pd

# Load the final combined dataset
df = pd.read_csv("processed/final_dataset.csv")

# Show first few rows
print(df.head())

# Check the distribution of labels
print(df['label'].value_counts())
