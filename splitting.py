import pandas as pd
import os

# Read original file
df = pd.read_csv('accepted_2007_to_2018Q4.csv')

# Shuffle and split
df_80 = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
df_20 = df.drop(df_80.index).reset_index(drop=True)

# Create data folder if it doesn't exist
#os.makedirs('data', exist_ok=True)

# Save to files
df_80.to_csv('data_80.csv', index=False)
df_20.to_csv('data_20.csv', index=False)

print(f"✅ Split complete:")
print(f"   Total rows : {len(df)}")
print(f"   80% file   : {len(df_80)} rows → data_80.csv")
print(f"   20% file   : {len(df_20)} rows → data_20.csv")