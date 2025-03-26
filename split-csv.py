import pandas as pd

file_name_no_ext = "sample-pt"

df = pd.read_csv(f"{file_name_no_ext}.csv")

df1 = df.iloc[:len(df)//2]
df2 = df.iloc[len(df)//2:]

df1.to_csv(f"{file_name_no_ext}_1.csv", index=False)
df2.to_csv(f"{file_name_no_ext}_2.csv", index=False)

print("dataset split successfully.")
