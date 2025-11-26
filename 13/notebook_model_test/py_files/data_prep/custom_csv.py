import pandas as pd

df = pd.read_csv("buckley_pde_data.csv")

df['x'] = df["x"].mul(3.0)
df['t'] = df["t"].mul(3.0)

df.to_csv("test1.csv")

