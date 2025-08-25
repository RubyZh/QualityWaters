import pandas as pd
import numpy as np
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import LabelEncoder as le
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork
from causalnex.discretiser import Discretiser

df = pd.read_csv('data.csv')

drop_col = ['Date','Range','Scale']
data = df.drop(columns=drop_col)


struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)

for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])
print(struct_data.head(5))
struct_data=data.dropna()
struct_data.fillna(0, inplace=True)

sm = from_pandas(struct_data)
sm.remove_edges_below_threshold(0.8)

bn = BayesianNetwork(sm)

discretised_data = data.copy()
data_vals = {col: data[col].unique() for col in data.columns}

failures_map = {v: 'no-failure' if v == [0]
                else 'have-failure' for v in data_vals['failures']}
studytime_map = {v: 'short-studytime' if v in [1,2]
                 else 'long-studytime' for v in data_vals['studytime']}
discretised_data["failures"] = discretised_data["failures"].map(failures_map)
discretised_data["studytime"] = discretised_data["studytime"].map(studytime_map)


discretised_data["absences"] = Discretiser(method="fixed",
                          numeric_split_points=[1, 10]).transform(discretised_data["absences"].values)
discretised_data["G1"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G1"].values)
discretised_data["G2"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G2"].values)
discretised_data["G3"] = Discretiser(method="fixed",
                          numeric_split_points=[10]).transform(discretised_data["G3"].values)
