import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from sklearn.preprocessing import StandardScaler
from sympy.codegen.cnodes import struct

df = pd.read_csv('data_daily.csv')

# drop_col = ['Date','AQI','topTemp','bottomTemp','isRainy','windScale']
drop_col = ['Date','AQI']
data = df.drop(columns=drop_col)
data.info()

data = data.dropna(axis='index',how='any')
missing_ratio = data.isnull().mean() * 100
print("缺失值比例：\n", missing_ratio)

struct_data = data.copy()
numeric_cols = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
struct_data[numeric_cols] = struct_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
print(struct_data.head(5))
print("NaN数量:\n", struct_data.isnull().sum())
struct_data = struct_data.dropna()

sm = from_pandas(
    struct_data,
    max_iter=500,
)

sm.remove_edges_below_threshold(0.8)
# sm.add_edge('PM10','PM2.5') # 化学证据：PM10是经过变化形成PM2.5的物质之一
# sm.add_edge('SO2','PM2.5') # 多项研究证据，相关性高
# sm.remove_edge('SO2','O3') # 相关性弱，无相关证据表明因果关系
# # 其他无证明的边，根据Pearson相关系数较高，不再删除

sm.remove_edge('bottomTemp', 'topTemp')
sm.remove_edge('isRainy', 'topTemp')
sm.remove_edge('isRainy', 'bottomTemp')
sm.remove_edge('windScale', 'topTemp')
sm.remove_edge('windScale', 'bottomTemp')
sm.remove_edge('isRainy', 'O3')
sm.remove_edge('isRainy', 'CO')
sm.remove_edge('windScale', 'O3')
sm.remove_edge('windScale', 'SO2')
sm.remove_edge('topTemp', 'NO2')

print(sm.edges())
viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
viz.show('supporting_files/2.html')

# sm_copy = sm.copy()
# edges_related_to_PM25 = [
#     (parent, target)
#     for parent, target in sm_copy.edges if 'PM2.5' in (parent, target)
# ]
# print(edges_related_to_PM25)
# edges_to_remove = [
#     (parent, target)
#     for parent, target in sm_copy.edges
#     if (parent, target) not in edges_related_to_PM25
# ]
# # 删除不相关的边
# for parent, target in edges_to_remove:
#     sm_copy.remove_edge(parent, target)
#
# sm_copy = sm_copy.get_largest_subgraph()
# viz_deleted = plot_structure(
#     sm_copy,
#     all_node_attributes=NODE_STYLE.WEAK,
#     all_edge_attributes=EDGE_STYLE.WEAK,
# )
# viz_deleted.show('supporting_files/3.html')

#
# from causalnex.network import BayesianNetwork
# bn = BayesianNetwork(sm)
#
#
# discretised_data = struct_data.copy()
#
# data_vals = {col: struct_data[col].unique() for col in struct_data.columns}
#
# PM25_map = {v: 1 if v <= 75
#                 else 0 for v in data_vals['PM2.5']}
# PM10_map = {v: 1 if v <= 50
#                 else 0 for v in data_vals['PM10']}
# O3_map = {v: 1 if v <= 100
#                 else 0 for v in data_vals['O3']}
# NO2_map = {v: 1 if v <= 40
#                 else 0 for v in data_vals['NO2']}
# SO2_map = {v: 1 if v <= 50
#                 else 0 for v in data_vals['SO2']}
# CO_map = {v: 1 if v <= 4
#                 else 0 for v in data_vals['CO']}
#
# # 'O3', 'NO2', 'SO2', 'CO'
#
# discretised_data['PM2.5'] = discretised_data['PM2.5'].map(PM25_map)
# discretised_data['PM10'] = discretised_data['PM10'].map(PM10_map)
# discretised_data['O3'] = discretised_data['O3'].map(O3_map)
# discretised_data['NO2'] = discretised_data['NO2'].map(NO2_map)
# discretised_data['SO2'] = discretised_data['SO2'].map(SO2_map)
# discretised_data['CO'] = discretised_data['CO'].map(CO_map)
#
# from sklearn.model_selection import train_test_split
#
# train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1, random_state=7)
#
# bn = bn.fit_node_states(discretised_data)
#
# bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
#
# predictions = bn.predict(discretised_data, 'PM2.5')
#
#
# from causalnex.evaluation import classification_report
#
# classification_report(bn, test, 'PM2.5')
#
# from causalnex.evaluation import roc_auc
# roc, auc = roc_auc(bn, test, 'PM2.5')
# print('AUC:',auc)
#
