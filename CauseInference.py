import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import LabelEncoder as le
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from sklearn.preprocessing import StandardScaler
from sympy.codegen.cnodes import struct

df = pd.read_csv('data.csv')

drop_col = ['Date','Range']
data = df.drop(columns=drop_col)
scale_mapping = {
    "优": 1,
    "良": 2,
    "轻度污染": 3,
    "中度污染": 4,
    "重度污染": 5
}
data['Scale'] = data['Scale'].map(scale_mapping)
# 缺失值处理
median_cols = ["Sunshine duration(h)", "Rain Days(d)"] # 中位数填充
data[median_cols] = data[median_cols].fillna(data[median_cols].median())
numeric_cols = ["Thermal power generation (billion kilowatt-hours)",
                "Industrial enterprises value","Regional GDP (100 million yuan)",
                "Regional GDP (100 million yuan)",
                "Gross domestic product (100 million yuan)",
                "Per capita consumption expenditure (yuan)",
                "Passenger transport volume (10,000 person-times)"] # 均值填充
# data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())
data["Consumer price index"].fillna(method="bfill", inplace=True)  # 用后一个非空值填充
data["Air pollution control equipment (units (sets))"].fillna(method="ffill", inplace=True)  # 用前一个非空值填充
# numeric_cols.append("Air pollution control equipment (units (sets))")
# numeric_cols.append("Consumer price index")
# scaler = StandardScaler()
# data[numeric_cols] = scaler.fit_transform(data[numeric_cols]) # 标准化处理

missing_ratio = data.isnull().mean() * 100
print("缺失值比例：\n", missing_ratio)

struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)
print(struct_data.head(5))
struct_data=data.dropna()

sm = from_pandas(
    struct_data,
    max_iter=500,
)

sm.remove_edges_below_threshold(0.8)
print(sm.edges())
viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
viz.show('supporting_files/0.html')

sm_copy = sm.copy()
edges_related_to_PM25 = [
    (parent, target)
    for parent, target in sm_copy.edges if 'PM2.5' in (parent, target)
]
print(edges_related_to_PM25)
edges_to_remove = [
    (parent, target)
    for parent, target in sm_copy.edges
    if (parent, target) not in edges_related_to_PM25
]
# 删除不相关的边
for parent, target in edges_to_remove:
    sm_copy.remove_edge(parent, target)

sm_copy = sm_copy.get_largest_subgraph()
viz_deleted = plot_structure(
    sm_copy,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
viz_deleted.show('supporting_files/1.html')

import graphviz
from causalnex.network import BayesianNetwork
import networkx as nx

# 转换为DOT格式
bn = BayesianNetwork(sm)
G = bn
dot_data = nx.nx_pydot.to_pydot(G)

# 4. 自定义样式
dot = graphviz.Source(
    dot_data.to_string(),
    engine="dot",
    format="pdf",
    graph_attr={
        "rankdir": "LR",  # 从左到右布局
        "fontname": "Arial",
        "bgcolor": "transparent"
    },
    node_attr={
        "shape": "ellipse",
        "style": "filled",
        "fillcolor": "#E1F5FE",
        "fontname": "Arial"
    },
    edge_attr={
        "arrowsize": "0.8",
        "penwidth": "1.5"
    }
)

# 5. 突出显示关键节点
for node in dot_data.get_nodes():
    if node.get_name() == '"PM2.5"':
        node.set_fillcolor("#FF6B6B")

# 6. 保存
dot.render("paper_causal_graph", cleanup=True)