import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# importing trackman and blast data
df = pd.read_csv("blastonly.csv")

# normalizing variables
scaler = MinMaxScaler()
df['AttackAngle mean'] = scaler.fit_transform(df[["AttackAngle mean"]])
df['AttackAngle std'] = scaler.fit_transform(df[["AttackAngle std"]])
df['BatSpeed mean'] = scaler.fit_transform(df[["BatSpeed mean"]])
df['ConnectionMean mean'] = scaler.fit_transform(df[["ConnectionMean mean"]])
df['ConnectionMean std'] = scaler.fit_transform(df[["ConnectionMean std"]])
df['PlanarEfficiency mean'] = scaler.fit_transform(df[["PlanarEfficiency mean"]])
df['PlanarEfficiency std'] = scaler.fit_transform(df[["PlanarEfficiency std"]])
df['RotationalAcceleration mean'] = scaler.fit_transform(df[["RotationalAcceleration mean"]])
# df['ExitSpeed max'] = scaler.fit_transform(df[["ExitSpeed max"]])
# df['ExitSpeed mean'] = scaler.fit_transform(df[["ExitSpeed mean"]])
# df['ExitSpeed std'] = scaler.fit_transform(df[["ExitSpeed std"]])
# df['VertAngle max'] = scaler.fit_transform(df[["VertAngle max"]])
# df['VertAngle mean'] = scaler.fit_transform(df[["VertAngle mean"]])
# df['VertAngle std'] = scaler.fit_transform(df[["VertAngle std"]])

# K Means Clustering
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=10, max_iter=300, random_state=26)
kmeans = kmeans.fit(df[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean',
                        'ConnectionMean std', 'PlanarEfficiency mean', 'PlanarEfficiency std',
                        'RotationalAcceleration mean']])
df.loc[:, 'labels'] = kmeans.labels_

df2 = pd.read_csv("blastonly.csv")
df2 = pd.merge(df2, df[['BatterId', 'labels']], on='BatterId', how='left')
df2.to_csv('kmeans_results.csv', index=False)
summary_table = df2.groupby('labels').agg(['mean'])
summary_table.columns = summary_table.columns.droplevel(1)
summary_table = summary_table.round(decimals=3)
summary_table.rename(index={0: 'Group 0', 1: 'Group 1', 2: 'Group 2', 3: 'Group 3', 4: 'Group 4'}, inplace=True)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=summary_table.values, colLabels=summary_table.columns, rowLabels=summary_table.index,
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
plt.show()
