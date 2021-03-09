import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


# creating elbow graph
kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 26, }
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean', 'ConnectionMean std',
                   'PlanarEfficiency mean', 'PlanarEfficiency std', 'RotationalAcceleration mean']])
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("SSE", fontsize=12)
plt.title("Elbow Graph", fontsize=16)
# plt.savefig('elbow.png')
plt.show()

# automatic elbow locator
kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(kl.elbow)

# creating silhouette score graph
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean', 'ConnectionMean std',
                   'PlanarEfficiency mean', 'PlanarEfficiency std', 'RotationalAcceleration mean']])
    score = silhouette_score(df[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean',
                                 'ConnectionMean std', 'PlanarEfficiency mean', 'PlanarEfficiency std',
                                 'RotationalAcceleration mean']], kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("Silhouette Coefficient", fontsize=12)
plt.title("Silhouette Scores", fontsize=16)
# plt.savefig('d_silhouette.png')
plt.show()

# # importing blast data only
# df2 = pd.read_csv("blastonly.csv")
#
# # normalizing variables
# scaler = MinMaxScaler()
# df2['AttackAngle mean'] = scaler.fit_transform(df2[["AttackAngle mean"]])
# df2['AttackAngle std'] = scaler.fit_transform(df2[["AttackAngle std"]])
# df2['BatSpeed mean'] = scaler.fit_transform(df2[["BatSpeed mean"]])
# df2['ConnectionMean mean'] = scaler.fit_transform(df2[["ConnectionMean mean"]])
# df2['ConnectionMean std'] = scaler.fit_transform(df2[["ConnectionMean std"]])
# df2['PlanarEfficiency mean'] = scaler.fit_transform(df2[["PlanarEfficiency mean"]])
# df2['PlanarEfficiency std'] = scaler.fit_transform(df2[["PlanarEfficiency std"]])
# df2['RotationalAcceleration mean'] = scaler.fit_transform(df2[["RotationalAcceleration mean"]])
#
#
# # creating elbow graph
# kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 26, }
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(df2[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean', 'ConnectionMean std',
#                     'PlanarEfficiency mean', 'PlanarEfficiency std', 'RotationalAcceleration mean']])
#     sse.append(kmeans.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters", fontsize=12)
# plt.ylabel("SSE", fontsize=12)
# plt.title("Elbow Graph", fontsize=16)
# # plt.savefig('elbow.png')
# plt.show()
#
#
# # automatic elbow locator
# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# print(kl.elbow)
#
#
# # creating silhouette score graph
# silhouette_coefficients = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(df2[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean', 'ConnectionMean std',
#                     'PlanarEfficiency mean', 'PlanarEfficiency std', 'RotationalAcceleration mean']])
#     score = silhouette_score(df2[['AttackAngle mean', 'AttackAngle std', 'BatSpeed mean', 'ConnectionMean mean',
#                                   'ConnectionMean std', 'PlanarEfficiency mean', 'PlanarEfficiency std',
#                                   'RotationalAcceleration mean']], kmeans.labels_)
#     silhouette_coefficients.append(score)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters", fontsize=12)
# plt.ylabel("Silhouette Coefficient", fontsize=12)
# plt.title("Silhouette Scores", fontsize=16)
# # plt.savefig('d_silhouette.png')
# plt.show()
