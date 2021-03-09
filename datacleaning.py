import pandas as pd
from pandas import DataFrame
import numpy as np

# reading in the raw data
# df = pd.read_csv("trackman_data.csv")
df2 = pd.read_csv("blast_data.csv")

# combining connection and early connection metrics
df2['ConnectionMean'] = df2[['Connection', 'EarlyConnection']].mean(axis=1)

# aggregating blast motion data into metrics
blast_motion_players = df2.groupby('BatterId').agg(
    {
        'AttackAngle': ["mean", "std"],
        'BatSpeed': "mean",
        'ConnectionMean': ["mean", "std"],
        "PlanarEfficiency": ["mean", "std"],
        "RotationalAcceleration": "mean"
    }
)

# # aggregating trackman data into metrics
# trackman_players = df.groupby('BatterId').agg(
#     {
#         'ExitSpeed': ["max", "mean", "std"],
#         'VertAngle': ["max", "mean", "std"],
#     }
# )

# combining trackman and blast motion metrics by player
# full_data = blast_motion_players.join(trackman_players, how='outer')

# removing multilevel columns
blast_motion_players.columns = blast_motion_players.columns.map(' '.join).str.strip(' ')

# splitting the data into players that have trackman/blast motion data and players that have only blast motion
# track_blast = full_data[full_data['ExitSpeed mean'].notnull()]
# blast_only = full_data[full_data['ExitSpeed mean'].isnull()]

# converting BatterId from index to column
blast_motion_players.reset_index(inplace=True)
blast_motion_players = blast_motion_players.rename(columns={'index': 'BatterId'})
# blast_only.reset_index(inplace=True)
# blast_only = blast_only.rename(columns={'index': 'BatterId'})


# converting blast motion columns from radians to degrees
def rad_to_deg(ra):
    return ra * (180 / np.pi)


blast_motion_players['AttackAngle mean'] = rad_to_deg(blast_motion_players['AttackAngle mean'])
blast_motion_players['AttackAngle std'] = rad_to_deg(blast_motion_players['AttackAngle std'])
blast_motion_players['ConnectionMean mean'] = rad_to_deg(blast_motion_players['ConnectionMean mean'])
blast_motion_players['ConnectionMean std'] = rad_to_deg(blast_motion_players['ConnectionMean std'])
# blast_only['AttackAngle mean'] = rad_to_deg(blast_only['AttackAngle mean'])
# blast_only['AttackAngle std'] = rad_to_deg(blast_only['AttackAngle std'])
# blast_only['ConnectionMean mean'] = rad_to_deg(blast_only['ConnectionMean mean'])
# blast_only['ConnectionMean std'] = rad_to_deg(blast_only['ConnectionMean std'])

# dropping trackman columns from blast only data
# blast_only = blast_only.drop(['ExitSpeed max', 'ExitSpeed mean', 'ExitSpeed std', 'VertAngle max', 'VertAngle mean',
#                               'VertAngle std'], axis=1)

blast_motion_players.to_csv('blastonly.csv', index=False)
# blast_only.to_csv('blastonly.csv', index=False)
