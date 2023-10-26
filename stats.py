import pandas as pd
from sklearn import linear_model


def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value




# Import the average on-field % CSV
dfs_trend = []
for files in range(1, 32):
    data = pd.read_csv(f'trends\\new_name{files}.csv', skiprows=1)
    dfile = pd.DataFrame(data)
    dfs_trend.append(dfile)
trend_df = pd.concat(dfs_trend, ignore_index=True)
trend_df = trend_df[['Player', 'Avg']]
trend_df.rename(columns={'Player': 'Nickname'}, inplace=True)
trend_df.rename(columns={'Avg': 'AvgOnFieldPercent'}, inplace=True)


# Import the targets CSV
target_data = pd.read_csv(f'targets\\WRTERB-targets.csv', skiprows=1)
target_result = pd.DataFrame(target_data)
target_result = target_result[['Name', 'TAR']]
target_result.rename(columns={'Name': 'Nickname'}, inplace=True)


# Import passing CSV
passing_data = pd.read_csv("player_stats\\passing-basic-stats.csv", skiprows=1)
passing_result = pd.DataFrame(passing_data)
passing_result = passing_result[['Name', 'COMP%', 'YPA', 'TD']]
passing_result.rename(columns={'Name': 'Nickname', 'COMP%': 'Pass_COMP%', 'YPA': 'Pass_YPA', 'TD': 'Pass_TD'}, inplace=True)


# Import receiving CSV
receiving_data = pd.read_csv("player_stats\\receiving-basic-stats.csv", skiprows=1)
receiving_result = pd.DataFrame(receiving_data)
receiving_result = receiving_result[['Name', 'YPC', 'YPT', 'TD']]
receiving_result.rename(columns={'Name': 'Nickname', 'YPC': 'Rec_YPC', 'YPT': 'Rec_YPT', 'TD': 'Rec_TD'}, inplace=True)


# Import rushing CSV
rushing_data = pd.read_csv("player_stats\\rushing-basic-stats.csv", skiprows=1)
rushing_result = pd.DataFrame(rushing_data)
rushing_result = rushing_result[['Name', 'ATT', 'AVG', 'TD']]
rushing_result.rename(columns={'Name': 'Nickname', 'ATT': 'Rush_ATT', 'AVG': 'Rush_AVG', 'TD': 'Rush_TD'}, inplace=True)


# Import data from game CSV's
data = pd.read_csv('player.csv')
df = pd.DataFrame(data)


# points per $1000 spent
df['FPPG_1000'] = df['FPPG'] / df['Salary'] * 1000


# Finds the Injured players; leaves in the QUESTIONABLE players to generate lists
df_dropd = df.dropna(subset=['FPPG_1000'])
df_dropd = df_dropd[(df_dropd['Injury Indicator'].isna()) | (df_dropd['Injury Indicator'] == 'Q')]


# Gets the names of all players hurt; including QUESTIONABLE
non_df_dropd = df.dropna(subset=['Injury Indicator'])
non_df_dropd = non_df_dropd[['Nickname', 'Injury Indicator', 'Injury Details']]


# Sort the DataFrames
#df_sort_fppg = df_dropd.sort_values(by='FPPG', ascending=False)
df_sort_ratio = df_dropd.sort_values(by='FPPG_1000', ascending=False)


# player.csv 
trend_result = df_sort_ratio[['Id', 'Nickname', 'Position', 'Team' , 'FPPG', 'FPPG_1000', 'Salary', 'Injury Indicator','Injury Details']]


# Use reduce from functool lib to merge df's
dfs = [trend_result, trend_df, target_result, passing_result, receiving_result, rushing_result]
final_result = reduce(lambda left, right: pd.merge(left, right, how='left', on='Nickname'), dfs)
final_result.to_csv('final_stat.csv')


# SK-Learn
# Need to split by Pos.
final_result.fillna(0.0, inplace=True)
X = final_result[['AvgOnFieldPercent', 'TAR', 'Pass_COMP%', 'Pass_YPA', 'Pass_TD', 'Rec_YPC', 'Rec_YPT', 'Rec_TD', 'Rush_ATT', 'Rush_AVG', 'Rush_TD']]
y = final_result[['FPPG', 'Salary']]


regr = linear_model.LinearRegression()
regr.fit(X, y)


a_list = ['AvgOnFieldPercent', 'TAR', 'Pass_COMP%', 'Pass_YPA', 'Pass_TD', 'Rec_YPC', 'Rec_YPT', 'Rec_TD', 'Rush_ATT', 'Rush_AVG', 'Rush_TD']
final_coef = pd.DataFrame({'AvgOnFieldPercent': [regr.coef_[0][0], regr.coef_[1][0]], 'TAR': [regr.coef_[0][1], regr.coef_[1][1]], 'Pass_COMP%': [regr.coef_[0][2], regr.coef_[1][2]], 'Pass_YPA': [regr.coef_[0][3], regr.coef_[1][3]], 'Pass_TD': [regr.coef_[0][4], regr.coef_[1][4]], 'Rec_YPC': [regr.coef_[0][5], regr.coef_[1][5]], 'Rec_YPT': [regr.coef_[0][6], regr.coef_[1][6]], 'Rec_TD': [regr.coef_[0][7], regr.coef_[1][7]], 'Rush_ATT': [regr.coef_[0][8], regr.coef_[1][8]], 'Rush_AVG': [regr.coef_[0][9], regr.coef_[1][9]], 'Rush_TD': [regr.coef_[0][10], regr.coef_[1][10]]})
final_coef.to_csv('final_coef.csv')
