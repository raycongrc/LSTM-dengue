from datetime import datetime
from pandas import read_csv

def parse(x):
    '''
	a helper function to parse the time format in the .csv file
	'''
    return datetime.strptime(x, '%Y-%m-%d')


# read the .csv file into pandas.DataFrame object
# dataset = read_csv('Iquitos_Training_Data.csv',  parse_dates = ['week_start_date'], index_col=2,date_parser=parse)
dataset = read_csv('San_Juan_Training_Data.csv', parse_dates=['week_start_date'], index_col=2, date_parser=parse)
# drop the column 'season' and 'season_week'.
# Keeping the column 'week_start_date' is enough to distinguish the weeks
dataset.drop(['season', 'season_week'], axis=1, inplace=True)

# move the last column 'total_cases' to the first
cols = dataset.columns.tolist()
cols = cols[-1:] + cols[:-1]
dataset = dataset[cols]

# save the pandas.DataFrame object to a new .csv file
dataset.to_csv('train_data.csv')
print(dataset.keys)
print(dataset.shape)