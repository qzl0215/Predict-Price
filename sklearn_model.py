import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from pandas.tseries.holiday import USFederalHolidayCalendar

'''
below is a dummy but efficient way of creating holiday mapping,
with federal holidays as index and holiday names as elements
'''
holidays_list = USFederalHolidayCalendar().holidays(start='2015-07-03', end='2017-12-31')
# get federal holiday date list

holidays_index = pd.to_datetime(holidays_list)  # convert string to date type
holiday_mapping = pd.DataFrame(index=holidays_index)
holiday_mapping['holiday_name'] = ['H_Independence', 'H_Labor', 'H_Columbus', 'H_Veterans', 'H_Thanksgiving',
                                   'H_Christmas', 'H_NewYear', 'H_Martin', 'H_President', 'H_Memorial',
                                   'H_Independence', 'H_Labor', 'H_Columbus', 'H_Veterans', 'H_Thanksgiving',
                                   'H_Christmas', 'H_NewYear', 'H_Martin', 'H_President', 'H_Memorial',
                                   'H_Independence', 'H_Labor', 'H_Columbus', 'H_Veterans', 'H_Thanksgiving',
                                   'H_Christmas']


'''
this method converts classification variables to dummy variables
for example, validating carriers are UA, AA, ...,  new features will be UA, AA, ... with 0/1 as value
parameter df is the dataframe
parameter ar is the classification array
prefix is the prefix to be added before the new feature
'''
def create_new_features(df, ar, prefix):
    for feature in ar.unique():
        if feature == '':  # this avoid creating empty feature
            continue  # if the feature is empty, skip the rest part
        new_str = prefix + '_' + str(feature)
        df[new_str] = 0  # initiate default value as 0 for all elements
        for i in range(0, len(ar)):
            if feature == ar[i]:
                df.set_value(i, new_str, 1)


'''
this method help get the closest holiday given a date string
'''
def get_closest_holiday_date(d):
    closest_holiday = holidays_list[holidays_list.get_loc(d, method='nearest')]
    return closest_holiday


'''
this method help get the closest Holiday Name in string given a date string
for example, if d = '1/5/2017', the return will be H_NewYear
'''
def get_closest_holiday_str(d):
    closest_holiday = get_closest_holiday_date(d)
    return holiday_mapping.ix[closest_holiday.strftime('%Y-%m-%d')][0]


'''
this method calculates the number of days to closest holiday
if d1 passed the closest holiday, the output will be negative
for example, if d1 = 1/2/2016, return will be -1 because it passed 1/1/2016 (New Year)
'''
def days_to_closest_holiday(d1):
        d2 = get_closest_holiday_date(d1)
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        return (d2 - d1).days

ori_data = pd.read_csv('boscun.csv')  # read original data

# add weekday feature for both departure date and return date
ori_data['departure_weekday'] = ori_data.departure_odate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%A'))
ori_data['return_weekday'] = ori_data.return_odate.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%A'))

# add holiday
ori_data['departure_holiday'] = ori_data.departure_odate.apply(get_closest_holiday_str)
ori_data['return_holiday'] = ori_data.return_odate.apply(get_closest_holiday_str)

# convert micro seconds to time
ori_data['departure_time'] = ori_data.departure_ms.apply(lambda x: datetime.fromtimestamp(x / 1000.0))
ori_data['return_time'] = ori_data.return_ms.apply(lambda x: datetime.fromtimestamp(x / 1000.0))

'''
X is the input matrix, all values must be numerical
'''
X = pd.DataFrame()  # initiate empty matrix X
# number of days from booking date to departure date
X['advance'] = preprocessing.scale(ori_data.advance.values)
# number of normalized available seats
X['available_seats'] = preprocessing.scale(ori_data.available_seats.values)

# the year of the date, reflects inflation etc.
X['departure_year'] = preprocessing.scale(ori_data.departure_odate.apply(lambda x: int(x[0:4])).values)
X['return_year'] = preprocessing.scale(ori_data.return_odate.apply(lambda x: int(x[0:4])).values)

# number of days to closest holiday, can be negative
X['departure_days_to_closest_holiday'] = ori_data.departure_odate.apply(days_to_closest_holiday)
X['return_days_to_closest_holiday'] = ori_data.return_odate.apply(days_to_closest_holiday)

# normalized duration
X['departure_duration'] = preprocessing.scale(ori_data.outgoing_duration.values)
X['return_duration'] = preprocessing.scale(ori_data.outgoing_duration.values)

# number of stops
X['departure_stops'] = ori_data.outgoing_stops
X['return_stops'] = ori_data.outgoing_stops

'''
next loop is to convert time to value, steps as below:
First, deparuture_time contains the info of date and time, departure_date only contains the info of date
Second, convert both to seconds and substract, only time in seconds left
Third, normalize the seconds
Fourth, do the same for both departure time and return time
'''
for i in range(0, len(X.departure_year)):
    departure_date = ori_data.get_value(i, 'departure_odate')  # typeof string
    departure_time = ori_data.get_value(i, 'departure_time')
    departure_time_in_seconds = (departure_time - datetime.strptime(departure_date, '%Y-%m-%d')).seconds
    X.set_value(i, 'departure_time', departure_time_in_seconds)

    return_date = ori_data.get_value(i, 'return_odate')  # typeof string
    return_time = ori_data.get_value(i, 'return_time')
    return_time_in_seconds = (return_time - datetime.strptime(return_date, '%Y-%m-%d')).seconds
    X.set_value(i, 'return_time', return_time_in_seconds)
X['departure_time'] = preprocessing.scale(X['departure_time'])
X['return_time'] = preprocessing.scale(X['return_time'])

# below convert category features to logistic variables
create_new_features(X, ori_data.major_carrier_id, 'major_carrier')  # major carrier (UA, AA, DL,...)
create_new_features(X, ori_data.validating_carrier, 'validating_carrier')  # validating carrier (AA, AC, AM,...)
create_new_features(X, ori_data.lowest_cabin_class, 'lowest_class')  # lowest class (E, B)
create_new_features(X, ori_data.highest_cabin_class, 'highest_class')  # highest class (E, F, B, EP, U)
create_new_features(X, ori_data.departure_weekday, 'departure_weekday')  # weekday of departure date (Monday,...)
create_new_features(X, ori_data.return_weekday, 'return_weekday')  # weekday of return date (Monday,...)
create_new_features(X, ori_data.departure_holiday, 'departure')  # closest holiday to departure date (Christmas,...)
create_new_features(X, ori_data.return_holiday, 'return')  # closest holiday to return date (Christmas,...)

# save the converted input X into csv file
X.to_csv('boscun_X.csv', index=False)
# save the output y into csv file
ori_data['total_usd'].to_csv('boscun_y.csv', index=False)