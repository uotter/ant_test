import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import src.iolib as il
import math as math


def get_date_features(holiday_df):
    '''
        生成日期特征
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    weekday_number = 0  # 工作日的第几天
    holiday_number = 0  # 休息日的第几天
    total_weekday = 0  # 当前日期所属的连续工作日区间总日数
    total_holiday = 0  # 当前日期所属的连续休息日区间总日数
    loc_in_month = 0  # 一月中的上中下旬
    lastday_before_holiday = 0  # 是否是休假前上班的最后一天
    firstday_after_holiday = 0  # 是否是休假后上班的第一天
    month = 0  # 月份
    is_holiday_yesterday = 1
    date_feature_df = pd.DataFrame(
        columns=["date", "weekday_number", "holiday_number", "total_weekday", "total_holiday", "loc_in_month",
                 "lastday_before_holiday", "firstday_after_holiday", "month", "is_holiday"])
    for index, row in holiday_df.iterrows():
        insert_dic = {}
        date_str = row["datestr"]
        is_holiday = row["weekday"]
        date_time = strptime(date_str, "%Y-%m-%d")
        month = date_time.month
        day = date_time.day
        if date_time < strptime("2017-07-29", "%Y-%m-%d"):
            if day <= 10:
                loc_in_month = 0
            elif day <= 20:
                loc_in_month = 1
            else:
                loc_in_month = 2
            if is_holiday == 1 and is_holiday_yesterday == 1:
                weekday_number = 0
                holiday_number += 1
                firstday_after_holiday = 0
            elif is_holiday == 1 and is_holiday_yesterday == 0:
                weekday_number = 0
                holiday_number = 0
                holiday_number += 1
                firstday_after_holiday = 0
            elif is_holiday == 0 and is_holiday_yesterday == 1:
                holiday_number = 0
                weekday_number = 0
                weekday_number += 1
                firstday_after_holiday = 1
            elif is_holiday == 0 and is_holiday_yesterday == 0:
                holiday_number = 0
                weekday_number += 1
                firstday_after_holiday = 0
            else:
                print("Holiday count wrong.")
            date_time_tomorrow = date_time + datetime.timedelta(1)
            date_str_tomorrow = strftime(date_time_tomorrow, "%Y-%m-%d")
            is_holiday_tomorrow = holiday_df[holiday_df["datestr"] == date_str_tomorrow]["weekday"].values.tolist()[0]
            if is_holiday == 0 and is_holiday_tomorrow == 1:
                lastday_before_holiday = 1
            else:
                lastday_before_holiday = 0
            if is_holiday == 0:
                total_holiday = 0
                is_holiday_iter = 0
                date_time_iter = date_time
                weekdays_after = 0
                weekdays_before = 0
                while is_holiday_iter == 0 and date_time_iter < strptime("2017-07-29", "%Y-%m-%d"):
                    date_time_iter = date_time_iter + datetime.timedelta(1)
                    date_str_iter = strftime(date_time_iter, "%Y-%m-%d")
                    is_holiday_iter = holiday_df[holiday_df["datestr"] == date_str_iter]["weekday"].values.tolist()[0]
                    weekdays_after += 1
                is_holiday_iter = 0
                date_time_iter = date_time
                while is_holiday_iter == 0 and date_time_iter > strptime("2017-01-01", "%Y-%m-%d"):
                    date_time_iter = date_time_iter - datetime.timedelta(1)
                    date_str_iter = strftime(date_time_iter, "%Y-%m-%d")
                    is_holiday_iter = holiday_df[holiday_df["datestr"] == date_str_iter]["weekday"].values.tolist()[0]
                    weekdays_before += 1
                total_weekday = weekdays_after + weekdays_before - 1
            else:
                total_weekday = 0
                is_holiday_iter = 1
                date_time_iter = date_time
                weekdays_after = 0
                weekdays_before = 0
                while is_holiday_iter == 1 and date_time_iter < strptime("2017-07-29", "%Y-%m-%d"):
                    date_time_iter = date_time_iter + datetime.timedelta(1)
                    date_str_iter = strftime(date_time_iter, "%Y-%m-%d")
                    is_holiday_iter = holiday_df[holiday_df["datestr"] == date_str_iter]["weekday"].values.tolist()[0]
                    weekdays_after += 1
                is_holiday_iter = 1
                date_time_iter = date_time
                while is_holiday_iter == 1 and date_time_iter > strptime("2017-01-01", "%Y-%m-%d"):
                    date_time_iter = date_time_iter - datetime.timedelta(1)
                    date_str_iter = strftime(date_time_iter, "%Y-%m-%d")
                    is_holiday_iter = holiday_df[holiday_df["datestr"] == date_str_iter]["weekday"].values.tolist()[0]
                    weekdays_before += 1
                total_holiday = weekdays_after + weekdays_before - 1
            insert_dic["date"] = date_str
            insert_dic["weekday_number"] = weekday_number
            insert_dic["holiday_number"] = holiday_number
            insert_dic["total_weekday"] = total_weekday
            insert_dic["total_holiday"] = total_holiday
            insert_dic["loc_in_month"] = loc_in_month
            insert_dic["lastday_before_holiday"] = lastday_before_holiday
            insert_dic["firstday_after_holiday"] = firstday_after_holiday
            insert_dic["month"] = month
            insert_dic["is_holiday"] = is_holiday
            date_feature_df = date_feature_df.append(insert_dic, ignore_index=True)
            is_holiday_yesterday = is_holiday
    date_feature_df["date"] = pd.to_datetime(date_feature_df["date"])
    date_feature_df = date_feature_df.set_index("date")
    date_feature_df = date_feature_df.sort_index()
    return date_feature_df


def get_market_features():
    '''
        生成市场特征
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    caihui_index_df = il.get_caihui_index_net_matrix("2017-01-01", "2017-07-29", fill=False)
    shibor_index_df = il.get_shibor_net_matrix("2017-01-01", "2017-07-29", fill=False)
    wind_index_df = il.get_wind_index_net_matrix("2017-01-01", "2017-07-29", fill=False)
    index_df = pd.concat([caihui_index_df, shibor_index_df, wind_index_df], axis=1)
    mix_df = pd.concat([day_data, index_df], axis=1)
    mix_df = mix_df.fillna(method="pad")
    mix_df = mix_df.fillna(method="bfill")
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    before_days = 5
    market_feature_df = pd.DataFrame()
    mix_df_describe = mix_df.describe()
    save_mean = mix_df_describe.loc["mean", "save"]
    max_count_date = 1
    min_count_date = 1
    max_count_ratio = 0.5
    min_count_ratio = -0.4
    for index, row in mix_df.iterrows():
        date_time = index
        date_str = index._date_repr
        save = row["save"]
        shibor_2w = row["2W"]
        hs300 = row["HS300"]
        zh500 = row["中证500"]
        cyb = row["创业板综"]
        msci = row["MSCI新兴市场"]
        market_dic = {}
        # market_dic["min_count_date"] = min_count_date
        # market_dic["max_count_date"] = max_count_date
        min_count_date += 1
        max_count_date += 1
        if math.isnan(save):
            pass
        else:
            save_change_ratio = (save - save_mean) / save_mean
            if save_change_ratio > max_count_ratio:
                max_count_date = 1
            if save_change_ratio < min_count_ratio:
                min_count_date = 1
        for day_change in range(before_days):
            day_change += 1
            before_start_day = date_time - datetime.timedelta(day_change)
            if before_start_day < strptime("2017-01-01", "%Y-%m-%d"):
                # market_dic["save"+"_"+str(day_change)] = 0
                market_dic["shibor_2w" + "_" + str(day_change)] = 0
                market_dic["hs300" + "_" + str(day_change)] = 0
                market_dic["zh500" + "_" + str(day_change)] = 0
                market_dic["cyb" + "_" + str(day_change)] = 0
                market_dic["msci" + "_" + str(day_change)] = 0
                # market_dic["sp500" + "_" + str(day_change)] = 0
            else:
                # market_dic["save" + "_" + str(day_change)] = mix_df.loc[before_start_day,"save"]
                market_dic["shibor_2w" + "_" + str(day_change)] = mix_df.loc[before_start_day, "2W"]
                market_dic["hs300" + "_" + str(day_change)] = mix_df.loc[before_start_day, "HS300"]
                market_dic["zh500" + "_" + str(day_change)] = mix_df.loc[before_start_day, "中证500"]
                market_dic["cyb" + "_" + str(day_change)] = mix_df.loc[before_start_day, "创业板综"]
                market_dic["msci" + "_" + str(day_change)] = mix_df.loc[before_start_day, "MSCI新兴市场"]
                # market_dic["sp500" + "_" + str(day_change)] = mix_df.loc[before_start_day, "S&P 500"]
        market_dic["date"] = date_time
        market_feature_df = market_feature_df.append(market_dic, ignore_index=True)
    market_feature_df["date"] = pd.to_datetime(market_feature_df["date"])
    market_feature_df = market_feature_df.set_index("date")
    market_feature_df = market_feature_df.sort_index()
    return market_feature_df


def generate_dataset(date_feature_df, day_data):
    '''
        组合X和Y，生成数据集
    '''
    day_data_temp = day_data.copy()
    date_feature_df_norm = (date_feature_df - date_feature_df.min()) / (date_feature_df.max() - date_feature_df.min())
    date_feature_df_norm.insert(len(date_feature_df_norm.columns), "save", day_data_temp.pop("save"))
    return date_feature_df_norm


def generate_rnn_dataset(date_feature_df, day_data):
    '''
        组合X和Y，生成数据集
    '''
    day_data_temp = day_data.copy()
    day_data_temp = day_data_temp.shift(1)
    day_data_temp = day_data_temp.fillna(method="bfill")
    date_feature_df_norm = (date_feature_df - date_feature_df.min()) / (date_feature_df.max() - date_feature_df.min())
    date_feature_df_norm.insert(len(date_feature_df_norm.columns), "save", day_data_temp.pop("save"))
    return date_feature_df_norm


def get_combine_features(features_list):
    '''
        组合市场特征和时间特征
    '''
    combine_features = pd.concat(features_list, axis=1)
    return combine_features


def get_rnn_market_features():
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    caihui_index_df = il.get_caihui_index_net_matrix("2017-01-01", "2017-07-29", fill=False)
    shibor_index_df = il.get_shibor_net_matrix("2017-01-01", "2017-07-29", fill=False)
    wind_index_df = il.get_wind_index_net_matrix("2017-01-01", "2017-07-29", fill=False)
    index_df = pd.concat([caihui_index_df, shibor_index_df, wind_index_df], axis=1)
    mix_df = pd.concat([day_data, index_df], axis=1)
    mix_df = mix_df.fillna(method="pad")
    mix_df = mix_df.fillna(method="bfill")
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    market_feature_df = pd.DataFrame()
    for index, row in mix_df.iterrows():
        date_time = index
        date_str = index._date_repr
        save = row["save"]
        shibor_2w = row["2W"]
        hs300 = row["HS300"]
        zh500 = row["中证500"]
        cyb = row["创业板综"]
        msci = row["MSCI新兴市场"]
        market_dic = {}
        market_dic["shibor_2w"] = shibor_2w
        market_dic["hs300"] = hs300
        market_dic["zh500"] = zh500
        market_dic["cyb"] = cyb
        market_dic["msci"] = msci
        if math.isnan(save):
            market_dic["save_before"] = 0
        else:
            market_dic["save_before"] = save
        market_dic["date"] = date_time
        market_feature_df = market_feature_df.append(market_dic, ignore_index=True)
    market_feature_df["date"] = pd.to_datetime(market_feature_df["date"])
    market_feature_df = market_feature_df.set_index("date")
    market_feature_df = market_feature_df.sort_index()
    return market_feature_df


if __name__ == "__main__":
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    date_feature_df_out = get_date_features(holiday_df)
    market_feature_df_out = get_market_features()
    combine_features_out = get_combine_features([date_feature_df_out, market_feature_df_out])
    dateset = generate_dataset(combine_features_out, day_data)
    print(dateset)
