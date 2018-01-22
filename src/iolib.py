import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime

project_root_path = os.path.abspath('..')
raw_data_name = project_root_path + r"\data\raw_data.csv"
caihui_index_path = project_root_path + r"\data\caihui_index.csv"
wind_index_path = project_root_path + r"\data\wind_index.xls"
shibor_index_path = project_root_path + r"\data\Shibor.csv"
rnn_result_path_xls = project_root_path + r"\data\rnn_result.xls"
caihui_index_dic = {"300": "HS300"}

def read_raw_data(filename=raw_data_name):
    '''
        读入原始数据
    '''
    raw_data_df = pd.read_csv(filename)
    raw_data_df_se = raw_data_df[raw_data_df["type"] == "SE"]
    raw_data_df_se = raw_data_df_se[["time", "amount"]]
    raw_data_df_se.columns = ["time", "SE"]
    raw_data_df_se["time"] = pd.to_datetime(raw_data_df_se["time"])
    raw_data_df_se = raw_data_df_se.set_index("time")
    raw_data_df_re = raw_data_df[raw_data_df["type"] == "RE"]
    raw_data_df_re = raw_data_df_re[["time", "amount"]]
    raw_data_df_re.columns = ["time", "RE"]
    raw_data_df_re["time"] = pd.to_datetime(raw_data_df_re["time"])
    raw_data_df_re = raw_data_df_re.set_index("time")
    raw_data_combine_df = pd.concat([raw_data_df_se, raw_data_df_re], axis=0)
    raw_data_combine_df = raw_data_combine_df.fillna(0)
    raw_data_combine_df = raw_data_combine_df.sort_index()
    raw_data_combine_df.insert(2,"save",raw_data_combine_df["SE"]-raw_data_combine_df["RE"])
    return raw_data_combine_df


def get_matrix_by_day(minute_df):
    '''
        将原始数据按天汇总
    '''
    current_date = "2017-01-01"
    re_total = 0
    se_total = 0
    day_matrix_df = pd.DataFrame(columns=["time","RE","SE","save"])
    for index, row in minute_df.iterrows():
        date_str = index._date_repr
        re = row["RE"]
        se = row["SE"]
        if date_str == current_date:
            re_total += re
            se_total += se
        else:
            save_total = se_total-re_total
            insert_dic = {}
            insert_dic["time"] = current_date
            insert_dic["RE"] = re_total
            insert_dic["SE"] = se_total
            insert_dic["save"] = save_total
            day_matrix_df = day_matrix_df.append(insert_dic,ignore_index=True)
            re_total = re
            se_total = se
            current_date = date_str

    day_matrix_df["time"] = pd.to_datetime(day_matrix_df["time"])
    day_matrix_df = day_matrix_df.set_index("time")
    return day_matrix_df


# start和end是形如"2017-01-01"的字符串，分别表示开始时间和结束时间
# 返回在开始时间和结束时间之间的日期字符串列表（不包括end这一天）
def dateRange(start, end, step=1, format="%Y-%m-%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]


def get_caihui_index_net_matrix(start_date_str, end_date_str, fill=True):
    '''
        读入市场数据1
    '''
    index_return_df = pd.DataFrame()
    datelist = dateRange(start_date_str, end_date_str)
    index_net_raw = pd.read_csv(caihui_index_path, dtype="str")
    index_net = index_net_raw[["icode", "mcap", "tdate"]]
    index_net_columns = ["symbol", "mcap", "date"]
    index_net.columns = index_net_columns
    index_net = index_net.drop_duplicates(["symbol", "date"])
    index_net = index_net[index_net["symbol"].isin(caihui_index_dic.keys())]
    index_net = index_net[index_net["date"].isin(datelist)]
    index_symbol_list = list(set(index_net["symbol"].values.tolist()))
    for symbol in index_symbol_list:
        sub_index_df = index_net[index_net["symbol"] == symbol]
        sub_index_df["date"] = pd.to_datetime(sub_index_df["date"])
        sub_index_df = sub_index_df.set_index("date")
        sub_index_mcap_df = sub_index_df["mcap"].astype('float64')
        index_return_df.insert(0, caihui_index_dic[symbol], sub_index_mcap_df)
    index_return_df = index_return_df.sort_index()
    if fill:
        index_return_df = index_return_df.fillna(method="pad")
        index_return_df = index_return_df.fillna(method="bfill")
    return index_return_df


def get_wind_index_net_matrix(start_date_str, end_date_str, fill=True):
    '''
        读入市场数据2
    '''
    datelist = dateRange(start_date_str, end_date_str)
    index_net_raw = pd.read_excel(wind_index_path)
    index_net_raw = index_net_raw[index_net_raw["date"].isin(datelist)]
    index_net_raw["date"] = pd.to_datetime(index_net_raw["date"])
    index_net_raw = index_net_raw.set_index("date")
    index_net_raw = index_net_raw.sort_index()
    if fill:
        index_net_raw = index_net_raw.fillna(method="pad")
        index_net_raw = index_net_raw.fillna(method="bfill")
    return index_net_raw

def get_shibor_net_matrix(start_date_str, end_date_str, fill=True):
    '''
        读入shibor数据
    '''
    datelist = dateRange(start_date_str, end_date_str)
    index_net_raw = pd.read_csv(shibor_index_path)
    index_net_raw = index_net_raw[index_net_raw["date"].isin(datelist)]
    index_net_raw["date"] = pd.to_datetime(index_net_raw["date"])
    index_net_raw = index_net_raw.set_index("date")
    index_net_raw = index_net_raw.sort_index()
    if fill:
        index_net_raw = index_net_raw.fillna(method="pad")
        index_net_raw = index_net_raw.fillna(method="bfill")
    return index_net_raw


if __name__ == "__main__":
    minute_data = read_raw_data()
    day_data = get_matrix_by_day(minute_data)
    # minute_data.to_excel(project_root_path+r"\data\minute_matrix.xls")
    caihui_index_df = get_caihui_index_net_matrix("2017-01-01", "2017-07-24", fill=False)
    shibor_index_df = get_shibor_net_matrix("2017-01-01", "2017-07-24", fill=False)
    wind_index_df = get_wind_index_net_matrix("2017-01-01", "2017-07-24", fill=False)
    index_df = pd.concat([caihui_index_df,shibor_index_df,wind_index_df],axis=1)
    day_data.to_excel(project_root_path + r"\data\day_matrix.xls")
    mix_df = pd.concat([day_data,index_df],axis=1)
    mix_df = mix_df.fillna(method="pad")
    return_corr = mix_df.corr()
    print(return_corr)
    mix_df_norm = (mix_df - mix_df.min()) / (mix_df.max() - mix_df.min())
    mix_df_norm.plot(kind="line")
    # plt.plot(mix_df_norm.index.values, mix_df_norm["save"].values, 'r', linewidth=2.5, linestyle="-", label="save")
    # plt.plot(mix_df_norm.index.values, mix_df_norm["HS300"].values, 'g', linewidth=2.5, linestyle="-", label="HS300")
    plt.legend(loc='upper left')
    plt.xlabel('time')
    plt.ylabel('amount')
    plt.grid()
    plt.show()
    # print(data)
