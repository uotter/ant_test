import os as os
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
from sklearn.cross_validation import train_test_split
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库
import numpy as np
import src.iolib as il
import src.features_genarator as fg
import datetime as datetime


best_gbr_n = 90
best_rf_n = 33
# best_gbr_n = 130
# best_rf_n = 187

def opt_detail():
    '''
        调优GBR和RF所用函数
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    date_feature_df_out = fg.get_date_features(holiday_df)
    market_feature_df_out = fg.get_market_features()
    combine_features_out = fg.get_combine_features([date_feature_df_out, market_feature_df_out])
    dataset_total = fg.generate_dataset(combine_features_out, day_data)
    date_dataset_total = fg.generate_dataset(date_feature_df_out, day_data)
    dataset = dataset_total.ix[:strptime("2017-07-23","%Y-%m-%d"),:]
    date_dataset = date_dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
    data_predict_X = dataset_total.ix[strptime("2017-07-24","%Y-%m-%d"):,:len(dataset_total.columns) - 1]
    date_dataset_predict_X = date_dataset_total.ix[strptime("2017-07-24","%Y-%m-%d"):,:len(date_dataset_total.columns) - 1]
    X = dataset.ix[:, :len(dataset.columns) - 1]
    y = dataset.ix[:, len(dataset.columns) - 1:]
    X_date = date_dataset.ix[:, :len(date_dataset.columns) - 1]
    y_date = date_dataset.ix[:, len(date_dataset.columns) - 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)
    X_date_train, X_date_test, y_date_train, y_date_test = train_test_split(X_date, y_date, test_size=0.3,
                                                        random_state=1)
    model_names = ['RandomForest',  'GBR']  # 不同模型的名称列表

    n_estimators = np.linspace(1, 200, 50)
    result_curve_df = pd.DataFrame()
    count = 0
    for n in n_estimators:
        count += 1
        n = int(n)
        model_gbr = GradientBoostingRegressor(n_estimators=n)  # 建立梯度增强回归模型对象
        model_rf = RandomForestRegressor(n_estimators=n, criterion='mse', random_state=1, n_jobs=-1)
        model_dic = [model_rf, model_gbr]  # 不同回归模型对象的集合
        result_dic = {}
        for i in range(2):  # 读出每个回归模型对象
            model = model_dic[i]
            model_name = model_names[i]
            model.fit(X_train, y_train)
            tmp_score = mean_absolute_error(y_train, model.predict(X_train))  # 计算每个回归指标结果
            tmp_test_score = mean_absolute_error(y_test, model.predict(X_test))
            result_dic[model_name + "_train"] = tmp_score
            result_dic[model_name + "_test"] = tmp_test_score
            model = model_dic[i]
            model.fit(X_date_train, y_date_train)
            tmp_date_score = mean_absolute_error(y_date_train, model.predict(X_date_train))  # 计算每个回归指标结果
            tmp_date_test_score = mean_absolute_error(y_date_test, model.predict(X_date_test))
            result_dic[model_name + "_date_train"] = tmp_date_score
            result_dic[model_name + "_date_test"] = tmp_date_test_score
        result_dic["para"] = n
        result_curve_df = result_curve_df.append(result_dic, ignore_index=True)
        print("start with n = " + str(n) + ", " + str(count) + "/" + str(len(n_estimators)))
    result_curve_df = result_curve_df.set_index("para")
    result_curve_df = result_curve_df.sort_index()
    plt.figure(1, figsize=(15, 5))  # 创建画布
    result_curve_df.plot()
    plt.show()
    return result_curve_df


def predict_result():
    '''
        预测24-28日的值
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    date_feature_df_out = fg.get_date_features(holiday_df)
    market_feature_df_out = fg.get_market_features()
    combine_features_out = fg.get_combine_features([date_feature_df_out, market_feature_df_out])
    dataset_total = fg.generate_dataset(combine_features_out, day_data)
    date_dataset_total = fg.generate_dataset(date_feature_df_out, day_data)
    dataset = dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
    date_dataset = date_dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
    data_predict_X = dataset_total.ix[strptime("2017-07-24", "%Y-%m-%d"):, :len(dataset_total.columns) - 1]
    date_dataset_predict_X = date_dataset_total.ix[strptime("2017-07-24", "%Y-%m-%d"):,
                             :len(date_dataset_total.columns) - 1]
    X = dataset.ix[:, :len(dataset.columns) - 1]
    y = dataset.ix[:, len(dataset.columns) - 1:]
    model_gbr = GradientBoostingRegressor(n_estimators=best_gbr_n)  # 建立梯度增强回归模型对象
    model_rf = RandomForestRegressor(n_estimators=best_rf_n, criterion='mse', random_state=1, n_jobs=-1)
    model_gbr.fit(X,y)
    model_rf.fit(X,y)
    grb_predict_y = model_gbr.predict(data_predict_X)
    rf_predict_y = model_rf.predict(data_predict_X)
    print("gbr predict:")
    print(grb_predict_y)
    print("rf predict:")
    print(rf_predict_y)


def reg_case():
    '''
        综合使用多模型进行比对
    '''
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
    minute_data = il.read_raw_data()
    day_data = il.get_matrix_by_day(minute_data)
    date_feature_df_out = fg.get_date_features(holiday_df)
    market_feature_df_out = fg.get_market_features()
    combine_features_out = fg.get_combine_features([date_feature_df_out, market_feature_df_out])
    dataset_total = fg.generate_dataset(combine_features_out, day_data)
    date_dataset_total = fg.generate_dataset(date_feature_df_out, day_data)
    dataset = dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
    date_dataset = date_dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
    data_predict_X = dataset_total.ix[strptime("2017-07-24", "%Y-%m-%d"):, :len(dataset_total.columns) - 1]
    date_dataset_predict_X = date_dataset_total.ix[strptime("2017-07-24", "%Y-%m-%d"):,
                             :len(date_dataset_total.columns) - 1]
    X = date_dataset.ix[:, :len(date_dataset.columns) - 1]
    y = date_dataset.ix[:, len(date_dataset.columns) - 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)

    # 训练回归模型
    n_folds = 6  # 设置交叉检验的次数
    model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
    model_lr = LinearRegression()  # 建立普通线性回归模型对象
    model_etc = ElasticNet()  # 建立弹性网络回归模型对象
    model_svr = SVR()  # 建立支持向量机回归模型对象
    model_grid_svr = GridSearchCV(SVR(kernel='linear', gamma=0.1), cv=5,
                                  param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                              "gamma": np.logspace(-2, 2, 5)})
    model_gbr = GradientBoostingRegressor(n_estimators=best_gbr_n)  # 建立梯度增强回归模型对象
    model_rf = RandomForestRegressor(n_estimators=best_rf_n, criterion='mse', random_state=1, n_jobs=-1)
    model_names = ['RandomForest', 'LinearRegression', 'GBR']  # 不同模型的名称列表
    model_dic = [model_rf, model_lr, model_gbr]  # 不同回归模型对象的集合
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表
    pre_y_test_list = []  # 各个回归模型预测的y值列表
    for model in model_dic:  # 读出每个回归模型对象
        scores = cross_val_score(model, X_train, y_train, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        pre_y_list.append(model.fit(X_train, y_train).predict(X_train))  # 将回归训练中得到的预测y存入列表
        pre_y_test_list.append(model.fit(X_train, y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表
    # 模型效果指标评估
    n_samples, n_features = X_train.shape  # 总样本量,总特征数
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    model_metrics_test_list = []  # 回归评估指标列表
    for i in range(3):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        tmp_test_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_train, pre_y_list[i])  # 计算每个回归指标结果
            tmp_test_score = m(y_test, pre_y_test_list[i])
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
            tmp_test_list.append(tmp_test_score)
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
        model_metrics_test_list.append(tmp_test_list)
    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
    df3 = pd.DataFrame(model_metrics_test_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
    print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
    print(70 * '-')  # 打印分隔线
    print('cross validation result:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('regression metrics:')  # 打印输出标题
    print(df2)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
    print('regression test metrics:')  # 打印输出标题
    print(df3)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
    print('short name \t full name')  # 打印输出缩写和全名标题
    print('ev \t explained_variance')
    print('mae \t mean_absolute_error')
    print('mse \t mean_squared_error')
    print('r2 \t r2')
    print(70 * '-')  # 打印分隔线
    # 模型效果可视化
    plt.figure(1, figsize=(15, 5))  # 创建画布
    plt.plot(np.arange(X_train.shape[0]), y_train, color='r', label='true y')  # 画出原始值的曲线
    color_list = ['b', 'k', 'g', 'y', 'c']  # 颜色列表
    linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
    for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
        plt.plot(np.arange(X_train.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题
    plt.show()  # 展示图像


if __name__ == "__main__":
    # resutlt_df = opt_detail()
    # resutlt_df.to_csv("result1-200.csv")
    # print(resutlt_df)
    reg_case()
    # predict_result()
