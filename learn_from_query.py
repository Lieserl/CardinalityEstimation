import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import statistics as stats
import torch.utils.data
import model


def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # 变量解释 range_query: 一个 ParsedRangeQuery 类，即将 query 中信息提取出的结果
    # table_stats: 实际传入的是 title_stats.json 里提取出的数据
    # considered_cols: 在这个 range_query 中需要考虑的列(描述字段)
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = []
    # YOUR CODE HERE: extract features from query
    for col in considered_cols:
        min_val = table_stats.columns[col].min_val()
        max_val = table_stats.columns[col].max_val()
        (left, right) = range_query.column_range(col, min_val, max_val)
        norm_left = min_max_normalize(left, min_val, max_val)
        norm_right = min_max_normalize(right, min_val, max_val)
        feature.extend([norm_left, norm_right])

    # 理论上需要引入 considered_cols 但实际上传参时默认全部列都考虑，所以就可以省略
    avi_sel = stats.AVIEstimator.estimate(range_query, table_stats)
    ebo_sel = stats.ExpBackoffEstimator.estimate(range_query, table_stats)
    min_sel = stats.MinSelEstimator.estimate(range_query, table_stats)

    feature.extend([avi_sel, ebo_sel, min_sel])

    return feature


def preprocess_queries(queries, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queries:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.
        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(range_query, table_stats, columns)
        label = act_rows
        features.append(feature)
        labels.append(label)

    features = np.array(features)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.query_data = data

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


# def est_ai1(train_data, test_data, table_stats, columns):
#     """
#     produce estimated rows for train_data and test_data
#     """
#     train_dataset = QueryDataset(train_data, table_stats, columns)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
#     train_est_rows, train_act_rows = [], []
#     # YOUR CODE HERE: train procedure
#
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     model1 = model.Model1(num_input=15, para_1=100, para_2=30, para_3=50, num_output=1)
#     model1 = model1.to(device)
#
#     loss_fn = model.MSELoss()
#     loss_fn = loss_fn.to(device)
#
#     learning_rate = 1e-2
#     optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
#
#     epoch = 500
#     for i in range(epoch):
#         step = 0
#         ave_loss = 0
#         for data in train_loader:
#             features, act_rows = data
#             features = features.to(device)
#             act_rows = act_rows.to(device)
#             features = features.float()
#             est_rows = torch.abs(model1(features))
#
#             act = torch.maximum(act_rows, est_rows)
#             est = torch.minimum(act_rows, est_rows)
#             est = torch.where(est == 0, 1.0, est)
#             q_error = torch.div(act, est)
#             # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))
#
#             loss = loss_fn(q_error.float())
#             ave_loss += loss.item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1)
#             optimizer.step()
#
#             step = step + 1
#
#         print("epoch {}, loss {}".format(i + 1, ave_loss / step))
#
#     test_dataset = QueryDataset(test_data, table_stats, columns)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
#     test_est_rows, test_act_rows = [], []
#     # YOUR CODE HERE: test procedure
#     tot_loss = 0
#     with torch.no_grad():
#         step = 0
#         for data in test_loader:
#             features, act_rows = data
#             features = features.to(device)
#             act_rows = act_rows.to(device)
#             features = features.float()
#             est_rows = torch.abs(model1(features))
#             for i in range(10):
#                 test_est_rows.append(est_rows[i].item())
#                 test_act_rows.append(act_rows[i].item())
#
#             act = torch.maximum(act_rows, est_rows)
#             est = torch.minimum(act_rows, est_rows)
#             est = torch.where(est == 0, 1.0, est)
#             q_error = torch.div(act, est)
#
#             # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))
#
#             loss = loss_fn(q_error.float())
#             tot_loss = tot_loss + loss.item()
#
#             step = step + 1
#         print("step {}, loss {}".format(step, tot_loss / step))
#
#     return train_est_rows, train_act_rows, test_est_rows, test_act_rows

def est_ai1(data_set, table_stats, columns):
    """
    produce estimated rows for train_data and test_data
    """
    _data = list(zip(*preprocess_queries(data_set, table_stats, columns)))
    kfold = KFold(n_splits=10)

    train_est_rows, train_act_rows = [], []
    test_est_rows, test_act_rows = [], []

    _loss = []
    min_train_loss = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(_data)):
        print("Fold: {}".format(fold + 1))

        train_set = [_data[i] for i in train_idx]
        test_set = [_data[i] for i in test_idx]

        train_dataset = QueryDataset(train_set)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)

        # YOUR CODE HERE: train procedure

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model1 = model.Model1(num_input=15, para_1=100, para_2=30, para_3=50, num_output=1).to(device)

        loss_fn = model.MSELoss().to(device)

        learning_rate = 1e-2
        optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

        epoch = 300
        min_loss = 1e5
        for i in range(epoch):
            step = 0
            ave_loss = 0
            for data in train_loader:
                features, act_rows = data
                features = features.to(device)
                act_rows = act_rows.to(device)
                est_rows = torch.abs(model1(features.float()))

                act = torch.maximum(act_rows, est_rows)
                est = torch.minimum(act_rows, est_rows)
                est = torch.where(est == 0, 1.0, est)
                q_error = torch.div(act, est)
                # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))

                loss = loss_fn(q_error.float())
                ave_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1)
                optimizer.step()

                step = step + 1
            min_loss = min_loss if (min_loss < ave_loss / step) else (ave_loss / step)
            print("epoch {}, loss {}".format(i + 1, ave_loss / step))
        min_train_loss.append(min_loss)

        test_dataset = QueryDataset(test_set)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

        # YOUR CODE HERE: test procedure
        tot_loss = 0
        with torch.no_grad():
            step = 0
            for data in test_loader:
                features, act_rows = data
                features = features.to(device)
                act_rows = act_rows.to(device)
                est_rows = torch.abs(model1(features.float()))

                for i in range(10):
                    test_est_rows.append(est_rows[i].item())
                    test_act_rows.append(act_rows[i].item())

                act = torch.maximum(act_rows, est_rows)
                est = torch.minimum(act_rows, est_rows)
                est = torch.where(est == 0, 1.0, est)
                q_error = torch.div(act, est)

                # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))

                loss = loss_fn(q_error.float())
                tot_loss = tot_loss + loss.item()

                step = step + 1

            _loss.append(tot_loss / step)
            print("step {}, loss {}".format(step, tot_loss / step))

    for i, loss in enumerate(_loss):
        print("Fold {}, min_loss {}, trian_loss {}".format(i + 1, min_train_loss[i], loss))

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows

def est_ai2(data_set, table_stats, columns):
    """
    produce estimated rows for train_data and test_data
    """
    _data = list(zip(*preprocess_queries(data_set, table_stats, columns)))
    kfold = KFold(n_splits=10)

    train_est_rows, train_act_rows = [], []
    test_est_rows, test_act_rows = [], []

    _loss = []
    min_train_loss = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(_data)):
        print("Fold: {}".format(fold + 1))

        train_set = [_data[i] for i in train_idx]
        test_set = [_data[i] for i in test_idx]

        train_dataset = QueryDataset(train_set)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)

        # YOUR CODE HERE: train procedure

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model2 = model.Model2(num_input=15, para_1=100, para_2=50, para_3=50, num_output=1).to(device)

        loss_fn = model.MSELoss().to(device)

        learning_rate = 3e-2
        optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)

        epoch = 300
        min_loss = 1e5
        for i in range(epoch):
            step = 0
            ave_loss = 0
            for data in train_loader:
                features, act_rows = data
                features = features.to(device)
                act_rows = act_rows.to(device)
                est_rows = torch.abs(model2(features.float()))

                act = torch.maximum(act_rows, est_rows)
                est = torch.minimum(act_rows, est_rows)
                est = torch.where(est == 0, 1.0, est)
                q_error = torch.div(act, est)
                # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))

                loss = loss_fn(q_error.float())
                ave_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1)
                optimizer.step()

                step = step + 1

            min_loss = min_loss if (min_loss < ave_loss / step) else (ave_loss / step)
            print("epoch {}, loss {}".format(i + 1, ave_loss / step))

        min_train_loss.append(min_loss)

        test_dataset = QueryDataset(test_data, table_stats, columns)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
        test_est_rows, test_act_rows = [], []
        # YOUR CODE HERE: test procedure
        tot_loss = 0
        with torch.no_grad():
            step = 0
            for data in test_loader:
                features, act_rows = data
                features = features.to(device)
                act_rows = act_rows.to(device)
                features = features.float()
                est_rows = torch.abs(model2(features))
                for i in range(10):
                    test_est_rows.append(est_rows[i].item())
                    test_act_rows.append(act_rows[i].item())

                act = torch.maximum(act_rows, est_rows)
                est = torch.minimum(act_rows, est_rows)
                est = torch.where(est == 0, 1.0, est)
                q_error = torch.div(act, est)

                # q_error = torch.max(torch.div(act_rows, est_rows), torch.div(est_rows, act_rows))

                loss = loss_fn(q_error.float())
                tot_loss = tot_loss + loss.item()

                step = step + 1
            print("step {}, loss {}".format(step, tot_loss / step))

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, data_set, table_stats, columns):
    if model == 'mlp_adam':
        est_fn = est_ai1
    else:
        est_fn = est_ai2

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(data_set, table_stats, columns)

    # name = f'{model}_train_{len(train_data)}'
    # eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    # p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    # print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = 'data/title_stats.json'
    train_json_file = 'data/query_train_18000.json'
    test_json_file = 'data/validation_2000.json'
    kfold_json_file = 'data/kfold_20000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)
    with open(kfold_json_file, 'r') as f:
        kfold_data = json.load(f)

    eval_model('mlp_adam', kfold_data, table_stats, columns)
    # eval_model('mlp_sgd', kfold_data, table_stats, columns)