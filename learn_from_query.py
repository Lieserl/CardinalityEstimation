import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
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
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


def est_ai1(train_data, test_data, table_stats, columns):
    """
    produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure

    learning_rate = 1e-2
    epoch = 300
    para_1, para_2, para_3 = 100, 30, 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model.Model1(num_input=15, para_1=para_1, para_2=para_2, para_3=para_3, num_output=1).to(device)
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    loss_fn = model.MSELoss().to(device)

    min_loss = 1e5
    for i in range(epoch):
        step = 0
        ave_loss = 0
        for data in train_loader:
            features, act_rows = data
            features = features.to(device)
            act_rows = act_rows.to(device)
            est_rows = torch.abs(model1(features.float()))

            loss = loss_fn(est_rows, act_rows)
            ave_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1)
            optimizer.step()

            step = step + 1

        if min_loss > ave_loss / step:
            min_loss = ave_loss / step
        print("epoch {}, loss {}".format(i + 1, ave_loss / step))

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)
    test_est_rows, test_act_rows = [], []
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

            loss = loss_fn(est_rows, act_rows)
            tot_loss = tot_loss + loss.item()

            step = step + 1

        print("step {}, loss {}".format(step, tot_loss / step))

    print("min_train_loss: {}".format(min_loss))

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_ai2(train_data, test_data, table_stats, columns):
    """
    produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []

    learning_rate = 3e-2
    epoch = 300
    para_1, para_2, para_3 = 80, 60, 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = model.Model1(num_input=15, para_1=para_1, para_2=para_2, para_3=para_3, num_output=1).to(device)
    optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)
    loss_fn = model.MSELoss().to(device)

    for i in range(epoch):
        step = 0
        ave_loss = 0
        for data in train_loader:
            features, act_rows = data
            features = features.to(device)
            act_rows = act_rows.to(device)
            est_rows = torch.abs(model2(features.float()))

            loss = loss_fn(est_rows, act_rows)
            ave_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1)
            optimizer.step()

            step = step + 1

        print("epoch {}, loss {}".format(i + 1, ave_loss / step))

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)
    test_est_rows, test_act_rows = [], []

    tot_loss = 0
    with torch.no_grad():
        step = 0
        for data in test_loader:
            features, act_rows = data
            features = features.to(device)
            act_rows = act_rows.to(device)
            est_rows = torch.abs(model2(features.float()))

            for i in range(10):
                test_est_rows.append(est_rows[i].item())
                test_act_rows.append(act_rows[i].item())

            loss = loss_fn(est_rows, act_rows)
            tot_loss = tot_loss + loss.item()

            step = step + 1
        print("step {}, loss {}".format(step, tot_loss / step))

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp_adam':
        est_fn = est_ai1
    else:
        est_fn = est_ai2

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

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
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('mlp_adam', train_data, test_data, table_stats, columns)
    eval_model('mlp_sgd', train_data, test_data, table_stats, columns)