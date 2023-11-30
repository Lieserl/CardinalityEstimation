import evaluation_utils as eval_utils
import json
import statistics as stats
import range_query as rq
import learn_from_query


def gen_report(act, est_results):
    f = open("./eval/report.md", "w")
    f.write("| name | p50 | p80 | p90 | p99 |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    for name in est_results:
        est = est_results[name]
        eval_utils.draw_act_est_figure(name, act, est)
        p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(act, est)
        f.write("| %s | %.2f | %.2f | %.2f | %.2f |\n" % (name, p50, p80, p90, p99))

    f.write("\n")
    for name in est_results:
        f.write("![%s](%s.png)\n\n" % (name, name))
    f.close()

    with open('./eval/results.json', 'w') as outfile:
        est_results['act'] = act
        json.dump(est_results, outfile)


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

    est_avi, est_ebo, est_min_sel, act = [], [], [], []
    for item in test_data:
        range_query = rq.ParsedRangeQuery.parse_range_query(item['query'])
        est_avi.append(stats.AVIEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_ebo.append(stats.ExpBackoffEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        est_min_sel.append(stats.MinSelEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        act.append(item['act_rows'])

    _, _, est_AI1, _ = learn_from_query.est_AI1(train_data, test_data, table_stats, columns)
    _, _, est_AI2, _ = learn_from_query.est_AI2(train_data, test_data, table_stats, columns)

    gen_report(act, {
        "avi": est_avi,
        "ebo": est_ebo,
        "min_sel": est_min_sel,
        "your_model_1": est_AI1,
        "your_model_2": est_AI2,
    })