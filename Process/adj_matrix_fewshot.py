import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import pickle

def extract_data(task, n_samples, threshold):


    # load data
    # in each csv file under data/news/articles/raw, [n_samples] labeled training articles come first, 
    # followed by the test articles

    data_path = 'data/news_articles_raw/' + task + '_full_train' + str(n_samples) + '.csv'
    social_path = 'data/social_context_raw/' + task + '_socialcontext_train' + str(n_samples) + '.csv'
    data_df = pd.read_csv(data_path, encoding = 'utf-8')
    social_df = pd.read_csv(social_path, encoding = 'utf-8')
    news_id = data_df['news_id'].tolist()
    sid_list = social_df['sid'].tolist()
    user_list = social_df['uid'].tolist()
    sid = list(dict.fromkeys(sid_list))
    print(len(sid))
    print(len(sid_list))

    # collect label knowledge from training samples to build a ground-truth matrix 
    train_dist = []
    for i in range(n_samples):
        if i < n_samples // 2:
            train_dist.append([1.0, 0.0])
        else:
            train_dist.append([0.0, 1.0])
    
    train_dist = np.array(train_dist)

    uid = []
    for x, y in zip(sid_list, user_list):
        if x in news_id:
            uid.append(y)
    count = 0 
    for n in news_id:
        if n in sid_list:
            count = count + 1
        else:
            print(n)
    user = list(dict.fromkeys(uid))
    print(len(uid), len(user), count)


    # only consider the active social users, who has at least [threshold] engagements in spreading news articles
    c = Counter({k: c for k, c in Counter(uid).items() if c >= threshold})
    freq_users = list(dict(c).keys())
    print(len(freq_users))

    Uids = {idx:i for i, idx in enumerate(freq_users)}
    Sids = {idx:i for i, idx in enumerate(sid)}


    US_relation = []
    freq_news = []
    for x, y in zip(sid_list, user_list):
        if y in freq_users:
            US_relation.append([x, y])
            freq_news.append(x)


    # locate the news without active social users, and assign unique node IDs to each news article
    news = list(dict.fromkeys(freq_news))
    not_prop = list(set(sid) ^ set(news))

    new_id = 0
    for id in not_prop:
        print(id)
        temp = len(freq_users) + new_id 
        US_relation.append([id, temp])
        Uids[temp] = temp
        new_id = new_id + 1
        

    print(len(freq_users))
    print(len(not_prop))

    US_relation = np.array([(Uids[u], Sids[s], 1) for s,u in US_relation])
    adj = sp.csc_matrix((US_relation[:, 2], (US_relation[:, 0], US_relation[:, 1])), shape=(len(freq_users) + len(not_prop), len(sid)), dtype=np.float32)

    # compute adjacency matrix based on the user engagement matrix
    adj = adj.transpose().dot(adj)


    # normalize the adjacency matrix
    rowsum = np.array(adj.sum(1))
    D_row = np.power(rowsum, -0.5).flatten()
    D_row[np.isinf(D_row)] = 0.
    D_row = sp.diags(D_row)

    colsum = np.array(adj.sum(0))
    D_col = np.power(colsum, -0.5).flatten()
    D_col[np.isinf(D_col)] = 0.
    D_col = sp.diags(D_col)

    adj = adj.dot(D_col).transpose().dot(D_row).transpose() 


    pickle.dump([train_dist, adj], open('data/adjs_from_scratch/' + task + '_nn_relations_' + str(n_samples) + '.pkl', 'wb'))


# set default threshold to 5, in line with our work
threshold_u = 5

extract_data('fang', 16, threshold_u)
# extract_data('fang', 32, threshold_u)
# extract_data('fang', 64, threshold_u)
# extract_data('fang', 128, threshold_u)
# extract_data('politifact', 16, threshold_u)
# extract_data('politifact', 32, threshold_u)
# extract_data('politifact', 64, threshold_u)
# extract_data('politifact', 128, threshold_u)
# extract_data('gossipcop', 16, threshold_u)
# extract_data('gossipcop', 32, threshold_u)
# extract_data('gossipcop', 64, threshold_u)
# extract_data('gossipcop', 128, threshold_u)
