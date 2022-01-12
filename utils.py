import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.extmath import randomized_svd


bearer_token = "AAAAAAAAAAAAAAAAAAAAALZSXQEAAAAARi%2F72S3XwjHsF98xzNH%2B9jdLbHo%3Df74ruLqy9lsoRWmONo53iN5DaStNGFl5Z3Cl4oyqTuxD3xvSsz"


def get_content_by_twitter_id(twitter_id, bearer_token):
    end_point = f"https://api.twitter.com/2/tweets/{twitter_id}"
    param = {"expansions": "author_id",
             "tweet.fields": "created_at"}
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(end_point, headers=headers, params=param)
    return response.json()


def get_content_by_twitter_id_list(twitter_id_list, bearer_token):
    end_point = "https://api.twitter.com/2/tweets"
    param = {"ids": ",".join(twitter_id_list),
             "expansions": ["author_id"],
             "tweet.fields": ["created_at"]}
    headers = {
        "Authorization": f"Bearer {bearer_token}"}
    response = requests.get(end_point, headers=headers, params=param, )
    return response.json()


def extract_line_content(line):
    eid, label, tweets = line.split("\t")
    eid = eid.split(":")[1]
    label = label.split(":")[1]
    tweets = tweets.split(" ")
    if tweets[-1] == "\n":
        return eid, label, tweets[:-1]
    return eid, label, tweets


def get_twitter_conversations(file):
    eids, labels, twitter_ids = [], [], []
    with open(file) as f:
        for line in f.readlines():
            line_eid, line_label, line_twitter_ids = extract_line_content(line)
            eids.append(line_eid)
            labels.append(line_label)
            twitter_ids.append(line_twitter_ids)
    return eids, labels, twitter_ids


def get_twitter_from_dir(dir):
    files = list(Path(dir).glob("*.csv"))
    eids = []
    dfs = []
    for f in files:
        # df = df.reindex(sorted(df.columns), axis=1)
        eids.append(f.stem)
        dfs.append(pd.read_csv(f))
    return eids, dfs


def get_user_engagement(eids, dfs):
    user_id = {}
    eid_engagement = {}
    for eid, df in zip(eids, dfs):
        engagement = {}
        count = df["author_id"].value_counts()
        for index, c in count.iteritems():
            try:
                u_id = user_id[index]
            except KeyError:
                u_id = len(user_id)
                user_id[index] = u_id
            engagement[u_id] = c
        eid_engagement[eid] = engagement
    return user_id, eid_engagement


def get_matrix_from_engagement(user_id, engagement):
    m = np.zeros((len(user_id), len(engagement)))
    for i, val in enumerate(engagement.values()):
        m[list(val.keys()), i] = list(val.values())
    return m, np.clip(m, 0, 1)


def get_reduced_svm_mat(mat, k_dim, random_state=42):
    u, s, vt = randomized_svd(mat, n_components=k_dim, random_state=random_state)
    return u.dot(np.diag(s))


def get_user_matrices(dir, n_components, random_state=42, return_user_dict=True):
    eids, dfs = get_twitter_from_dir(dir)
    user_dict, engagement = get_user_engagement(eids, dfs)
    user_mat, user_article_mat = get_matrix_from_engagement(user_dict, engagement)
    user_mat_reduced = get_reduced_svm_mat(user_mat, n_components[0], random_state)
    user_article_mat_reduced = get_reduced_svm_mat(user_article_mat, n_components[1], random_state)
    if return_user_dict:
        return user_dict, user_mat_reduced, user_article_mat_reduced
    return user_mat_reduced, user_article_mat_reduced


def add_bins(dfs, bin_size, threshold):
    for df in dfs:
        df["bin"] = pd.cut(df["timedelta"], range(0, min(int(df["timedelta"].max() + bin_size), threshold), bin_size),
                           include_lowest=True, right=False, labels=False)
        df["timedelta_previous_bin"] = df["bin"].diff(periods=1)
        df.at[0, "timedelta_previous_bin"] = 0
    return dfs
