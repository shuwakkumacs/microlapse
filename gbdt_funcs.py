import pandas as pd
import xgboost as xgb
import numpy as np
import json
from sklearn import model_selection
import re
import string
import operator
from bs4 import BeautifulSoup
from bs4.element import Comment
import html2text
import math
from IPython import embed
import nltk
from datetime import datetime
import glob
import random
from numba import jit,njit

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lightgbm as lgb
from IPython.core.display import display, HTML
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.special import expi,expn

INPUTS = ["button","checkbox","hidden","radio","select_option","submit","text","textarea"]

def format_posthitsurveyanswer(df_phs):
    def finalize(new_row):
        if new_row["ans6"] is not None:
            new_row["action"] = "submit_new"
            new_row["work_time"] = new_row["ans5"]
            new_row["work_time_type"] = new_row["ans6"]
            new_row["ans5"] = None
            new_row["ans6"] = None
        elif new_row["ans2"] is not None:
            new_row["action"] = "return"
            new_row["work_time"] = new_row["ans2"]
            new_row["work_time_type"] = new_row["ans3"]
            new_row["ans2"] = None
            new_row["ans3"] = None
        else:
            new_row["action"] = "submit"
            new_row["work_time"] = new_row["ans0"]
            new_row["work_time_type"] = new_row["ans1"]
            new_row["ans0"] = None
            new_row["ans1"] = None
        return new_row

    columns = ["hit_id", "group_id", "worker_id", "action", "ans0", "ans1", "ans2", "ans3", "ans4", "ans5", "ans6", "work_time", "work_time_type", "created"]
    df = pd.DataFrame({c:[] for c in columns})
    df["hit_id"] = df["hit_id"].astype(int)
    df["worker_id"] = df["worker_id"].astype(int)
    
    df = df[columns]
    
    new_rows = []
    new_row = {c:None for c in columns}
    hit_id = None
    for idx,row in df_phs.iterrows():
        if hit_id!=row["hit_id"]:
            if hit_id is not None:
                new_row = finalize(new_row)
                new_rows.append(new_row)
            new_row = {c:None for c in columns}
            hit_id = row["hit_id"]
    
        new_row["hit_id"]    = int(row["hit_id"])
        new_row["group_id"]  = row["group_id"]
        new_row["worker_id"] = int(row["worker_id"])
        new_row["created"]   = row["created"]
        new_row["ans{}".format(row["idx"])] = row["answer"]
    new_row = finalize(new_row)
    new_rows.append(new_row)
    df = df.append(new_rows)
    df = df.drop(["ans5", "ans6"], axis=1)
    df.to_csv("./csv/chun/scraper_posrhitsurveyanswer_formatted.csv")
    return df


def format_requesterratings(df):

    NULL_DUMMY = -10000
    NULL = np.nan

    col_list = ["to2_all_reward", "to2_all_comm", "to2_all_recommend", "to2_all_rejected", "to2_all_tos", "to2_all_broken", "to2_recent_reward", "to2_recent_comm", "to2_recent_recommend", "to2_recent_rejected", "to2_recent_tos", "to2_recent_broken"]

    df.ix[df["to2_all_pending"].isnull(), ["to2_all_pending"]] = NULL
    df.ix[df["to2_recent_pending"].isnull(), ["to2_recent_pending"]] = NULL
    for c in col_list:
        if "tos" in c or "broken" in c:
            df.ix[df[c].isnull(), [c]] = "[{},{}]".format(NULL_DUMMY,NULL_DUMMY)
            listlen = 2
        else:
            df.ix[df[c].isnull(), [c]] = "[{},{},{}]".format(NULL_DUMMY,NULL_DUMMY,NULL_DUMMY)
            listlen = 3
        df[c] = df[c].apply(lambda x: json.loads(x))
        for i in range(listlen):
            new_c = "{}_{}".format(c,i)
            df[new_c] = df[c].apply(lambda x: x[i])
            df.ix[df[new_c]==NULL_DUMMY, [new_c]] = NULL
        df = df.drop(c, axis=1)

    df = df.drop(["id"], axis=1)
    #print(df.columns)
    return df

def count_words_length_from_inner_html(df):
    df["WLEN:inner_html"] = 0
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = ["'", '"', ':', ';', '.', ',', '!', '?', "'s", "http", "https", "’", "s", "*", "“", \
                  "(", ")", "[", "]", "-", "--", "#", "$", "<", ">", "{", "}"]
    for idx,row in df.iterrows():
        try:
            text = row["TXT:inner_html"]
            words = nltk.word_tokenize(text)
            stop_prefices = ["//", "/"]
            words = [w for w in words if (w not in stop_words) and sum([w.startswith(p) for p in stop_prefices])==0]
            df.loc[idx, "WLEN:inner_html"] = len(words)
        except:
            df.loc[idx, "WLEN:inner_html"] = 0
    return df, df[["META:hit_id","WLEN:inner_html"]]

def drop_contains(df, targets):
    for target in targets:
        columns_del = [column for column in df.columns if target in column]
        df = df.drop(columns_del, axis=1)
    return df

def keep_contains(df, targets):
    columns_keep = []
    for target in targets:
        columns_keep_each = [column for column in df.columns if target in column]
        columns_keep.extend(columns_keep_each)
    df = df[list(set(columns_keep))]
    return df


def get_xgb_model(model_type="classifier"):
    if model_type=="classifier":
        return xgb.XGBClassifier(random_state=random.randint(0,10000))
    elif model_type=="regressor":
        return xgb.XGBRegressor(random_state=random.randint(0,10000))
    else:
        raise Exception("{} is invalid model type name".format(model_type))

def set_hyper_params(mod, params, X_train, y_train, ki=0, time=datetime.now(), cache=False):
    if type(mod)==xgb.XGBClassifier:
        model_type = "cls"
    elif type(mod)==xgb.XGBRegressor:
        model_type = "reg"

    if cache:
        latest = glob.glob("xgb_params/{}_*_{}.csv".format(model_type,ki))[-1]
        print("Using cache file: {}".format(latest))
        f = open(latest, "r")
        params = json.loads(f.read())
        f.close()
    else:
        mod_cv = model_selection.GridSearchCV(mod, params, verbose=0)
        mod_cv.fit(X = X_train, y = y_train)
        params = mod_cv.best_params_
        f = open("xgb_params/{}_{}_{}.csv".format(model_type,time.strftime("%Y%m%d_%H%M%S"),ki), "w")
        json.dump(params,f)
        f.close()
    mod = mod.__class__(**params)
    return mod,params

def print_feature_importances(columns, values, filist=None, thresh=0.0, maxnum=9999999, do_print=True):
    #print("===== FEATURE_IMPORTANCES =====")
    columns = [columns[i] for i in np.argsort(values)[::-1]]
    feature_importances = [values[i] for i in np.argsort(values)[::-1]]
    for i,name in enumerate(columns):
        if filist is not None:
            if name in filist:
                filist[name].append(feature_importances[i])
            else:
                filist[name] = [feature_importances[i]]
        if do_print and not (feature_importances[i]<=thresh or i==maxnum):
            if "to1_" in name or "to2_" in name or "tv_" in name:
                print("\033[91m{} {}\033[0m".format(name, feature_importances[i]))
            elif "KW" in name:
                print("\033[92m{} {}\033[0m".format(name, feature_importances[i]))
            else:
                print(name, feature_importances[i])
    #print("===============================")
    return filist

def get_worker_open_kfold_dataset(df, worker_ids, cv):
    X_train_all = [None]*cv
    y_train_all = [None]*cv
    hit_ids_train_all = [None]*cv
    X_test_all = [None]*cv
    y_test_all = [None]*cv
    hit_ids_test_all = [None]*cv
    worker_ids_test_all = [None]*cv
    X_train_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    y_train_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    X_test_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    y_test_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]

    for i, (train_idx,test_idx) in enumerate(model_selection.KFold(n_splits=cv, shuffle=True).split(df["META:worker_id"].values)):
        X_train_all[i] = df[df["META:worker_id"].isin(train_idx)].reset_index(drop=True)
        y_train_all[i] = df.loc[df["META:worker_id"].isin(train_idx), "PHS:work_time"].reset_index(drop=True)
        hit_ids_train_all[i] = X_train_all[i]["META:hit_id"]
        X_train_all[i] = X_train_all[i].drop(["META:hit_id","META:worker_id","PHS:work_time"], axis=1)
        X_test_all[i] = df[df["META:worker_id"].isin(test_idx)].reset_index(drop=True)
        y_test_all[i] = df.loc[df["META:worker_id"].isin(test_idx), "PHS:work_time"].reset_index(drop=True)
        worker_ids_test_all[i] = X_test_all[i]["META:worker_id"]
        hit_ids_test_all[i] = X_test_all[i]["META:hit_id"]
        X_test_all[i] = X_test_all[i].drop(["META:hit_id","META:worker_id","PHS:work_time"], axis=1)

    return {
        "train": {
            "X": X_train_all,
            "y": y_train_all,
            "hit_ids": hit_ids_train_all
        },
        "test": {
            "X": X_test_all,
            "y": y_test_all,
            "hit_ids": hit_ids_test_all,
            "worker_ids": worker_ids_test_all,
        }
    }

def get_worker_closed_kfold_dataset(df, worker_ids, cv):
    X_train_all = [None]*cv
    y_train_all = [None]*cv
    hit_ids_train_all = [None]*cv
    X_test_all = [None]*cv
    y_test_all = [None]*cv
    hit_ids_test_all = [None]*cv
    worker_ids_test_all = [None]*cv
    X_train_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    y_train_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    X_test_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]
    y_test_sep = [[None for i in range(len(worker_ids))] for j in range(cv)]

    for wi,worker_id in enumerate(worker_ids):
        X_worker = df[df["META:worker_id"]==worker_id]
        y_worker = X_worker["PHS:work_time"]

        if len(X_worker)<4:
            continue
        for ki, (train_idx,test_idx) in enumerate(model_selection.KFold(n_splits=cv,shuffle=True,random_state=None).split(X_worker, y_worker)):
            X_train_sep[ki][wi] = X_worker.iloc[train_idx]
            X_test_sep[ki][wi] = X_worker.iloc[test_idx]
            y_train_sep[ki][wi] = y_worker.iloc[train_idx]
            y_test_sep[ki][wi] = y_worker.iloc[test_idx]

    # combine across all worker data for each fold
    for ki in range(cv):
        X_train_all[ki] = pd.concat(X_train_sep[ki], ignore_index=True)
        y_train_all[ki] = pd.concat(y_train_sep[ki], ignore_index=True)
        hit_ids_train_all[ki] = X_train_all[ki]["META:hit_id"]
        X_train_all[ki] = X_train_all[ki].drop(["META:hit_id","META:worker_id","PHS:work_time"], axis=1)

        X_test_all[ki] = pd.concat(X_test_sep[ki], ignore_index=True)
        y_test_all[ki] = pd.concat(y_test_sep[ki], ignore_index=True)
        hit_ids_test_all[ki] = X_test_all[ki]["META:hit_id"]
        worker_ids_test_all[ki] = X_test_all[ki]["META:worker_id"]
        X_test_all[ki] = X_test_all[ki].drop(["META:hit_id","META:worker_id","PHS:work_time"], axis=1)

    return {
        "train": {
            "X": X_train_all,
            "y": y_train_all,
            "hit_ids": hit_ids_train_all
        },
        "test": {
            "X": X_test_all,
            "y": y_test_all,
            "hit_ids": hit_ids_test_all,
            "worker_ids": worker_ids_test_all,
        }
    }

def get_worker_open_dataset(df, worker_ids):
    X_train_all = [None]*len(worker_ids)
    y_train_all = [None]*len(worker_ids)
    X_test_all = [None]*len(worker_ids)
    y_test_all = [None]*len(worker_ids)

    for i,worker_id in enumerate(worker_ids):
        X_train_all[i] = df[df["META:worker_id"]!=worker_id]
        y_train_all[i] = X_train_all[i]["PHS:work_time"]
        X_train_all[i] = X_train_all[i].drop(["META:worker_id","PHS:work_time"], axis=1)
        X_test_all[i] = df[df["META:worker_id"]==worker_id]
        y_test_all[i] = X_test_all[i]["PHS:work_time"]
        X_test_all[i] = X_test_all[i].drop(["META:worker_id","PHS:work_time"], axis=1)

    return X_train_all, y_train_all, X_test_all, y_test_all

def my_normalize(val, mode=None, method=None):
    if not mode:
        return val
    if not method:
        method = "log10"
    if method=="sqrt":
        if mode=="enc":
            return math.sqrt(val)
        elif mode=="dec":
            return math.pow(val,2)
    elif method=="log10":
        if mode=="enc":
            return math.log10(val)
        elif mode=="dec":
            return math.pow(10,val)

def convert_textdata_to_topn_kw_features(df, colname, n=300):
    def remove_duplicates(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def remove_common(seq):
        remove_words = ["is", "am", "a", "an", "the", "your", "to", "my", ""]
        return [x for x in seq if not (x in remove_words)]
    
    word_count_dict = {}
    word_list = []
    doc_list = []
    for idx,row in df.iterrows():
        # html text
        if type(row[colname])!=str:
            word_list.append([])
            doc_list.append("")
        else:
            all_words = [re.sub('^[{0}]+|[{0}]+$'.format(string.punctuation), '', w.lower()) for w in row[colname].split()] 
            doc_list.append(" ".join(all_words))
            word_list_each = remove_duplicates(all_words)
            word_list.append(word_list_each)
            for word in word_list_each:
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1

    topn = sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=False)[:n]
    topn = remove_common(topn)
    for word in topn:
        df["KW_{}:'{}'".format(colname,word[0])] = pd.Series([1 if word[0] in word_list_each else 0 for word_list_each in word_list])
        
    df = df.drop([colname],axis=1)
    
    return df

def extract_keywords_appearance_from_text(df):
    df_all_text = df["TXT:title"].str.cat(df["TXT:description"].apply(lambda x: " {}".format(x))).str.cat(df["TXT:inner_html"].apply(lambda x: " {}".format(x)))
    keywords = {
        "minute": ["minute", "minutes"],
        "survey": ["survey", "surveys", "questionnaire", "questionnaires"],
        "instruction": ["instruction", "instrictions"],
        "turkprime.com": ["turkprime.com"],
        "research": ["research", "researches"],
        "participation": ["participation", "participations", "participate", "participates", "participated", "participating"],
        "description": ["description", "descriptions", "describe", "describes", "described", "describing"],
        "opinion": ["opinion", "opinions"],
        "play": ["play", "plays", "played", "playing", "playback", "play-back"],
        "click": ["click", "clicks", "clicked", "clicking"],
        "qualification": ["qualification", "qualifications", "qualify", "qualifies", "qualified", "qualifying"],
        "comment": ["comment", "comments", "commented", "commenting"],
        "copy": ["copy", "copies", "copied", "copying"],
        "paste": ["paste", "pastes", "pasted", "pasting"],
        "return": ["return", "returns", "returned", "returning"],
        "answer": ["answer", "answers", "answered", "answering"],
        "summary": ["summary", "summaries", "summarize", "summarizes", "summarized", "summarizing"],
        "watch": ["watch", "watches", "watched", "watching"],
        "leave": ["leave", "leaves", "left", "leaving"],
        "comprehension": ["comprehension", "comprehensions", "comprehend", "comprehends", "comprehended", "comprehending"],
        "identification": ["identification", "identifications", "identify", "identifies", "identified", "identifying"],
        "read": ["read", "reads", "reading"],
        "example": ["example", "examples"],
        "image": ["image", "images"],
        "design": ["design", "designs", "designed", "designing"],
        "note": ["note", "notes", "noted", "noting"],
        "type": ["type", "types", "typed", "typing"],
        "bonus": ["bonus", "bonuses"],
        "video": ["video", "videos"],
        "website": ["website", "websites", "site", "sites"]
    }
    
    df_kw_list_series = df_all_text.apply(lambda x: [xx.lower() for xx in nltk.word_tokenize(x)] if type(x)==str else x)
    flag = False
    for key,kws in keywords.items():
        df["KW:{}".format(key)] = df_kw_list_series.apply(lambda tokens: 1 if type(tokens )==list and any(kw in tokens for kw in kws) else 0)
    return df

def convert_inner_html_to_words_count(df):
    def remove_duplicates(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    all_words_count_all = []
    unique_words_count_all = []
    for idx,row in df.iterrows():
        if type(row["TXT:inner_html"])!=str:
            all_words = []
            unique_words = []
        else:
            all_words = [re.sub('^[{0}]+|[{0}]+$'.format(string.punctuation), '', w.lower()) for w in row["TXT:inner_html"].split()] 
            unique_words = remove_duplicates(all_words)
        all_words_count_all.append(len(all_words))
        unique_words_count_all.append(len(unique_words))

    df["TXTM:inner_html_words_count_all"] = pd.Series(all_words_count_all) 
    df["TXTM:inner_html_words_count_unique"] = pd.Series(unique_words_count_all) 
    
    return df

def get_append_data(df,df_html):
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.bypass_tables = True
    data = {
        "texts": [],
        "text_url_counts": [],
        "a_href_counts": [],
        "all_url_counts": [],
        "qualtrics_url_counts": [],
        "img_src_counts": [],
        "audio_src_counts": [],
        "video_src_counts": []
    }
    for idx,row in df_html.iterrows():
        if idx%100==0:
            print(idx)
        html = row["html"]
        soup = BeautifulSoup(html, "html.parser")
        te = soup.findAll(text=True)
        visible_texts = filter(tag_visible, te)
        text = u" ".join([t.strip() for t in visible_texts]).replace("\"", "").replace("'", "")
        #text = h.handle(html).replace("__","").replace("#","").replace("**","").replace("<table>","").replace("<tr>","").replace("<td>","").replace("</table>","").replace("</tr>","").replace("</td>","").replace("\n","   ")
        urls = list(set([u[0] for u in re.findall("(((http|https):\/\/)?(([A-Za-z0-9\-]+\.)+(com|net|org|gov|edu|mil|int|be|cl|fi|in|jp|nu|pt|pw|py|gl|ly|io|re|sa|se|su|tn|tr|io|de|cn|uk|info|nl|eu|ru)|((?<!.)[0-9]{1,3}\.){3}[0-9]{1,3})([?:/][A-Za-z0-9-._~:/?#@!$&()+;=%]*)?)(?=[\s'\"<>])", text)]))
        a_href_urls = list(set([atag.get("href") for atag in soup.find_all("a") if atag.get("href") and not atag.get("href").startswith("javascript") and not atag.get("href").startswith("mailto") and not atag.get("href").startswith("#")]))
        all_urls = list(set(urls+a_href_urls)) 
        img_src_urls = list(set([atag.get("src") for atag in soup.find_all("img") if atag.get("src")]))
        audio_src_urls = list(set([atag.get("src") for atag in soup.find_all("audio") if atag.get("src")]))
        video_src_urls = list(set([atag.get("src") for atag in soup.find_all("video") if atag.get("src")]))
        df_html.loc[idx, "TXT:inner_html"] = text
        df_html.loc[idx, "URL:text_url_counts"] = len(urls)
        df_html.loc[idx, "URL:a_href_counts"] = len(a_href_urls)
        df_html.loc[idx, "URL:all_url_counts"] = len(all_urls)
        df_html.loc[idx, "URL:qualtrics_url_counts"] = len([url for url in all_urls if "qualtrics.com" in url])
        df_html.loc[idx, "URL:img_src_counts"] = len(img_src_urls)
        df_html.loc[idx, "URL:audio_src_counts"] = len(audio_src_urls)
        df_html.loc[idx, "URL:video_src_counts"] = len(video_src_urls)
        for type in ["hidden","submit","text","radio","checkbox","range","button","file","number","password","url","date","time","email","reset","tel"]:
            df_html.loc[idx, "INP:{}".format(type)] = len(soup.findAll("input", {"type": type}))
        df_html.loc[idx, "INP:text"] += len(soup.findAll("input", {"type": ""}))
        df_html.loc[idx, "INP:button"] += len(soup.findAll("button"))
        df_html.loc[idx, "INP:select_option"] = len(soup.findAll("option"))
        df_html.loc[idx, "INP:textarea"] = len(soup.findAll("textarea"))

    df_html = df_html.drop(["html"], axis=1)

    df = pd.merge(df, df_html, how="left", left_on="META:hit_id", right_on="hit_id")

    df = df.drop(["hit_id"], axis=1)

    return df,df_html

def append_submit_count(df):
    df = df.sort_values(["META:worker_id","META:group_id","META:hit_id"], ascending=True)
    submit_count = 1
    total_work_time = 0
    worker_id_last = None
    group_id_last = None
    for idx,row in df.iterrows():
        worker_id = row["META:worker_id"]
        group_id = row["META:group_id"]
        work_time = row["PHS:work_time"]
        if not (worker_id_last==worker_id and group_id_last==group_id):
            worker_id_last = worker_id
            group_id_last = group_id
            submit_count = 0
        df.ix[idx, "META:submit_count"] = submit_count
        df.ix[idx, "META:total_work_time"] = total_work_time
        submit_count += 1
        total_work_time += work_time
    #df.to_csv("csv/second_all_new_20181002.csv", quotechar="", escapechar="\\")
    return df, df[["META:hit_id","META:submit_count"]]

def train(all_data, xgb_model_type, cv_idx, gs_exec_level="all", time=datetime.now(), categories=None, labels=None, params=None):
    X_train = all_data["train"]["X"][cv_idx]

    mod = get_xgb_model(xgb_model_type)
    if xgb_model_type=="classifier":
        if not categories or not labels:
            raise Exception("categories and labels required for classifier")
        y_train = pd.cut(all_data["train"]["y"][cv_idx], categories, labels=labels)
    elif xgb_model_type=="regressor":
        y_train, y_test = all_data["train"]["y"][cv_idx].apply(lambda x: my_normalize(x,"enc")), \
                          all_data["test"]["y"][cv_idx].apply(lambda x: my_normalize(x,"enc"))
        
    if not params:
        params = {
            "learning_rate":[0.1,0.2],
            "max_depth": [5,6],
            "subsample":[0.5,0.6],
            "colsample_bytree": [0.5,0.6],
            "n_estimators": [10,30]
        }

        if gs_exec_level=="all":
            mod, params = set_hyper_params(mod, params, X_train, y_train, ki=cv_idx, time=time, cache=False)
        elif gs_exec_level=="once":
            if cv_idx==0:
                mod, params = set_hyper_params(mod, params, X_train, y_train, ki=cv_idx, time=time, cache=False)
            else:
                mod, params = set_hyper_params(mod, params, X_train, y_train, ki=cv_idx, time=time, cache=True)
        elif gs_exec_level=="never":
            mod, params = set_hyper_params(mod, params, X_train, y_train, ki=cv_idx, time=time, cache=True)
    else:
        print("using cached params")
        mod = mod.__class__(**params)
                                                                                                  
    mod.fit(X_train, y_train, eval_metric="mae")
    return mod,params

def my_confusion_matrix(cls_result,labels):
    # test, pred -> pandas.Series
    cmx_data = np.zeros((len(labels),len(labels)),dtype=np.int32)
    hit_ids_data = []
    for i in range(len(labels)):
        hit_ids_data_row = []
        test_i = cls_result.loc[cls_result["y_test_cls"]==labels[i]]
        for j in range(len(labels)):
            cmx_data[i,j] = len(test_i.loc[test_i["y_pred_cls"]==labels[j]])
            hit_ids_data_row.append(test_i.loc[test_i["y_pred_cls"]==labels[j],"data"].tolist())
        hit_ids_data.append(hit_ids_data_row)
            
    return cmx_data, hit_ids_data

def draw_pred_result_heatmap(cls_result,cls):
    cmx_data, hit_ids_data = my_confusion_matrix(cls_result,cls)
    df_cmx, df_cmx_perc = get_df_cmx_with_perc(cmx_data,cls)
#     draw_heatmap(df_cmx_perc)
    
    return df_cmx, hit_ids_data
    
def get_df_cmx_with_perc(cmx_data,cls):
    row_sum = np.sum(cmx_data,axis=1)
    idx = []
    for c, rsum in zip(cls,row_sum):
        idx.append("{}  ({})".format(c,rsum))
#         idx.append("{}".format(c))
    df_cmx = pd.DataFrame(cmx_data, index=idx, columns=cls)
    df_cmx_perc = df_cmx.divide(row_sum,axis="rows").transpose()
    
    return df_cmx, df_cmx_perc

def draw_heatmap(df_cmx, figsize=(32,9), font_scale=3, cbar=False, cmap="Blues", vmax=1, fmt=".2g", annot=True):
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)
    ax = sns.heatmap(df_cmx, cmap=cmap, cbar=cbar, vmax=vmax, fmt=fmt, annot=annot)
    for i in range(len(df_cmx.index)):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=5))
    plt.xlabel("Actual working time [seconds]  (# of HIT records)")
    plt.ylabel("Predicted working time [seconds]")

def categorize_work_time_by_idx(work_time_upper, work_time):
    for i,t in enumerate(work_time_upper):
        if work_time<t:
            return i
    return len(work_time_upper)

def psychological_working_time(x,enc="enc",precision=0.1,K=100):
#     [a,b,c] = [108.25617935,53.25578258,-400.55663472]
#     [a,b,c] = [146.01,116.81,-629.38]
#     [a,b,c] = [109.82,62.82,-367.43]
#     [a,b,c] = [119.54,93.76,-456.72]
#     [a,b,c] = [ 137.78235923,  124.37127684, -587.36784393]
    [a,b,c] = [ 162.47780333,  175.29841573, -770.07622944]
#     [a,b] = [0.18,143.15]
    
    if enc=="enc":
        return (np.exp(-c/a)*K*expi(np.log(x+b)+c/a))/a
#         return K*np.log(abs(a*x+b))/a
    elif enc=="dec":
        y=x
        xrange = range(0,10000,100)
        xrange2 = np.arange(0,100,precision)
        x_ = -1
        for xr in xrange:
            if psychological_working_time(xr,"enc")>y:
                x_ = xr-100
                break
                
        pprev = 0
        for xr in xrange2:
            x = x_+xr
            p = psychological_working_time(x,"enc")
            if p>y:
                if abs(y-pprev)<abs(y-p):
                    return x-precision
                else:
                    return x
            else:
                pprev = p
                  
@jit
def encode_val(val, enc_method):
    if enc_method=="log10":
        return math.log(val,10)
    elif enc_method=="log":
        return math.log(val,math.e)
    elif enc_method=="psycho":
        return psychological_working_time(val,"enc")
    else:
        return val

@jit
def decode_val(val, enc_method):
    if enc_method=="log10":
        return math.pow(10,val)
    elif enc_method=="log":
        return math.pow(math.e,val)
    elif enc_method=="psycho":
        return psychological_working_time(val,"dec")
    else:
        return val

def get_categories(name):
    if name=="META:template":
        category_list = [
            "No template",
            "Survey Link",
            "Survey",
            "Image A/B",
            "Writing",
            "Data Collection",
            "Data Collect From Website",
            "Image Tagging",
            "Image Transcription",
            "Categorization",
            "Other"
        ]
    elif name=="QTR:employment_status":
        category_list = [
            "Employed full time",
            "Employed part time",
            "Prefer not to say",
            "Retired",
            "Student",
            "Unemployed looking for work",
            "Unemployed not looking for work"
        ]
    elif name=="QTR:gender":
        category_list = [
            "Male",
            "Female",
            "Prefer not to say"
        ]

    return category_list


def get_categorical(df, name, from_type="string"):
    category_list = get_categories(name)

    if from_type=="string":
        df[name] = df[name].apply(lambda x: category_list.index(x))
        return df
    elif from_type=="onehot":
        old_cols = [name+"_"+c.replace(" ","_") for c in category_list]
        x = df[old_cols].stack()
        x2 = pd.Series(x[x!=0].index.get_level_values(1))
        x2 = x2.apply(lambda x: x.replace(name+"_", "").replace("_", " ")).to_frame(name=name)

        df[old_cols].to_csv("{}.csv".format(name))

        x3 = get_categorical(x2, name)
        df[name] = x3[name].values
        df = df.drop(old_cols, axis=1)
        
        return df
    
def preprocess(df):
    ### WLEN
    df = extract_keywords_appearance_from_text(df)
    df["WLEN:inner_html"] = df["TXT:inner_html"].apply(lambda x: len(nltk.word_tokenize(x)) if type(x)!=float else 0)
    df = drop_contains(df, ["TXT:"])

    ### TMPL
    df = get_categorical(df,"META:template",from_type="string")
    #df["META:template"] = df["META:template"].apply(lambda x: x.replace(" ","_"))
    #df = pd.get_dummies(df, columns=["META:template"])

    ### QTR
    df = get_categorical(df,"QTR:employment_status",from_type="onehot")
    df = get_categorical(df,"QTR:gender",from_type="onehot")

    ### INP (perc)
    df_inp_cols = [col for col in df.columns if 'INP:' in col]
    df_inp_cols_perc = [col+"_perc" for col in df_inp_cols]
    df_inp = df[df_inp_cols]
    inp_sum = df_inp.sum(axis=1)[0]
    if inp_sum>0:
        df_inp_perc = df_inp.divide(inp_sum,axis=0)
    else:
        df_inp_perc = df_inp

    df[df_inp_cols_perc] = df_inp_perc

    ### REP
    df = df.drop(["REP:to1_comm",
                    "REP:to1_fast",
                    "REP:to1_tos"],axis=1)

    rep_ave_cols = ["REP:to1_pay","REP:to1_fair"]
    rep_zero_cols = ["REP:to1_reviews"]
    for c in rep_ave_cols:
        rep_ave = df.loc[~df[c].isnull(),c].median()
        if pd.isna(rep_ave):
            if c=="REP:to1_pay":
                rep_ave = 2.84
            elif c=="REP:to1_fair":
                rep_ave = 4.06
        df.loc[df[c].isnull(),c] = rep_ave
        df[c] = pd.to_numeric(df[c])
    for c in rep_zero_cols:
        df.loc[df[c].isnull(),c] = 0

    df = df.reindex(sorted(df.columns), axis=1)

    return df

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100