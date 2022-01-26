import pandas as pd
import os
import numpy as np


def load_ecb_forecasts(path, start=None, end=None):
    files_in_folder = os.listdir(path)
    files_in_folder = [f for f in files_in_folder if "Q" in f]
    files_in_folder = [f for f in files_in_folder if f.split(".")[0] >= start] if start else files_in_folder
    files_in_folder = [f for f in files_in_folder if f.split(".")[0] <= end] if end else files_in_folder

    f_list = []
    for i, f_name in enumerate(files_in_folder):
        sample_file = pd.read_csv(path + f"\\{f_name}", header=1)[:300]
        period = f_name.split(".")[0]
        year = period.split("Q")[0]
        if "Q1" in f_name:
            target = str(year) + "Dec"
        if "Q2" in f_name:
            target = str(int(year) + 1) + "Mar"
        if "Q3" in f_name:
            target = str(int(year) + 1) + "Jun"
        if "Q4" in f_name:
            target = str(int(year) + 1) + "Sep"
        f = sample_file[sample_file["TARGET_PERIOD"] == target].copy()
        f["PERIOD_ENUM"] = i
        f["FORECAST_DATE"] = period
        f_list.append(f)
    history = pd.concat(f_list, axis=0, ignore_index=True)
    history = history[[col for col in history.columns if "Unnamed" not in col]]
    c = history.columns[3:].to_list()
    c.remove("PERIOD_ENUM")
    c.remove("FORECAST_DATE")
    history = history[history.columns[:3].to_list() + ["PERIOD_ENUM"] + ["FORECAST_DATE"] + c]
    return history


def get_target_buckets(buckets_from, buckets_to):
    target_buckets = []
    for f, t in zip(buckets_from, buckets_to):
        name = ""
        if f is not None:
            f *= 1.0
            is_negf = "" if f >= 0 else "N"
            name += "F" + is_negf + str(abs(f)).replace(".", "_")
        if t is not None:
            t *= 1.0
            is_negp = "" if t >= 0 else "N"
            name += "T" + is_negp + str(abs(t)).replace(".", "_")
        target_buckets.append(name)
    return target_buckets


def which_bucket(num):
    if num <= -0.5:
        return "TN0_5"
    if -0.5 < num <= 0:
        return "FN0_5T0_0"
    if 0 < num <= 0.5:
        return "F0_0T0_5"
    if -0.5 < num <= 1:
        return "F0_5T1_0"
    if -0.5 < num <= 1.5:
        return "F1_0T1_5"
    if -0.5 < num <= 2:
        return "F1_5T2_0"
    if -0.5 < num <= 2.5:
        return "F2_0T2_5"
    if -0.5 < num <= 3:
        return "F2_5T3_0"
    if -0.5 < num <= 3.5:
        return "F3_0T3_5"
    if -0.5 < num <= 4:
        return "F3_5T4_0"
    if num > 4:
        return "F4_0"


def get_bounds(names):
    u_s1 = [col.split("T") for col in names]
    upper_bound = []
    for el in u_s1:
        if len(el) == 1:
            upper_bound.append(None)
        else:
            upper_bound.append(float(el[1].replace("_", ".").replace("N", "-")))

    l_s1 = [col.split("F") for col in names]
    lower_bound = []
    for el in l_s1:
        if len(el) == 1:
            lower_bound.append(None)
        else:
            lower_bound.append(float(el[1][:3].replace("_", ".").replace("N", "-")))

    return upper_bound, lower_bound


def assign_buckets(target_buckets, all_buckets):
    up_target, down_target = get_bounds(target_buckets)
    d = {}
    assigned = []
    buckets = all_buckets.copy()
    for i, t in enumerate(target_buckets):
        dd = []
        buckets = [buckets[k] for k in range(len(buckets)) if k not in assigned]
        up_all, down_all = get_bounds(buckets)
        assigned = []
        for j, a in enumerate(buckets):
            if (up_all[j] is not None) and (up_target[i] is not None) and (up_all[j] <= up_target[i]):
                if (down_all[j] is not None) and (down_target[i] is not None) and (down_all[j] > down_target[i]):
                    dd.append(a)
                    assigned.append(j)
                elif (down_all[j] is None) and (down_target[i] is None):
                    dd.append(a)
                    assigned.append(j)
                elif (up_all[j] <= up_target[i]) and (down_target[i] is None):
                    dd.append(a)
                    assigned.append(j)
                elif (up_all[j] <= up_target[i]):
                    dd.append(a)
                    assigned.append(j)
            elif (up_all[j] is None) and (up_target[i] is None):
                if (down_all[j] >= down_target[i]):
                    dd.append(a)
                    assigned.append(j)
            elif (up_all[j] is None) and (up_target[i] is not None) and (down_all[j] is not None) and (
                    down_target[i] is not None) and (down_all[j] == down_target[i]):
                dd.append(a)
                assigned.append(j)
            else:
                pass
        d[t] = dd
    return d


def create_hist_pivot(history_clean):
    hist = history_clean[["PERIOD_ENUM", "FCT_SOURCE", "TARGET_PERIOD"]].groupby(
        ["FCT_SOURCE", "TARGET_PERIOD"]).count().reset_index().pivot(index="TARGET_PERIOD", columns="FCT_SOURCE",
                                                                     values="PERIOD_ENUM").reset_index().copy()
    hist["YEAR"] = hist["TARGET_PERIOD"].str[:4]
    hist["MONTH"] = hist["TARGET_PERIOD"].str[4:]
    hist.loc[hist["MONTH"] == "Mar", "MONTH"] = 3
    hist.loc[hist["MONTH"] == "Jun", "MONTH"] = 6
    hist.loc[hist["MONTH"] == "Sep", "MONTH"] = 9
    hist.loc[hist["MONTH"] == "Dec", "MONTH"] = 12
    hist["DATE"] = pd.to_datetime(hist[['YEAR', 'MONTH']].assign(DAY=1))
    hist = hist.sort_values("DATE").set_index("DATE")[
        history_clean.FCT_SOURCE.astype(int).sort_values().astype("str").unique()]
    return hist


def count_max_cont_missingness(hist):
    d_max_nan = {}
    for c in hist.columns:
        d_max_nan[c] = hist[c].isnull().astype(int).groupby(hist[c].notnull().astype(int).cumsum()).cumsum().max()
    d_max_nan = pd.DataFrame(d_max_nan.values(), index=d_max_nan.keys())
    d_max_nan.columns = ["Count"]
    return d_max_nan
