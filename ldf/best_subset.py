import itertools
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def best_n_avg(sources, n, buckets, all_periods, filled_history, target_buckets):
    def avg_calc(i, c):
        df_bucket = filled_history[["TARGET_PERIOD", "BUCKET"]].drop_duplicates().set_index("TARGET_PERIOD", drop=True)
        df_agg = filled_history[(filled_history.FCT_SOURCE.isin(c))][["TARGET_PERIOD"] + target_buckets].groupby(
            "TARGET_PERIOD").mean() # .mean() / .sum()
        ret = np.log(df_agg.lookup(df_bucket.BUCKET.index, df_bucket.BUCKET.values) / 100)
        df = pd.DataFrame([i * np.ones(len(ret)), np.arange(len(ret)), ret]).T
        df.columns = ["comb", "period_enum", "log_score"]
        return df

    comb = list(set(itertools.combinations(sources, n)))
    comb = [list(t) for t in comb]
    ret = Parallel(n_jobs=-1)(delayed(avg_calc)(i, c) for i, c in tqdm(enumerate(comb)))
    ret = pd.concat(ret).pivot(index="period_enum", columns="comb", values="log_score")
    return ret, comb