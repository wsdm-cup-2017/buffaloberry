# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:35:03 2016

@author: Rafael Crescenzi
"""

import os
from time import time

import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

import xgboost

from rev_parser import parse_xml

class Classifier(object):

    def __init__(self, working_dir):
        self.working_dir = os.path.abspath(working_dir)
        self.mappers_dir = os.path.join(self.working_dir, "mappers")
        self.counters_dir = os.path.join(self.working_dir, "counters")
        self.models_dir = os.path.join(self.working_dir, "models")

        for d in ["mappers", "models", "counters"]:
            tdir = os.path.join(self.working_dir, d)
            if not os.path.exists(tdir):
                os.makedirs(tdir)

        self.mappers = {m.split(".")[0]: pd.Series.from_csv(os.path.join(self.mappers_dir, m), encoding="utf-8") for m in
                        os.listdir(self.mappers_dir)
                        if m.endswith(".csv")}

        self.counters = {m.split(".")[0]: pd.Series.from_csv(os.path.join(self.counters_dir, m), encoding="utf-8") for m in
                        os.listdir(self.counters_dir)
                        if m.endswith(".csv")}

        if os.path.exists(os.path.join(self.models_dir, "model/model.pkl")):
            self.model = joblib.load(os.path.join(self.models_dir, "model/model.pkl"))

        if os.path.exists(os.path.join(self.models_dir, "train_cols.csv")):
            with open(os.path.join(self.models_dir, "train_cols.csv")) as f:
                self.train_cols = [a.strip() for a in f.readline().split(",")]

        if os.path.exists(os.path.join(self.models_dir, "priv_users.csv")):
            self.priv_users = pd.read_csv(os.path.join(self.models_dir, "priv_users.csv"),
                                          index_col="userid").index.tolist()

        self.meta_headers = None
        self.rolling_probs = pd.DataFrame([], columns=["sessid", "single_prob", "sess_prob"])
        self.rolling_probs_2 = defaultdict(list)

        if os.path.exists(os.path.join(self.mappers_dir, "unique_tags.csv")):
            with open(os.path.join(self.mappers_dir, "unique_tags.csv")) as f:
                self.unique_tags = [a.strip() for a in f.readline().split(",")]
        else:
            self.unique_tags = []

        self.n_revs = 0
        self.start = time()

    def create_mappings(self, df):
        tipos = df.dtypes
        for c in df.columns:
            if tipos[c] == "object":
                if c == "timestamp": continue
                serie = df[c].str.lower()
                if c == "REVISION_TAGS":
                    self.set_unique_tags(serie)
                vals = serie.dropna().unique().compute().values
                if c == "afectedProperty":
                    vals = np.asarray(list({v.split(":")[0].strip() for v in vals}))
                mapper = pd.Series(np.arange(vals.shape[0]),
                                   index=vals, name=c)
                mapper.to_csv(os.path.join(self.mappers_dir, c + ".csv"), encoding="utf-8")
                self.mappers[c] = mapper
            elif c in ["itemid", "userid"]:
                counter = df[c].dropna().value_counts().compute()
                if -1 in counter.index:
                    counter.ix[-1] = 1
                self.counters[c] = counter
                counter.to_csv(os.path.join(self.counters_dir, c + ".csv"), encoding="utf-8")

    def set_unique_tags(self, tags):
        self.unique_tags = set()
        for t in tags.unique().compute().values:
            if type(t) is str:
                for tag in t.split(","):
                    self.unique_tags.add(tag)
        self.unique_tags = [a.strip() for a in self.unique_tags]
        with open(os.path.join(self.mappers_dir,
                               "unique_tags.csv"), "w") as cw:
            cw.write(",".join(self.unique_tags))

    def parse_date(self, data_str):
        try:
            res = int(data_str.split(" ")[-1].split(":")[0].strip())
        except:
            res = -1
        return res

    def apply_mappings(self, df):
        for c in df:
            if c == "timestamp":
                df["hour"] = df[c].fillna(-1).apply(lambda x: self.parse_date(x))
                df.drop("timestamp", axis=1, inplace=True)
            elif c in self.mappers.keys():
                if c == "REVISION_TAGS":
                    temp = df[c].dropna()
                    for tag in sorted(self.unique_tags):
                        tag_name = "TAG:" + tag
                        idx = temp[temp.map(lambda row_tag: tag in row_tag)]
                        idx = idx.index
                        df[tag_name] = 0
                        df.ix[idx, tag_name] = 1
                        del idx
                if c == "afectedProperty":
                    df[c] = df[c].fillna("").astype(str).str.lower().apply(lambda x: x.split(":")[0].strip())
                else:
                    df[c] = df[c].fillna("").str.lower()
                df = df.join(self.mappers[c].rename(c+"_encoded"), on=c)
                df = df.drop(c, axis=1)
            if c in self.counters.keys():
                df[c] = df[c].fillna(-2)
                df = df.join(self.counters[c].rename(c+"_freq"), on=c)
                df[c+"_freq"] = df[c+"_freq"].fillna(0)
        return df.fillna(-1)

    def apply_mappings_dict(self, df):
        for c in list(df.keys()):
            if c == "timestamp":
                df["hour"] = self.parse_date(df[c])
            elif c in self.mappers.keys():
                if c == "REVISION_TAGS":
                    for tag in sorted(self.unique_tags):
                        tag_name = "TAG:" + tag
                        if tag in df[c]:
                            df[tag_name] = 1
                        else:
                            df[tag_name] = 0
                if c == "afectedProperty":
                    df[c] = df[c].lower().split(":")[0].strip()
                else:
                    df[c] = df[c].lower()
                df[c + "_encoded"] = self.mappers[c].get(df[c], -1)
            if c in self.counters.keys():
                df[c + "_freq"] = self.counters[c].get(df[c], 0)
        return df


    def predict_proba(self, meta_text, xml_text):

        self.n_revs += 1
        meta = meta_text.splitlines()

        if len(meta) > 1:
            self.meta_headers = [c.strip() for c in meta[0].strip().strip("\n").strip().split(",")]

        try:
            meta = [c.strip() for c in meta[-1].strip().strip("\n").strip().split(",")]
            meta = {k: v for k, v in zip(self.meta_headers, meta)}
            revid = int(meta["REVISION_ID"])
        except Exception as e:
            print("error 1")
            return meta[-1].split(",")[0].strip(), "0.02"


        try:
            rev = parse_xml(xml_text.splitlines())
            for k in meta:
                if (len(meta[k]) > 0) or (k == "REVISION_TAGS"):
                    rev[k] = meta[k]

            rev = self.apply_mappings_dict(rev)
            rev["priv_user"] = rev["userid"] in self.priv_users
            rev["is_reg"] = rev["userid"] != -1
            rev_ar = []
            for k in self.train_cols:
                val = np.float32(rev.get(k, -1))
                if np.isfinite(val):
                    rev_ar.append(val)
                else:
                    rev_ar.append(-1)
            if rev["action_encoded"] == 0:
                prob = -1000.0
            else:
                rev = xgboost.DMatrix(data=np.asarray([rev_ar]), feature_names=self.train_cols)
                prob = self.model.booster().predict(rev, output_margin=False)[0]
            ok_prob = True
        except Exception as e:
            if "<ip" in xml_text:
                prob = 0.1
            else:
                prob = 0.01
            ok_prob = False
            print("error 2")

        try:
            if ok_prob:
                sessid = int(meta["REVISION_SESSION_ID"])
                self.rolling_probs_2[sessid].append(prob)
                prob = np.mean(self.rolling_probs_2[sessid])
        except Exception as e:
            print("error 3")

        if self.n_revs % 1000 == 0:
            print(str(revid), str(prob))
            print("done", self.n_revs, "in", self.start - time())
        return str(revid), str(prob)
