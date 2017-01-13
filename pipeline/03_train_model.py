# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8 15:17:05 2016

@author: Rafael Crescenzi
"""
########## IMPORTS ###########
import os
import argparse
import shutil
import logging

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

########## CONFIG ###########

parser = argparse.ArgumentParser(description='WSDM Cup Model Trainer')
parser.add_argument('-dir', action = "store", dest = "dir", help = 'working directory', required = True)

base_path = os.path.abspath(parser.parse_args().dir)

truth_path = os.path.join(base_path, "truth")
proc_path = os.path.join(base_path, "encoded_data")
production_path = os.path.join(base_path, "production")
models_path = os.path.join(production_path, "models")

########## SCRIPT ###########

try:
    logging.basicConfig(filename=os.path.join(base_path, "train_model.log"),
                        level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("***** Starting Model Training *****")

    if not os.path.exists(truth_path):
        raise Exception("Directory truth data does not exists")

    if not os.path.exists(proc_path):
        raise Exception("Directory for encoded feature data does not exists")

    if os.path.exists(models_path):
        shutil.rmtree(models_path)

    os.makedirs(models_path)

    train_files = sorted([os.path.join(proc_path, f) for f in os.listdir(proc_path) if f.endswith(".bz2")])
    truth_files = sorted([os.path.join(truth_path, f) for f in os.listdir(truth_path) if f.endswith(".csv")])

    logging.info("train files:\n" + ",\n".join(train_files))
    logging.info("truth files:\n" + ",\n".join(truth_files))

    logging.info("Constructing Training Datasets")

    labels = pd.concat([(pd.read_csv(f, index_col="REVISION_ID")["ROLLBACK_REVERTED"] == "T").astype(np.float32) for f in truth_files])
    priv_users = pd.read_csv("priv_users.csv", index_col="userid").astype(np.float32)
    priv_users.columns = ["priv_user"]
    train = pd.concat([pd.read_csv(f, index_col="revisionid").drop("REVISION_SESSION_ID", axis=1).astype(np.float32) for f in train_files])

    train = train.join(priv_users, on="userid").fillna(0)
    train["is_reg"] = (train.userid != -1).astype(np.float32)

    y_train = labels.ix[train.index]

    logging.info("Training Model")

    learner = XGBClassifier(n_estimators=200, max_depth=7)
    learner.fit(train, y_train,
                eval_set=[(train, y_train)],
                eval_metric="auc", early_stopping_rounds=200)

    learner_path = os.path.join(models_path, "model")
    os.mkdir(learner_path)
    learner_path = os.path.join(learner_path, "model.pkl")
    joblib.dump(learner, learner_path)

    logging.info("Copying files to production folder")

    train_cols = ",".join([c for c in train.columns])
    with open(os.path.join(models_path, "train_cols.csv"), "w") as f:
        f.write(train_cols)

    shutil.copy("priv_users.csv", models_path)

    for py_file in ["classifier.py", "rev_parser.py", "utils.py", "Client.py"]:
        shutil.copy(py_file, production_path)

except Exception as e:
    logging.exception("***** An error ocurred *****")
finally:
    logging.info("***** Finished *****")