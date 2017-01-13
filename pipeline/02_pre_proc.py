# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 14:47:15 2016

@author: Rafael Crescenzi
"""
########## IMPORTS ###########
import gc
import os
import logging
from timeit import default_timer as timer
import argparse

import numpy as np
import pandas as pd
from dask import dataframe as dd

from classifier import Classifier
from utils import data_dtypes, meta_dtypes

########## CONFIG ###########

parser = argparse.ArgumentParser(description='WSDM Cup XML parser')
parser.add_argument('-dir', action = "store", dest = "dir", help = 'working directory', required = True)

base_path = os.path.abspath(parser.parse_args().dir)

meta_path = os.path.join(base_path, "meta")
proc_path = os.path.join(base_path, "proc_data")
target_path = os.path.join(base_path, "encoded_data")
production_path = os.path.join(base_path, "production")

########## SCRIPT ###########

try:
    logging.basicConfig(filename=os.path.join(base_path, "pre_proc.log"),
                        level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("***** Starting Feature Processing *****")

    if not os.path.exists(meta_path):
        raise Exception("Directory for meta data does not exists")

    if not os.path.exists(proc_path):
        raise Exception("Directory for feature data does not exists")

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if not os.path.exists(production_path):
        os.makedirs(production_path)

    clf = Classifier(production_path)
    train_files = sorted([f for f in os.listdir(proc_path) if (f.endswith(".bz2")) and (not f.endswith("2016_03.bz2"))])
    logging.info("Will process files for mappings and counters: \n" + ",\n".join(train_files))

    data = dd.concat([dd.read_csv(urlpath=os.path.join(proc_path, f),
                                  dtype=data_dtypes,
                                  compression="bz2", blocksize=None)
                      for f in train_files])

    clf.create_mappings(data)
    del data

    meta = dd.read_csv(os.path.join(meta_path, "wdvc16_meta.csv"),
                       dtype=meta_dtypes)
    clf.create_mappings(meta)

    logging.info("Done computing mappings and counters")
    logging.info("Will process files for training: \n" + ",\n".join(train_files[-5:]))

    meta = pd.read_csv(os.path.join(meta_path, "wdvc16_meta.csv"),
                       index_col="REVISION_ID",  dtype=meta_dtypes)

    for f in train_files[-5:]:
        data = pd.read_csv(os.path.join(proc_path, f), dtype=data_dtypes, compression="bz2")
        data = data.join(meta, on="revisionid")
        clf.apply_mappings(data).to_csv(os.path.join(target_path, f),
                                        index=False, float_format='%.4g',
                                        compression="bz2", encoding="utf-8")
        del data
        gc.collect()
    del meta
    gc.collect()

    logging.info("Will process wdvc16_2016_03.bz2 for validation")
    data = pd.read_csv(os.path.join(proc_path, "wdvc16_2016_03.bz2"),
                       dtype=data_dtypes, compression="bz2")
    data = data.join(pd.read_csv(os.path.join(meta_path, "wdvc16_2016_03_meta.csv"),
                                 index_col="REVISION_ID", dtype=meta_dtypes),
                                 on="revisionid")
    clf.apply_mappings(data).to_csv(os.path.join(target_path, "wdvc16_2016_03.bz2"),
                       index=False, float_format='%.4g',
                       compression="bz2", encoding="utf-8")

except Exception as e:
    logging.exception("***** An error ocurred *****")
finally:
    logging.info("***** Finished *****")

