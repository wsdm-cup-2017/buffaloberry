# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:36:27 2016

@author: Rafael Crescenzi
"""
########## IMPORTS ###########
import time
import os
import subprocess
import logging
from timeit import default_timer as timer
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rev_parser import parse_xml, xml_generator
from utils import data_dtypes

########## CONFIG ###########

parser = argparse.ArgumentParser(description='WSDM Cup XML parser')
parser.add_argument('-dir', action = "store", dest = "dir", help = 'working directory', required = True)

base_path = os.path.abspath(parser.parse_args().dir)

file_path = os.path.join(base_path, "raw")
target_path = os.path.join(base_path, "proc_data")

########## SCRIPT ###########

try:
    logging.basicConfig(filename=os.path.join(base_path, "parse_xml.log"),
                        level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("***** Starting XML parser *****")

    if not os.path.exists(file_path):
        raise Exception("Directory for raw xml data does not exists")

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    with Parallel(n_jobs=-1, backend="threading", verbose=1) as parallel:

        del_func = delayed(parse_xml)

        files = sorted([f for f in os.listdir(file_path) if f.endswith(".xml.7z")])
        logging.info("Will process files: \n" + ",\n".join(files))
        for zip_file in files:

            file_name = zip_file.strip(".7z")
            target_file = os.path.join(target_path, file_name).strip(".xml") + ".bz2"
            logging.info("*" * 10)

            if os.path.exists(target_file):
                logging.info("skipping " + zip_file)
                logging.info("*" * 10)
            else:
                logging.info("doing " + zip_file)
                logging.info("*" * 10)

                start = timer()

                if file_name not in os.listdir(file_path):
                    print(subprocess.check_output(["7za", "e", "-o" + file_path, os.path.join(file_path, zip_file)]).decode("utf-8"))

                revisions = pd.DataFrame.from_dict(parallel(del_func(rev) for rev in xml_generator( os.path.join(file_path, file_name))))

                for col in data_dtypes:
                    if col not in revisions:
                        revisions[col] = np.nan

                time.sleep(5)
                revisions.to_csv(target_file, index=False, float_format='%.4g', compression="bz2", encoding="utf-8")

                os.remove(os.path.join(file_path, file_name))

                j = len(revisions)
                enlapsed = timer() - start
                logging.info("done {0} revision for file {1} in {2:.1f} minutes, projected: {3:.2f} hours".format(j, zip_file, enlapsed / 60, enlapsed / j * 72500000 / 60 / 60))
except Exception as e:
    logging.exception("***** An error ocurred *****")
finally:
    logging.info("***** Finished *****")