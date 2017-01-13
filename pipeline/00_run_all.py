# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:06:08 2017

@author: Rafael Crescenzi
"""

import subprocess

work_dir = "C:/Users/rcrescenzi/Documents/Personal/data/wsdm/WD/"

for proc in ["01_parse_xml.py", "02_pre_proc.py", "03_train_model.py"]:
    print("running", proc)
    subprocess.call(["python.exe", proc, "-dir", work_dir])
