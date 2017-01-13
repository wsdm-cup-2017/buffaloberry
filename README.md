# buffaloberry
The Buffaloberry Vandalism Detector

This repository contains the code for the Winning Solution to the WSDM Cup task of Vandalism Detection. http://www.wsdm-cup-2017.org/vandalism-detection.html

Dependencies:

- Python 3.4 (https://www.python.org/)
- nltk 3.2.1 (http://www.nltk.org/)
- fuzzywuzzy  0.14.0 (https://github.com/seatgeek/fuzzywuzzy)
- scipy 0.18.1 (https://www.scipy.org/)
- langid 1.1.6 (https://github.com/saffsd/langid.py)
- numpy 1.11.2 (http://www.numpy.org/)
- pandas 0.19.1 (http://pandas.pydata.org/)
- joblib (https://pythonhosted.org/joblib/)
- xgboost 0.6 (http://xgboost.readthedocs.io/)
- scikit-learn 0.18 (http://scikit-learn.org/)
- dask 0.12.0 (http://dask.pydata.org/en/latest/)
- python_Levenshtein 0.12.0 (optional for performance, https://pypi.python.org/pypi/python-Levenshtein/0.12.0)
- 7zip (http://www.7-zip.org/, the executable 7za must be in the path or in the pipeline directory)


The recommended way to run the entire pipeline is to use WinPython 3.4.4.5Qt5 (https://winpython.github.io/), a portable distribution of python that does not interfere with other Python installations and comes packed with some of the most useful data science libraries. Only fuzzywuzzy (optionally with python Levenshtein), langid and xgboost, need to be installed on top of WinPython for this program to run on Windows.

## Building the program

For building the client, the scripts expect the following directory structure: 

```
working directory 
└─── raw
│   │   wdvc16_2012_10.xml.7z
│   │   wdvc16_2012_11.xml.7z
│   │   wdvc16_2013_01.xml.7z
│   │   wdvc16_2013_03.xml.7z
│   │   wdvc16_2013_05.xml.7z
│   │   wdvc16_2013_07.xml.7z
│   │   wdvc16_2013_09.xml.7z
│   │   wdvc16_2013_11.xml.7z
│   │   wdvc16_2014_01.xml.7z
│   │   wdvc16_2014_03.xml.7z
│   │   wdvc16_2014_05.xml.7z
│   │   wdvc16_2014_07.xml.7z
│   │   wdvc16_2014_09.xml.7z
│   │   wdvc16_2014_11.xml.7z
│   │   wdvc16_2015_01.xml.7z
│   │   wdvc16_2015_03.xml.7z
│   │   wdvc16_2015_05.xml.7z
│   │   wdvc16_2015_07.xml.7z
│   │   wdvc16_2015_09.xml.7z
│   │   wdvc16_2015_11.xml.7z
│   │   wdvc16_2016_01.xml.7z
│   │   wdvc16_2016_03.xml.7z
└─── truth
│   │   wdvc16_truth.csv
│   │   wdvc16_2016_03_truth.csv
└─── meta
    │   wdvc16_meta.csv
    │   wdvc16_2016_03_meta.csv
```

Files in the "raw" directory are expected to be zipped, they are extracted on the fly, processed and then the unzipped file is deleted. A side effect of this is that the script will run if the files are unzipped, but will delete them after processing. Files in the truth and meta directories are expected to be unzipped, as they are much smaller. 

The program will create additional directories within "working directory" for storing data files and the final production client with all it's required files. 

The following commands needs to be run in the "pipeline" directory in order to build the production client

### Feature Extractor
```
python 01_parse_xml.py -dir <working_directory>
```
### Feature Extractor
```
python 02_pre_proc.py -dir <working_directory>
```
### Training

```
python 03_train_model.py -dir <working_directory>
```

### Alternative: run all

There is 00_run_all.py file that runs all three scripts consecutively, this files does not accept command line parameters, it must be edited and the variable "work_dir" must be changed to the working directory as stated before. After that just run

```
python 00_run_all.py
```

##  Testing / Production

To run the final production client, go into the production folder that was created in the "working directory" and run:
```
python Client.py -d <HOST_NAME:PORT> -a <AUTHENTICATION_TOKEN>
```
Where:
HOST_NAME:PORT: the host name and port of the testing server that sends the revisions in xml format and the metadata in cvs format.
AUTHENTICATION_TOKEN: a token to identify the client in the server