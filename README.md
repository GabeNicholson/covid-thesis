
### Reproducible material for my thesis "Is the News Always Negative?"

Steps to reproduce analysis notebook for figures and tables:

(1) Create a new virtual environment with python 3.10.10 and activate it.

(2) run `pip install -r requirements.txt`. 

(3) run `python cache_data_files.py` which will create the parquet files necessary to load into the notebook.

(4) Open  [covid_data_analysis.ipynb](covid_data_analysis.ipynb) and make sure to use the same kernel that you just created. 

