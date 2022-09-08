import os
import glob

#######################################################################################################################
###### Data Paths
#######################################################################################################################

DATA_DIR = "transparent_california_data"
DATA_PATHS = glob.glob(os.path.join(DATA_DIR, "university-of-california-*.csv"))
PHYSICIAN_SALARY_DATA_PATH = os.path.join(DATA_DIR, "physcian_salaries.csv")
PHYSICIAN_JOB_TITLES_PATH = os.path.join(DATA_DIR, "physician_job_titles.txt")	

#######################################################################################################################
###### End
#######################################################################################################################
