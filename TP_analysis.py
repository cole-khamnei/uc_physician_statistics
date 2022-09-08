import os
import sys

import pandas as pd
import numpy as np
import gender_guesser.detector as gender

import constants

sys.path.append("../helpers")
import helpers

#######################################################################################################################
###### Gender Utils
#######################################################################################################################

gd = gender.Detector(case_sensitive=False)

GENDER_REMAP = {
    "female": "Women",
    "mostly_female": "Mostly Women",
    "andy": "Androgenous",
    "unknown": "Unknown",
    "mostly_male": "Mostly Men",
    "male": "Men",
}

GENDER_SORT_KEY = dict(zip(GENDER_REMAP.values(), range(len(GENDER_REMAP))))
gender_sort_function = np.vectorize(GENDER_SORT_KEY.get)

#######################################################################################################################
###### Functions
#######################################################################################################################


def load_physician_job_titles():
    """"""
    with open(constants.PHYSICIAN_JOB_TITLES_PATH) as file:
        return sorted(set([title.lower() for title in file.read().split("\n")]))


def clean_name(name: str) -> str:
    """"""
    name = name.lower()
    name = name.strip(", md").strip(", dr.")
    if "," in name:
        splits = name.split(",")
        if len(splits) > 2:
            return name
        assert len(splits) == 2, f"Should only have one , in name if any. {name}"
        name = splits[1].strip() + " " + splits[0].strip()
    return name


def guess_gender(name: str) -> str:
    """"""
    first_name = name.split(" ")[0]
    return GENDER_REMAP.get(gd.get_gender(first_name))


def load_salary_data(data_path: str) -> pd.DataFrame:
    """"""
    if isinstance(data_path, str):
        salary_data = pd.read_csv(data_path, low_memory=False)
        salary_data.columns = [col.lower().strip().replace(" ", "_") for col in salary_data.columns]
        salary_data["pension_debt"] = 0
        salary_data.drop(columns=["pension_debt", "notes", "status", "agency"], inplace=True)
        salary_data.query('base_pay > 50_000', inplace=True)
        salary_data["employee_name"] = salary_data["employee_name"].apply(clean_name)
        salary_data.query("employee_name != 'not provided'", inplace=True)
        salary_data["gender"] = salary_data["employee_name"].apply(guess_gender)
        salary_data["job_title"] = salary_data["job_title"].str.lower()
        return salary_data
    
    return pd.concat([load_salary_data(data_path_i) for data_path_i in data_path])


def load_physician_salary_data(overwrite: bool = False) -> pd.DataFrame:
    """"""
    if os.path.exists(constants.PHYSICIAN_SALARY_DATA_PATH) and not overwrite:
        physician_salary_data = pd.read_csv(constants.PHYSICIAN_SALARY_DATA_PATH)
    else:
        all_salary_data = load_salary_data(data_paths)
        physician_salary_data = all_salary_data.loc[all_salary_data["job_title"].isin(physician_job_titles)]
        physician_salary_data.to_csv(constants.PHYSICIAN_SALARY_DATA_PATH, index=False)
    
    return physician_salary_data


#######################################################################################################################
###### End
#######################################################################################################################
