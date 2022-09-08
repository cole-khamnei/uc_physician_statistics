import requests
import time

import pandas as pd

from bs4 import BeautifulSoup as soup

import constants

#######################################################################################################################
###### Unknown
#######################################################################################################################

def name_token_search(salary_data: pd.DataFrame, name: str):
    """"""
    found_people = salary_data.loc[salary_data["employee_name"].apply(lambda s: search_token in s.lower())]

#######################################################################################################################
###### Unknown
#######################################################################################################################

def find_provider_salaries(provider_data: pd.DataFrame, salary_data: pd.DataFrame) -> pd.DataFrame:
    """"""
    found_providers, name_data = [], []
    for i, provider_data_row in provider_data.iterrows():
        provider_name = provider_data_row["name"]
        last_name = provider_name.split(" ")[-1]
        found_people = salary_data.loc[salary_data["employee_name"].apply(lambda s: s.endswith(last_name))]

        first_name = provider_name.split(" ")[0]
        found_people = found_people.loc[found_people["employee_name"].apply(lambda s: s.startswith(first_name))]

        if len(found_people) > 1:
            exact_matches = found_people["employee_name"] == provider_name.replace(".", "")
            if any(exact_matches):
                found_people = found_people.loc[exact_matches]

        if len(found_people) > 1:
            pass
#             print(f"More than one listing found for name: {provider_name}")
        elif len(found_people) == 0:
            pass
#             print(f"No listing found for name: {provider_name}")
        else:
            found_providers.append(found_people)
            name_data.append(provider_data_row)

    found_providers = pd.concat(found_providers)
    name_data = pd.concat(name_data, axis=1).transpose()

    for col in name_data.columns:
        found_providers[col] = name_data[col].values
    return found_providers


def clean_degree(degree: str) -> str:
    """"""
    return degree.replace(".", "").split("\n", maxsplit=1)[0]


def get_providers(url: str) -> pd.DataFrame:
    """  Scrapes a department webpage and finds all physicians"""
    if "{page_number}" in url or "{page_letter}" in url:
        provider_data = []
        numerical_index = "{page_number}" in url
        iterator = range(1, 20) if numerical_index else "abcdefghijklmnopqrstuvwxyz"
        for index in iterator:
            indexed_url = url.format(page_number=index) if numerical_index else url.format(page_letter=index)
            new_provider_data = get_providers(indexed_url)
            provider_data.append(new_provider_data)
            if len(new_provider_data) == 0:
                break
            time.sleep(1)
        return pd.concat(provider_data).reset_index()
    
    else:
        req = requests.get(url, headers=constants.REQUEST_HEADERS)
        html_data = soup(req.content, 'html.parser')

        provider_data = []
        for tag in html_data.find_all("a") + html_data.find_all("h2"):
            tag_text = tag.get_text().strip().replace("M.D.", "MD").replace("Ph.D.", "PhD")
            tag_text = tag_text.replace(", Jr.", "Jr.").replace("M.S.", "MS")
            if len(tag_text) > 50 or "&" in tag_text:
                continue
            if "MD" in tag_text or "DO" in tag_text:
                if "," not in tag_text:
                    tag_text = tag_text.replace("MD", ", MD")
                provider_name, degree = tag_text.split(",", maxsplit=1)
                degree = degree.strip().strip(",")
                if provider_name.endswith(" MD"):
                    provider_name = provider_name.strip(" MD").strip()
                    degree = "MD, " + degree
                provider_data.append([provider_name.strip(","), clean_degree(degree)])

        return pd.DataFrame(provider_data, columns=["name", "degree"])


def get_department_providers(department: str) -> pd.DataFrame:
    """"""

    assert department in UCSF_DEPARTMENT_URLS, f"{department} is not a valid department."
    url = UCSF_DEPARTMENT_URLS[department]
    if isinstance(url, list):
        provider_data = pd.concat([get_providers(url_i) for url_i in url])
        provider_data["department"] = department
    else:
        provider_data = get_providers(url)
        provider_data["department"] = department
    
    return provider_data