import os
import glob

#######################################################################################################################
###### basics
#######################################################################################################################

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}

#######################################################################################################################
###### Data Paths
#######################################################################################################################

DATA_DIR = "transparent_california_data"
DATA_PATHS = glob.glob(os.path.join(DATA_DIR, "university-of-california-*.csv"))
PHYSICIAN_SALARY_DATA_PATH = os.path.join(DATA_DIR, "physcian_salaries.csv")
PHYSICIAN_JOB_TITLES_PATH = os.path.join(DATA_DIR, "physician_job_titles.txt")	

#######################################################################################################################
###### webscraping paths
#######################################################################################################################

UCSF_DEPARTMENT_URLS = {
    "neurosurgery": "https://neurosurgery.ucsf.edu/faculty",
    "cardiology": "https://ucsfhealthcardiology.ucsf.edu/people",
    "ophthalmology": "https://ophthalmology.ucsf.edu/all-faculty/",
    "hematology / oncology": "https://ucsfhealthhemonc.ucsf.edu/people?combine=&page={page_number}",
    "otolaryngology": "https://ohns.ucsf.edu/people",
    "pediatrics": "https://pediatrics.ucsf.edu/pediatrics/faculty/{page_letter}",
    "nephrology": "https://nephrology.ucsf.edu/people?combine=&page={page_number}",
    "gastroenterology": "https://gastroenterology.ucsf.edu/people?combine=&page={page_number}",
    "psychiatry": "https://psychiatry.ucsf.edu/faculty",
    "neurology": "https://neurology.ucsf.edu/faculty",
    "emergency medicine": "https://emergency.ucsf.edu/people",
    "pulmonology": "https://pulmonary.ucsf.edu/people?combine=&page={page_number}",
    "infectious diseases": "https://infectiousdiseases.ucsf.edu/people?combine=&page={page_number}",
    "anesthesiology": "https://anesthesia.ucsf.edu/people/faculty?page={page_number}",
    "dermatology": "http://www.dermatology.ucsf.edu/adult-dermatology-1",
    "pathology": "https://pathology.ucsf.edu/about/faculty/faculty-directory",
    "family medicine": "https://fcm.ucsf.edu/people_bio?page={page_number}",
    "palliative care": "https://palliativemedicine.ucsf.edu/people",
    "geriatrics": "https://geriatrics.ucsf.edu/people",
    "endocrinology": "https://endocrine.ucsf.edu/faculty",
    "rheumatology": "https://rheumatology.ucsf.edu/people",
    "radiology": "https://radiology.ucsf.edu/people?type=171",
    "obstetrics / gynecology": "https://obgyn.ucsf.edu/our-faculty-members-by-last-name",
    "general surgery": "https://generalsurgery.ucsf.edu/faculty.aspx",
    "cardiothoracic surgery": "https://adultctsurgery.ucsf.edu/our-team.aspx",
    "vascular surgery": "https://vascularsurgery.ucsf.edu/meet-the-team.aspx",
    "radiation oncology": "https://radonc.ucsf.edu/about/our-team/medical-faculty/",
    "orthopedic surgery": "https://orthosurgery.ucsf.edu/patient-care/faculty?field_speciality_target_id=All&location=All&items_per_page=All&page=0",
    "internal medicine": ["https://ucsfhealthdgim.ucsf.edu/people?combine=&page={page_number}", 
                         "https://ucsfhealthhospitalmedicine.ucsf.edu/people?combine=&page={page_number}"],
    "obgyn-MFM": "https://obgyn.ucsf.edu/maternal-fetal-medicine/our-experts",
    "pediatrics-ID": "https://pediatrics.ucsf.edu/1619816/infectious-diseases-and-global-health-faculty?combine=&page={page_number}",
    "surgery-trauma": "https://zsfgsurgery.ucsf.edu/meet-the-team.aspx",
    "pediatrics-EM": "https://pediatrics.ucsf.edu/1619786/emergency-medicine-faculty?combine=&page={page_number}"
}

UCLA_DEPARTMENT_URLS = {
    "neurosurgery": "https://www.uclahealth.org/departments/neurosurgery/our-expert-team",
    "cardiology": "https://www.uclahealth.org/heart/adult-cardiology-team",
    "ophthalmology": "https://www.uclahealth.org/eye/our-providers",
    "hematology / oncology": "https://hemonc.med.ucla.edu/pages/whoweare",
    "otolaryngology": "https://www.uclahealth.org/departments/head-neck-surgery/our-expert-team",
    "pediatrics": "https://www.uclahealth.org/medical-services/general-pediatrics/our-physicians",
    "nephrology": None,
    "gastroenterology": "https://www.uclahealth.org/gastro/our-expert-team",
    "psychiatry": "https://www.uclahealth.org/locations/psychiatry",
    "neurology": "https://www.uclahealth.org/departments/neurology/faculty-staff/all-faculty-staff",
    "emergency medicine": "https://www.uclahealth.org/emergency-medicine/faculty-directory",
    "pulmonology": "https://www.uclahealth.org/pulmonary/our-team",
    "infectious diseases": "https://www.uclahealth.org/infectious-diseases/physicians-and-faculty",
    "anesthesiology": "https://www.uclahealth.org/departments/anes/our-physicians/faculty-physicians",
    "dermatology": "https://www.uclahealth.org/dermatology/our-expert-team",
    "pathology": "https://www.uclahealth.org/departments/pathology/meet-our-faculty/primary-faculty",
    "family medicine": "https://www.uclahealth.org/family-medicine/faculty-1114",
    "palliative care": "https://www.uclahealth.org/palliative-care/palliative-care-team",
    "geriatrics": "https://www.uclahealth.org/geriatrics/meet-our-team",
    "endocrinology": None,
    "rheumatology": "https://www.uclahealth.org/rheumatology/our-doctors",
    "radiology": "https://www.uclahealth.org/radiology/our-faculty",
    "obstetrics / gynecology": "https://www.uclahealth.org/obgyn/meet-our-expert-team",
    "general surgery": "https://www.uclahealth.org/medical-services/surgery/general-surgery/expert-team",
    "cardiothoracic surgery": "https://www.uclahealth.org/departments/surgery/divisions/cardiac-surgery",
    "vascular surgery": "https://www.uclahealth.org/departments/surgery/education/residency/vascular-surgery-residency-program/faculty",
    "radiation oncology": "https://www.uclahealth.org/departments/radonc/our-expert-team",
    "orthopedic surgery": "https://www.uclahealth.org/ortho/our-expert-team",
    "internal medicine": None
}

#######################################################################################################################
###### End
#######################################################################################################################