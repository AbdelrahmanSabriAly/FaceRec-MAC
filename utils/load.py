# Import necessary libraries
import os
import json
import cv2
import pandas as pd
import numpy as np
import pickle
from datetime import date
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
from tqdm import tqdm

from utils.Face_Recognition import recognize_face


def get_keys(path):
    with open(path) as f:
        return json.load(f)
    

def create_keyfile_dict():
    keys = get_keys(".secret\\attendance-monitoring-393.json")

    variables_keys = {
        "type": keys['type'],
        "project_id": keys['project_id'],
        "private_key_id": keys['private_key_id'],
        "private_key": keys['private_key'],
        "client_email": keys['client_email'],
        "client_id": keys['client_id'],
        "auth_uri": keys['auth_uri'],
        "token_uri": keys['token_uri'],
        "auth_provider_x509_cert_url": keys['auth_provider_x509_cert_url'],
        "client_x509_cert_url": keys['client_x509_cert_url'],
        "universe_domain":keys['universe_domain']
    }
    return variables_keys

# Google Sheet details
SHEET_NAME = 'Your Sheet Name'
ID_COLUMN_NAME = 'ID'
TIMESTAMP_COLUMN_NAME = 'Timestamp'
dictionary = {}



def find_and_remove_duplicates(sheet):
    all_records = sheet.get_all_records()
    ids_to_records = {}
    rows_to_delete = []

    # Find the latest entry for each ID
    for idx, record in enumerate(all_records):
        current_id = record[ID_COLUMN_NAME]
        if current_id in ids_to_records:
            existing_record = ids_to_records[current_id]
            if existing_record[TIMESTAMP_COLUMN_NAME] < record[TIMESTAMP_COLUMN_NAME]:
                rows_to_delete.append(existing_record['row'])
                ids_to_records[current_id] = {'row': idx + 2, TIMESTAMP_COLUMN_NAME: record[TIMESTAMP_COLUMN_NAME]}
            else:
                rows_to_delete.append(idx + 2)
        else:
            ids_to_records[current_id] = {'row': idx + 2, TIMESTAMP_COLUMN_NAME: record[TIMESTAMP_COLUMN_NAME]}

    # Delete rows with older timestamps
    if rows_to_delete:
        for row_num in reversed(rows_to_delete):
            sheet.delete_row(row_num)

    return sheet

def download_image_from_drive(url, output_dir):
    output_path = os.path.join(output_dir, '')
    gdown.download(url, output_path, quiet=False)
    output =os.path.join(output_dir, os.listdir(output_dir)[0])
    return output

def load_dict(year):
    if year == 2025:
        file_name = f"{year}_data.pkl"
        if os.path.exists(file_name):
            os.remove(file_name)

        student_data = []

        sheet_name = "Attendance monitoring (Responses)"

        # Authorize with Google Sheets API using credentials
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(create_keyfile_dict(), scope)
        client = gspread.authorize(creds)
        # Open the Google Sheet by its name
        sheet = client.open(sheet_name).sheet1
        sheet = find_and_remove_duplicates(sheet)
        data = sheet.get_all_values()  # Get all the values from the sheet

        num_rows = len(data)  # Get the number of rows in the sheet

        # Get all the values from the sheet
        data = sheet.get_all_records()
        output_dir = 'IMAGES'
        counter = 0
        # Process the data
        for row in tqdm(data):
            timestamp = row['Timestamp']
            id = row['ID']
            name_in_arabic = row['Name in Arabic']
            image_url = row['Image']
            image_url = image_url.replace("open","uc")

            output = download_image_from_drive(image_url, output_dir)
            image = cv2.imread(output)
            # Process the image using face recognition functions
            feats, faces = recognize_face(image)

            if faces is None:
                continue

            # Extract user_id from the uploaded file's name
            dictionary[id] = feats[0]
            student_data.append({"Name": name_in_arabic, "ID": id})

            os.remove(output)
            print(100*(float(counter)/num_rows))
            counter+=(1/num_rows)
            
            
        return dictionary
