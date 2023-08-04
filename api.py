import cv2
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException, File, UploadFile, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, StreamingResponse, FileResponse, HTMLResponse,PlainTextResponse
from typing import List
import base64
import tempfile
import pickle
import os
import time
# Import necessary libraries
import os
import json
import cv2
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
from tqdm import tqdm

app = FastAPI()

# =========================== {Face recog} ==========================================
# Threshold for cosine similarity used for face recognition
COSINE_THRESHOLD = 0.5


# Function to load the face detection and recognition models, and the user dictionary
def load_models():
    # Initialize models for face detection & recognition
    weights = os.path.join( "./models",
                            "face_detection_yunet_2023mar_int8.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    face_detector.setScoreThreshold(0.87)

    weights = os.path.join( "./models", "face_recognition_sface_2021dec_int8.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    return face_detector, face_recognizer

face_detector, face_recognizer = load_models()


# Function to match the given feature with the features in the dictionary using the recognizer
def match( feature1,dictionary):
    max_score = 0.0
    sim_user_id = ""
    # Loop through each user in the dictionary and find the one with the highest similarity score
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = face_recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    # Check if the maximum similarity score is above the threshold for authentication
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

# Function to recognize faces in the given image using the face detector and recognizer
def recognize_face(image,file_name = None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        # Detect faces in the image using the face detector
        dts = time.time()
        _, faces = face_detector.detect(image)
        # Check if the image contains a face (if file_name is provided, otherwise it's not required)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        print(f'time detection  = {time.time() - dts}')
        # Extract features for each detected face
        for face in faces:
            rts = time.time()

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            print(f'time recognition  = {time.time() - rts}')

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None


# ========================================================================================


# ===================================={Load}=================================================
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
        creds = ServiceAccountCredentials.from_json("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        # Open the Google Sheet by its name
        sheet = client.open(sheet_name).sheet1
        sheet = find_and_remove_duplicates(sheet)
        data = sheet.get_all_values()  # Get all the values from the sheet

        #num_rows = len(data)  # Get the number of rows in the sheet

        # Get all the values from the sheet
        data = sheet.get_all_records()
        output_dir = 'IMAGES'
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
            
        return dictionary

# ===========================================================================================




templates = Jinja2Templates(directory="templates")
attended_id = []
# Helper function to get the attended_id as a dependency
def get_attended_id():
    return attended_id

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def post_form(
    request: Request,
    type: str = Form(...),
):
    
    if type == "Real time":
        return RedirectResponse(url="/realtime", status_code=303)
    elif type == "From images":
        return RedirectResponse(url="/from_images", status_code=303)
    else:
        raise HTTPException(status_code=400, detail="Invalid type selected")

@app.get("/realtime", response_class=HTMLResponse)
async def get_realtime_page(request: Request):
    # Check if the file exists before showing the download button
    file_exists = os.path.exists("realtime_list.pkl")
    return templates.TemplateResponse("realtime.html", {"request": request, "file_exists": file_exists})

@app.post("/start_realtime", response_class=HTMLResponse)
async def start_realtime(
    request: Request,
    hours: int = Form(...),
    minutes: int = Form(...),
    seconds: int = Form(...),
    year: int = Form(...),
):
    
    if year != 2025:
        raise HTTPException(status_code=400, detail="Comming Soon")
    attended_id = []
    duration_in_seconds = hours * 3600 + minutes * 60 + seconds
    
    

    async def process_frames():
        dictionary = load_dict(year)
        file_name = 'realtime_list.pkl'
        video_capture = cv2.VideoCapture(0)
        start_time = time.time()

        while time.time() - start_time < duration_in_seconds:

            ret, frame = video_capture.read()
            print(f"time is : {time.time()}")
            fetures, faces = recognize_face(frame)
            for face, feature in zip(faces, fetures):
                result, user = match(feature,dictionary)
                box = list(map(int, face[:4]))
                color = (0, 255, 0) if result else (0, 0, 255)
                thickness = 2
                cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

                id_name, score = user if result else ("unknown", 0.0)
                text = "{0} ({1:.2f})".format(id_name, score)
                position = (box[0], box[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)
                if id_name not in attended_id and id_name != "unknown":
                    attended_id.append(id_name)

            _, encoded_frame = cv2.imencode('.png', frame)
            frame_base64 = base64.b64encode(encoded_frame).decode("utf-8")

            yield (
                b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n'
                + base64.b64decode(frame_base64.encode())
                + b'\r\n'
            )
        
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(attended_id, f)

        video_capture.release()

    return StreamingResponse(
        process_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Add a new route to display the real-time recognition results
@app.get("/real_time_results", response_class=HTMLResponse)
async def get_real_time_results(request: Request):
    # Load the template and render it
    return templates.TemplateResponse("real_time_results.html", {"request": request})



@app.get("/from_images", response_class=HTMLResponse)
async def get_from_images_page(request: Request):
    return templates.TemplateResponse("from_images.html", {"request": request})

@app.post("/start_from_images", response_class=HTMLResponse)
async def start_from_images(
    request: Request,
    images: List[UploadFile] = File(...),
    year: int = Form(...),
):
    file_name = 'from_images_id.pkl'
    if os.path.exists(file_name):
      os.remove(file_name)

    attended_id = []
    dictionary = load_dict(year)

    processed_images = []
    print("Year:", year)  # Print the year variable
    for image in images:
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        fetures, faces = recognize_face(img,dictionary)
        for idx, (face, feature) in enumerate(zip(faces, fetures)):
            result, user = match(feature,dictionary)
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)

            id_name, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id_name, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)
            if id_name not in attended_id and id_name != "unknown":
                attended_id.append(id_name)

        _, encoded_img = cv2.imencode('.png', img)
        preprocessed_image = encoded_img.tobytes()
        preprocessed_image_base64 = base64.b64encode(preprocessed_image).decode("utf-8")

        processed_images.append((preprocessed_image_base64, text))
    with open(file_name, 'wb') as f:
        pickle.dump(attended_id, f)
    num_images = len(processed_images)

    
    return templates.TemplateResponse("from_images_result.html", {"request": request, "num_images": num_images, "processed_images": processed_images})



@app.get("/download_from_images_id_names")
async def download_recognized_id_names():
    file_name = 'from_images_id.pkl'
    with open(file_name, 'rb') as f:
        attended_list = pickle.load(f)

    if not attended_list:
        raise HTTPException(status_code=404, detail="No attendance data available to download. Please take attendance first")

    # Convert attended_id elements to strings
    str_attended_id = [str(id) for id in attended_list]

    # Create a CSV file with the attended_id list
    csv_data = "id_name\n" + "\n".join(str_attended_id)

    # Save the CSV data to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    # Generate the CSV file as a response
    response = FileResponse(temp_file_path, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_list.csv"
    os.remove(file_name)
    return response

@app.get("/download_realtime_id_names")
async def download_realtime_id():
    file_name = 'realtime_list.pkl'
    with open(file_name, 'rb') as f:
        attended_list = pickle.load(f)
    
    if not attended_list:
        raise HTTPException(status_code=404, detail="No attendance data available to download. Please take attendance first")

    # Convert attended_id elements to strings
    str_attended_id = [str(id) for id in attended_list]

    # Create a CSV file with the attended_id list
    csv_data = "id_name\n" + "\n".join(str_attended_id)

    # Save the CSV data to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(csv_data)
        temp_file_path = temp_file.name

    # Generate the CSV file as a response
    response = FileResponse(temp_file_path, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_list.csv"
    os.remove(file_name)
    return response

# Handle "/robots.txt" request
@app.get("/robots.txt", response_class=PlainTextResponse)
async def get_robots_txt():
    return "User-agent: *\nDisallow: /"