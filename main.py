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

from utils.Face_Recognition import recognize_face, match, get_attendance_list
from utils.load import load_dict

app = FastAPI()
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