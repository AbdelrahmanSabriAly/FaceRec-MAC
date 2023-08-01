import base64
import cv2
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
import numpy as np
from utils.face_recognition import load_models_and_images, recognize_face, match

# Load face detection, recognition models, and user dictionary
face_detector, face_recognizer, dictionary = load_models_and_images()

# List of valid IDs from the user dictionary
valid_ids = str(dictionary.keys())

# Create a FastAPI instance
app = FastAPI()

# Use Jinja2 templates to serve the 'index.html'
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=Response)
async def login_page(request: Request):
    # Serve the login page using the 'login.html' template
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/process_login")
async def process_login(login_data: dict):
    # Extract the name and ID from the JSON payload
    name = login_data.get("name")
    entered_id = login_data.get("id")
    print(entered_id, name)

    # Check if the entered ID is valid
    if entered_id in valid_ids:
        # Redirect to the camera page with the entered name and ID as query parameters
        return {"redirect_url": f"/camera?name={name}&id={entered_id}"}
    else:
        # Return an error message to the frontend
        return {"error": "Invalid ID"}


@app.post("/process_selfie")
async def process_selfie(selfie_data: dict):
    # Extract the base64-encoded selfie data from the JSON payload
    selfie_data_url = selfie_data.get("selfieDataUrl")

    # Extract the base64-encoded image data part (remove the "data:image/png;base64," prefix)
    image_data = selfie_data_url.split(",")[1]

    # Decode the base64-encoded image data into bytes
    image_bytes = base64.b64decode(image_data)

    # Convert the bytes data to a NumPy array
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image using cv2
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Recognize faces in the image and perform face matching
    fetures, faces = recognize_face(image, face_detector, face_recognizer)
    if len(faces) == 0:
        # No faces detected, return a message to the frontend
        return {"message": "No faces detected in the image!"}

    for idx, (face, feature) in enumerate(zip(faces, fetures)):
        # Perform face matching and get the result and user information
        result, user = match(face_recognizer, feature, dictionary)

        # Draw a rectangle around the recognized face and display user information
        box = list(map(int, face[:4]))
        color = (0, 255, 0) if result else (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
        id_name, score = user if result else ("unknown", 0.0)
        text = "{0} ({1:.2f})".format(id_name, score)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        cv2.putText(image, text, position, font, scale,
                    color, thickness, cv2.LINE_AA)

    # Encode the processed image back to base64-encoded data URL
    _, processed_image_bytes = cv2.imencode(".png", image)
    processed_data_url = f"data:image/png;base64,{base64.b64encode(processed_image_bytes).decode()}"

    # Return the processed image data URL to the frontend
    return {"processedDataUrl": processed_data_url}


@app.get("/camera", response_class=Response)
async def camera_page(request: Request, name: str = None, id: str = None):
    # This route handles the camera page.
    # If either 'name' or 'id' is missing, redirect to the login page.
    if not name or not id:
        return templates.TemplateResponse("login.html", {"request": request})

    # Check if the entered ID is valid
    if id in valid_ids:
        # Serve the camera page using the 'camera.html' template
        return templates.TemplateResponse(
            "camera.html",
            {"request": request, "name": name, "id": id}
        )
    else:
        # If the entered ID is invalid, show an error message on the login page
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_message": "Invalid ID"}
        )
