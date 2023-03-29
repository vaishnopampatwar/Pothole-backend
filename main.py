from fastapi import FastAPI, File, UploadFile, Response, Header
from pydantic import BaseModel
import numpy as np
import cv2
import werkzeug
from fastapi.middleware.cors import CORSMiddleware
import base64
import aiofiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

@app.post("/calculate-area")
async def calculateArea(file: UploadFile = File(...)):
    imagefile = file
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    # imagefile.save(filename)
    async with aiofiles.open(filename, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    area = 0
    yolo = cv2.dnn.readNet(
        "models/yolov4_tiny_pothole.cfg",
        "models/yolov4_tiny_pothole_last.weights")
    classes = []
    with open("pothole.names", 'r') as f:
        classes = f.read().splitlines()
    # print(len(classes))
    img = cv2.imread(imagefile.filename)
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    # print(blob.shape)/
    i = blob[0].reshape(320, 320, 3)
    yolo.setInput(blob)
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layer_name)
    boxs = []
    confidences = []
    class_ids = []
    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxs.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # print("box",len(boxs))
    if len(boxs) != 0:
        theRequiredAreas = []
        indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxs), 3))
        for i in indexes.flatten():
            x, y, w, h = boxs[i]
            print("x " + str(x) + " y " + str(y) + " w " + str(w) + " h " + str(h) + "area " + str(w * h))
            area = w * h
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label + " ", (x, y + 20), font, 2, (255, 255, 255), 1)
            area *= 0.0264583333
            print(w)
            print(h)
            theRequiredAreas.append(area)
        print(max(theRequiredAreas))
        return str(max(theRequiredAreas))
    else:
        return "1"

class StringPayload(BaseModel):
    string: str
@app.post("/")
async def calculateArea(image: StringPayload):
    decoded_data = base64.b64decode(image.string)
    with open("pothole.png", "wb") as fh:
        fh.write(decoded_data)
    img = cv2.imread('pothole.png')
    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    # print("\nReceived image File name : " + imagefile.filename)
    # imagefile.save(filename)
    area = 0
    yolo = cv2.dnn.readNet(
        "models/yolov4_tiny_pothole.cfg",
        "models/yolov4_tiny_pothole_last.weights")
    classes = []
    with open("pothole.names", 'r') as f:
        classes = f.read().splitlines()
    # print(len(classes))
    # img = cv2.imread(imagefile.filename)
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    # print(blob.shape)/
    i = blob[0].reshape(320, 320, 3)
    yolo.setInput(blob)
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layer_name)
    boxs = []
    confidences = []
    class_ids = []
    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxs.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # print("box",len(boxs))
    if len(boxs) != 0:
        theRequiredAreas = []
        indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxs), 3))
        for i in indexes.flatten():
            x, y, w, h = boxs[i]
            print("x " + str(x) + " y " + str(y) + " w " + str(w) + " h " + str(h) + "area " + str(w * h))
            area = w * h
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label + " ", (x, y + 20), font, 2, (255, 255, 255), 1)
            area *= 0.0264583333
            print(w)
            print(h)
            theRequiredAreas.append(area)
        print(max(theRequiredAreas))
        return str(max(theRequiredAreas))
    else:
        return "1"

# if __name__ == '__main__':
#     uvicorn.run(  app, port=5000)
