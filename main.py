import numpy as np
import cv2
import imutils
import io
from io import BytesIO
from PIL import Image, ImageDraw
from flask import Flask,jsonify,request,send_file
import json
import os
from dotenv import load_dotenv

# set-up yoloV3 model
weightsPath = 'models/yolov3.weights'
configPath = 'models/yolov3.cfg'

# load yolo model
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load labels
LABELS = open('models/coco.names').read().strip().split("\n")

# detect bounding box corrdinates 
def detect_bounding_box(frame):
    Min_Confidence= 0.3
    NMS_Threshold= 0.3

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def detect_people(frame, net, ln, personIdx=0):

        (Height, Width) = frame.shape[:2]
        results = []
        #Constructing a blob from the input frame and performing a forward pass of the YOLO object detector
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        centroids = []
        confidences = []

        #Looping over each of the layer outputs
        for output in layerOutputs:
        #Looping over each of the detections
            for detection in output:
                #Extracting the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                #Filtering detections by:
                #1 Ensuring that the object detected was a person
                #2 Minimum confidence is met
                if classID == personIdx and confidence > Min_Confidence:
                    #Scaling the bounding box coordinates back relative to the size of the image
                    box = detection[0:4] * np.array([Width, Height, Width, Height])
                    (centerX, centerY, width, height) = box.astype("int")
                    # center.append([centerX, centerY])
                    #Using the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    xmin = int(centerX - (width / 2))
                    xmax = int(centerX + (width / 2))
                    ymin = int(centerY - (height / 2))
                    ymax = int(centerY + (height / 2))
                    #Updating the list of bounding box coordinates, centroids, and confidences
                    boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    boxes = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
    return boxes

def create_heatmap(heatmap, bbox_list):
    for i in range(len(bbox_list)):
        intensity = 1
        for j in range(len(bbox_list)):
            if (bbox_list[i][0] < bbox_list[j][2] and 
                bbox_list[j][0] < bbox_list[i][2] and
                bbox_list[i][1] < bbox_list[j][3] and 
                bbox_list[j][1] < bbox_list[i][3]):
                
                intensity += 1

        heatmap[int(bbox_list[i][1]):int(bbox_list[i][3]), int(bbox_list[i][0]):int(bbox_list[i][2])] += min(intensity*40, 254)   #top left, bottom right 
        
    return heatmap

# detect crowd and generate heatmap with respect to density
def crowd_density(im, box_list):
    heatmap =  np.zeros_like(im[:,:,0]).astype(np.float)
    heatmap = create_heatmap(heatmap, box_list)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    heatmap = cv2.blur(heatmap, (30,30))
    final = cv2.addWeighted(heatmap, 1.0, im, 0.5, 0)
    return heatmap, final
    
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

@app.route('/heatmap_generation',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        img_params =request.files['image'].read()
        npimg = np.fromstring(img_params, np.uint8)
        #load image
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # load image
        box_list =  detect_bounding_box(frame)
        # apply heatmap
        heatmap, final = crowd_density(frame, box_list)

        I = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/PNG') 
    return output


if __name__ == '__main__':
    app.run()                       
                


