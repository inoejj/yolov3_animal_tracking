from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-092--loss-4.340.h5")
detector.setJsonPath(r"Z:\DeepLabCut\misc\tensorflow\ImageAI\project\data\json\detection_config.json") # download via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json
detector.loadModel()

fileNames = os.listdir(r"Z:\DeepLabCut\misc\tensorflow\ImageAI\project\frames\Video8")
outputFolder = (r"Z:\DeepLabCut\misc\tensorflow\ImageAI\project\frames\output2")
inputPath = r"Z:\DeepLabCut\misc\tensorflow\ImageAI\project\frames\Video8"

##OUTPUT NEW IMAGES
for i in fileNames:
    inputImage = os.path.join(inputPath, i)
    img = cv2.imread(inputImage)
    outputFilenamePath = os.path.join(outputFolder, i)
    returned_image, detections = detector.detectObjectsFromImage(input_image=inputImage, output_type="array", minimum_percentage_probability=55)
    detections = detections[0:2]
    for i in detections:
        labelText = i["name"]
        if labelText == "White_mouse":
            color = (255, 255, 255)
        if labelText == "Black_mouse":
            color = (1, 173, 225)
        labelProb = i["percentage_probability"]
        labelBox = i["box_points"]
        cv2.rectangle(img, (labelBox[0], labelBox[1]), (labelBox[2], labelBox[3]), color, 2)
        cv2.putText(img, labelText, (labelBox[0], labelBox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, str(round(labelProb,2)), (labelBox[0], labelBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(outputFilenamePath,img)
    print(outputFilenamePath)





"""
EXAMPLE RESULT

hololens  :  39.69653248786926  :  [611, 74, 751, 154]
hololens  :  87.6643180847168  :  [23, 46, 90, 79]
hololens  :  89.25175070762634  :  [191, 66, 243, 95]
hololens  :  64.49641585350037  :  [437, 81, 514, 133]
hololens  :  91.78624749183655  :  [380, 113, 423, 138]

"""