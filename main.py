# import the inference-sdk
from inference_sdk import InferenceHTTPClient
import os, sys, time
import supervision as sv
import cv2
from roboflow import Roboflow
# from ultralytics import YOLO

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="yRhi0p59H1CSbgYgBwIL"
)

def visualize(results, image):
    detections = sv.Detections.from_inference(results)

    # create supervision annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # display the image
    sv.plot_image(annotated_image)

def test_openspace2():
    #loop images
    for testImage in os.listdir('test_images'):
        #create path and load
        imPath = os.path.join('test_images', testImage)
        image = cv2.imread(imPath)
        # infer on a local image
        results = CLIENT.infer(imPath, model_id="openspace2/2")

        # create license plate images
        bbox = results['predictions']
        for b in bbox:
            if b['class'] != 'licence':
                continue
            else:
                x,y,width,height = [round(c) for c in [b['x'], b['y'], b['width'], b['height']]]
                licenseImage = image[y-round(height/2):y+round(height/2), x-round(width/2):x+round(width/2)]
                try:os.mkdir('license_images')
                except:pass
                cv2.imwrite(f'license_images\\{testImage}_license.jpg', licenseImage)

        visualize(results,image)
        #initializing variables
        badgePred = 'None'
        badgeConf = 0
        predictions = results['predictions']
        #loop through preds to identify the badge prediction and license prediction
        for p in predictions:
            if p['class_id'] not in [15,23]:
                badgePred = p['class']
                badgeConf = p['confidence']
            elif p['class_id']==15:
                licensePred = p['class']
                licenseConf = p['confidence']

        #print results
        print(f'FOR IMAGE: {testImage} \n'
              f'Make: {badgePred} || Confidence: {round(badgeConf*100)}% \n'
              f'License: Detected || Confidence: {round(licenseConf*100)}%\n\n')


def test_openspace3():
    #loop images
    for testImage in os.listdir('license_images'):
        #create path and load
        imPath = os.path.join('license_images', testImage)
        image = cv2.imread(imPath)
        # infer on a local image
        results = CLIENT.infer(imPath, model_id="openspace3-license-plate/1")
        visualize(results,image)
        # initializing variables
        badgePred = 'None'
        badgeConf = 0
        predictions = results['predictions']
        #loop through preds to identify the badge prediction and license prediction
        for p in predictions:
            print(p)




