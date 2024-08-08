# import the inference-sdk
from inference_sdk import InferenceHTTPClient
import os, sys, time
import supervision as sv
import cv2

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

def main():
    for testImage in os.listdir('test_images'):
        imPath = os.path.join('test_images', testImage)
        image = cv2.imread(imPath)
        # infer on a local image
        results = CLIENT.infer(imPath, model_id="openspace2/2")
        visualize(results,image)

        badgePred = 'None'
        badgeConf = 0
        predictions = results['predictions']
        for p in predictions:
            if p['class_id'] not in [15,23]:
                badgePred = p['class']
                badgeConf = p['confidence']
            elif p['class_id']==15:
                licensePred = p['class']
                licenseConf = p['confidence']
        print(f'FOR IMAGE: {testImage} \n'
              f'Make: {badgePred} || Confidence: {round(badgeConf*100)}% \n'
              f'License: Detected || Confidence: {round(licenseConf*100)}%\n\n')

main()