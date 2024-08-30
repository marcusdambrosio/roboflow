import datetime as dt
import os
import sys
import json
import cv2
import shutil
def create_classes(classes):
    allClasses = []
    classMap = {}
    for id, c in enumerate(classes):
        if len(c)>1:
            continue
        classMap[str(c)] = id
        allClasses.append({
            'id':id,
            'name': str(c),
            'supercategory': 'none'
        })
    return classMap, allClasses

def create_images_annotations(imagePaths, classMap):
    allImgs = []
    allAnno = []
    for id, img in enumerate(imagePaths):
        currImg = cv2.imread(img)
        currClass = classMap[str(img.split('\\')[-2])]
        height, width, channel = currImg.shape

        allImgs.append({
            "id": id,
            "license": 1,
            "file_name": img.split('\\')[-1],
            "height": height,
            "width": width,
            "date_captured": str(dt.datetime.now())
        })

        allAnno.append({
            "id": id,
            "image_id": id,
            "category_id": currClass,
            "bbox": [
                0,
                0,
                width,
                height
            ],
            "area": height*width,
            "segmentation": [],
            "iscrowd": 0
        })

    return allImgs, allAnno


def create_coco_json(classes, imagePaths):
    #get data
    classMap, allClasses = create_classes(classes)
    allImgs, allAnno = create_images_annotations(imagePaths, classMap)

    dat = {
    "info": {
        "year": "2024",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Marcus DAmbrosio",
        "url": "TBD",
        "date_created": str(dt.datetime.now())
    },
    "licenses": [
        {
            "id": 1,
            "url": "TBD",
            "name": "Public Domain"
        }
    ],
    "categories": allClasses
        ,
    "images": allImgs
        ,
    "annotations": allAnno
    }

    #SAVE
    imageDir = imagePaths[0].split('\\')[0]
    with open(os.path.join(imageDir, f'{imageDir}_labels.coco.json'), 'w', encoding='utf-8') as f:
        json.dump(dat, f, indent=4)


def generate_image_labels(imageDir):
    classes = os.listdir(imageDir)
    imagePaths = []
    if f'{imageDir}_labels.coco.json' in classes:
        classes.remove(f'{imageDir}_labels.coco.json')
    for c in classes:
        if len(c)>1:
            continue
        currPaths = os.listdir(os.path.join(imageDir,c))
        currPaths = [os.path.join(imageDir, c, y) for y in currPaths]
        imagePaths+=currPaths
    create_coco_json(classes,imagePaths)

generate_image_labels('letter_dataset_5000')
sys.exit()

def unpack_images(imageDir):
    try:
        os.mkdir(imageDir+'\\all_images')
    except:
        pass
    for c in os.listdir(imageDir):
        for img in os.listdir(imageDir+'\\'+c):
            try:
                shutil.copy(f'{imageDir}\\{c}\\{img}', f'{imageDir}\\all_images\\{img}')
            except:
                pass

