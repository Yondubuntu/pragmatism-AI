import cv2
from flask import Flask, request, redirect, url_for, jsonify, Response
import os, io
from PIL import Image

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    # 반환할 respose를 생성합니다.
    response = {'success': False}

    # 파일의 입력을 요청합니다.
    request.files.get('file')
    # 파일을 읽습니다.
    image_requested = request.files['file'].read()
    # 읽은 파일을 엽니다.
    image = Image.open(io.BytesIO(image_requested))
    # RGB로 변환하여 줍니다.
    rgb_image = image.convert('RGB')
    # "image.jpg"로 저장합니다.
    rgb_image.save('image.jpg')

    # opencv를 이용하여 파일을 읽어줍니다.
    img = cv2.imread('image.jpg')

    # cv2.dnn을 이용하여 pretrianed된 tensorflow 모델을 열어줍니다.
    # 이때 Dockerfile에서 정상적으로 다운이 되지 않았다면 작동하지 않습니다.
    cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 
                                        './pretrained/model.pbtxt')

    # COCO Dataset에 대한 라벨을 딕셔너리 형태로 지정합니다.
    labels_to_names_0 = {0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
                        10:'fire hydrant',11:'street sign',12:'stop sign',13:'parking meter',14:'bench',15:'bird',16:'cat',17:'dog',18:'horse',19:'sheep',
                        20:'cow',21:'elephant',22:'bear',23:'zebra',24:'giraffe',25:'hat',26:'backpack',27:'umbrella',28:'shoe',29:'eye glasses',
                        30:'handbag',31:'tie',32:'suitcase',33:'frisbee',34:'skis',35:'snowboard',36:'sports ball',37:'kite',38:'baseball bat',39:'baseball glove',
                        40:'skateboard',41:'surfboard',42:'tennis racket',43:'bottle',44:'plate',45:'wine glass',46:'cup',47:'fork',48:'knife',49:'spoon',
                        50:'bowl',51:'banana',52:'apple',53:'sandwich',54:'orange',55:'broccoli',56:'carrot',57:'hot dog',58:'pizza',59:'donut',
                        60:'cake',61:'chair',62:'couch',63:'potted plant',64:'bed',65:'mirror',66:'dining table',67:'window',68:'desk',69:'toilet',
                        70:'door',71:'tv',72:'laptop',73:'mouse',74:'remote',75:'keyboard',76:'cell phone',77:'microwave',78:'oven',79:'toaster',
                        80:'sink',81:'refrigerator',82:'blender',83:'book',84:'clock',85:'vase',86:'scissors',87:'teddy bear',88:'hair drier',89:'toothbrush',
                        90:'hair brush'}

    # 원본 이미지가 Faster RCNN기반 네트웍으로 입력 시 resize됩니다. 
    # resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요가 필요합니다.
    rows = img.shape[0]
    cols = img.shape[1]
    # cv2의 rectangle()은 인자로 들어온 이미지 배열에 직접 사각형을 업데이트 하므로 그림 표현을 위한 별도의 이미지 배열 생성합니다. 
    draw_img = img.copy()

    # 원본 이미지 배열 BGR을 RGB로 변환하여 배열 입력, Tensorflow Faster RCNN은 size를 고정할 필요가 없는 것으로 추정됩니다. 
    cv_net.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))

    # Object Detection 수행하여 결과를 cvout으로 반환 합니다.
    cv_out = cv_net.forward().tolist()

    # response의 "cvout"에 결과를 저장하고 성공적으로 수행했다는 것을 success에 True로 바꿉니다.
    response['cvout'] = cv_out
    response['success'] = True

    # json형태로 반환합니다.
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))