import cv2
from flask import Flask, request, redirect, url_for, jsonify, Response
import os, io
from PIL import Image

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    response = {'success': False}

    request.files.get('file')
    image_requested = request.files['file'].read()
    image = Image.open(io.BytesIO(image_requested))
    rgb_image = image.convert('RGB')
    rgb_image.save('image.jpg')

    img = cv2.imread('image.jpg')

    cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 
                                        './pretrained/model.pbtxt')

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

    # 원본 이미지가 Faster RCNN기반 네트웍으로 입력 시 resize됨. 
    # resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요
    rows = img.shape[0]
    cols = img.shape[1]
    # cv2의 rectangle()은 인자로 들어온 이미지 배열에 직접 사각형을 업데이트 하므로 그림 표현을 위한 별도의 이미지 배열 생성. 
    draw_img = img.copy()

    # 원본 이미지 배열 BGR을 RGB로 변환하여 배열 입력. Tensorflow Faster RCNN은 size를 고정할 필요가 없는 것으로 추정. 
    cv_net.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))

    # Object Detection 수행하여 결과를 cvOut으로 반환 
    cv_out = cv_net.forward().tolist()

    response['cvout'] = cv_out
    response['success'] = True

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


# curl -X POST -F file=@aespa.png 'http://localhost:8080/api'
# curl -X POST -F file=@aespa.png 'https://faster-rcnn-api-kkwjtn22ja-uc.a.run.app/api'