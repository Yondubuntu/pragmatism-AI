# 파이썬 이미지를 기반으로 함
FROM tensorflow/tensorflow

# Docker 이미지 내부에서 RUN, CMD, ENTRYPOINT의 명령이 실행될 디렉토리를 설정 (작업 폴더 설정)
WORKDIR /app

# 현재 디렉터리에 있는 파일들을 이미지 내부 /app 디렉터리에 복사함
COPY . /app

RUN pip install opencv-contrib-python
RUN pip install Flask
RUN pip install pillow
RUN pip install gunicorn

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install libgl1-mesa-glx -y

RUN mkdir pretrained
RUN wget -O ./pretrained/model.tar.gz http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
RUN tar -xvf ./pretrained/model.tar.gz -C pretrained
RUN wget -O ./pretrained/model.pbtxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app
