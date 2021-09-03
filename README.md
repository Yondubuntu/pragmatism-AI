# pragmatism-AI
AI의 이론보다는 활용에 중심을 두는 '실용주의 AI'입니다.

대부분의 모델들은 구현 보다는 'Inference'에 초점을 맞추고 있습니다.

## Course
### Multi Layer Perceptron & Hyper Parameter Optimization
* [MLP](https://github.com/silverstar0727/pragmatism-AI/blob/main/DLbasic-HPO/MLP(MNIST).ipynb)
* [HPO 기초 적용](https://github.com/silverstar0727/pragmatism-AI/blob/main/DLbasic-HPO/W%26B_keras_sweep.ipynb)
* [Cifar10 데이터 셋에 HPO 적용](https://github.com/silverstar0727/pragmatism-AI/blob/main/DLbasic-HPO/MLP_HPO(cifar10).ipynb)

- - -

### Computer Vision
#### Image Classification
- [VGG](https://github.com/silverstar0727/pragmatism-AI/blob/main/CV/ImageClassification/VGG(cifar10).ipynb)
- [ResNet](https://github.com/silverstar0727/pragmatism-AI/blob/main/CV/ImageClassification/ResNet(cifar100).ipynb)
- MobileNet

#### Object Detection
- two-stage detector
  - RCNN
  - Fast RCNN 
  - Faster RCNN
- one-stage detector
  - SSD
  - Yolo v1, v2, v3 (Ultralytics Yolo)
  - RetinaDet

#### Image Segmentation
- U-Net
- Mask RCNN

#### Latest Models
- ViT(Vision Transformer)
- MLP-Mixer


- - -

### Natural Language Processing
- RNN
- LSTM
- Attention mechanism
- Transformer
- HuggingFace(+Finetuning)
  - pipeline
  - customizing
  - Classification
  - Summarization
  - Text Generization

- - -

### Model Deployment(with Docker)
- Tensorflow Serving
- [Flask REST API](https://github.com/silverstar0727/pragmatism-AI/tree/main/flask_api) 
  - Dockerizing REST API
  - GCP Cloud Run(serverless deployment)
- Firebase(for Mobile)


- - -

### ML Pipeline in GCP
- Dockerizing each ML tasks
- Github Actions
- [Kubeflow](https://github.com/silverstar0727/ML-Pipeline-Tutorial/tree/main/kubeflow-pipeline)
- [GCP Vertex AI](https://github.com/silverstar0727/ML-Pipeline-Tutorial/tree/main/vertex-ai-pipeline)

- - -

### CI/CD Tools & Others
- Tensorflow Extended
  - [Tensorflow Transform](https://github.com/silverstar0727/ML-Pipeline-Tutorial/blob/main/tfx-pipeline/tfx-components/TFT_tutorial.ipynb)
  - [Tensorflow Data Validation](https://github.com/silverstar0727/ML-Pipeline-Tutorial/blob/main/tfx-pipeline/tfx-components/TFDV_tutorial.ipynb)
  - Tensorflow Model Analysis
  - [Pipeline](https://github.com/silverstar0727/ML-Pipeline-Tutorial/tree/main/tfx-pipeline)
- GitLab CI
- mlflow(model registry)
- bentoml
- ML with kubernetes Job
