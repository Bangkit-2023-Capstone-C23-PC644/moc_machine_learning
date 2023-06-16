
# MOC Machine Learning Model

Medical Oversee Control is a machine learning model that is implemented in hospital CCTV cameras. This model provides special features as artificial intelligence which is capable of detecting the presence of certain objects. In this case, the model will be implemented to calculate the number of people waiting in line at the hospital. This model was built using the Single Shot Detector (SSD) model as a pre-trained architecture. The model is then trained using the transfer learning method so that it can detect the presence of people in the image.

## Authors
- [Wahyu Fatkhan H](https://github.com/wahyufatkhan)
- [Farid Abdullah M](https://github.com/farid-abd)
- [Lerincia Andriani](https://github.com/Stargazerin)

## Requirements
-	Python3.9
-	Anaconda Navigator
-	Anaconda Prompt
-	Jupyter Notebook

## Dependencies
- Python3.9 | Jupyter Notebook
- Anaconda Navigator | Matplotlib
- Anaconda Prompt | Numpy
- Jupyter notebook | OpenCV
- Protobuf 2.6 | Keras
- Pillow 1.0 | wheel-0.38.4
- lxml | pyqt-5.15.7
- tf Slim (which is included in the "tensorflow/models" checkout) | Pillow
- Pycocotools | Pandas
- Pyparsing | Avro-python3
- Apache-beam | Lxml
- Cython | Contextlib2
- Tf-slim | Six
- Lvis | Scipy
- tensorflow 2.9.0 | Tf-models-official
- tensorflow-addons 0.20.0 | Sacrebleu
- tensorflow-datasets 4.9.0 | tensorboard 2.9.0
- tensorflow-estimator 2.9.0 | tensorboard-data-server 0.6.1
- tensorflow-hub 0.12.0 | tensorboard-plugin-wit 1.8.1
- tensorflow-io 0.31.0 | tensorflow-metadata 1.13.0
- tensorflow-io-gcs-filesystem 0.31.0 | tensorflow-model-optimization 0.7.5
- tensorflow-text 2.10.0 | tensorflowjs 3.19.0
- ipykernel

## Datasets
https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

## Training
Training is carried out using a collection of image datasets that have been labeled using xml labels. The number of image label data used as training data is 110 images, which will be divided into 85 images for training and 25 images for testing. The training is carried out with a total of 2000 steps. The results of the training performed on the model will be stored in the form of a checkpoint file.

Command for Training:

```python
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path= Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
```
Command for evaluating:

```python
python Documents\Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir= Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path= Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir= Tensorflow\workspace\models\my_ssd_mobnet
```

## References

-	https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/welcome
-	https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/home/welcome 
-	https://youtu.be/yqkISICHH-U
-	https://youtu.be/rRwflsS67ow

