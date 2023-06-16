
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
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir= Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path= Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir= Tensorflow\workspace\models\my_ssd_mobnet
```
## Installation
Go to https://www.anaconda.com/products/individual and click the “Download” button
Run the downloaded executable (.exe) file to begin the installation. See here for more details

### Create a new Anaconda virtual environment
Open a new anaconda prompt window
```
conda create -n tensorflow pip python=3.9
```

### Activate the Anaconda virtual environment
Activating the newly created virtual environment is achieved by running the following in the Terminal window:
```
conda activate tensorflow
```
Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:
```
(tensorflow) C:\Users\microsoft>
```
### Clone tensorflow models in local folder
Cloning models from tensorflow github
```
cd ./path/foldername
git clone https://github.com/tensorflow/models.git
```
### Install protocol buffer
```
conda install protobuf
```
and then doing the compilation
```
cd foldername/models/research
protoc object_detection/protos/*.proto --python_out=.
```
### copy setup.py file into ./models/research
```
cd foldername/models/research
cp object_detection/packages/tf2/setup.py .
```
### Install Object Detection API
```
#From within TensorFlow/models/research/
python -m pip install .
```
### Reinstalling numpy from conda
numpy defaults from API Installation is not compatible.
So, need install numpy module from conda
```
conda install numpy
```
### Testing your Installation
```
# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
```
### Try out the examples
If the previous step completed successfully it means you have successfully installed all the components necessary to perform object detection using pre-trained models.

If you want to play around with some examples to see how this can be done, now would be a good time to have a look at the Examples section.

## References

-	https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/home/welcome
-	https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/home/welcome
-	https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html 
-	https://youtu.be/yqkISICHH-U
-	https://youtu.be/rRwflsS67ow

