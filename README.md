# zero2Deploy
Sentiment Analysis based on Tensorflow2.0 Kears with deploy

## Requirements
```
python3
tensorflow >= 2.0
```

## data & preprocess
The data provided in the `data/` directory is a csv file

In `data_util.py` I provide some funtions to process the csv file.

## Usage
This contains several steps:
1. Before you can get started on training the model, you mast run
```
python data_util.py
```

2. After the dirty preprpcessing jobs, you can try running an training experiment with some configurations by:
```
python train.py
```

3. After that, you get a fold named "my_cls_model", you can copy the fold to the deploy machine.  And run:
```
# install docker
sudo apt-get install docker # ignore if you have already install docker
# deploy serveing 
docker pull tensorflow/serving
# ignore steps above if you have already installed docker
docker run -it --rm -p 8500:8500 -p 8501:8501 -v "/my_cls_model:/models/my_cls_model" -e MODEL_NAME=my_cls_model tensorflow/serving
```

4. After the server is done, the you can run:
```
python server_test.py
```
to test the server connection is good.

5. Then you need start the Flask by:
```
python app.py
```
6. Finally you can send the request to serving by:
```
python simple_request.py
```

Follow the instruction. Hope you enjoy it.

## Reference 


## Folder Structure
```
├── data            - this fold contains all the data
│   ├── train
│   ├── dev
│   ├── test
│   ├── vocab
|   ├── vec
├── my_cls_model           - this fold contains the pb file to restore
├── train.py               - main entrance of the project
├── data_util.py           - preprocess the data
├── load_data.py           - data generator
├── server_test.py       - test if net works after deploy 
├── app.py                 - use flask  
├── simple_request.py      - use rest to send query
```

## To do
1. Still need parameters searching.
2. Need structure changing to satisfy parameters chosing.
3. Make codes nicer.
