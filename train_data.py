import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from os import listdir
from os.path import isfile, join


# data training
trained_models = {}
def train_data(dataset_path = 'Dataset/'):
    user_directories = [f for f in listdir(dataset_path) if not isfile(join(dataset_path, f))]
    def train_model(user_dir):
        user_path = join(dataset_path, user_dir)
        onlyfiles = [f for f in listdir(user_path) if isfile(join(user_path, f))]
        training_data, labels = [], []
        for i, file in enumerate(onlyfiles):
            image_path = join(user_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            training_data.append(image)
            labels.append(i)
        training_data = np.asarray(training_data, dtype=np.uint8)
        labels = np.asarray(labels, dtype = np.int32)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(training_data, labels)
        model.save(f'Trained_data/{user_dir}_model.yml')
        print(f'Data training done for {user_dir}')

        return model

    for user_dir in user_directories:
        model = train_model(user_dir)
        trained_models[user_dir] = model


    return trained_models
if __name__ == '__main__':
    train_data()