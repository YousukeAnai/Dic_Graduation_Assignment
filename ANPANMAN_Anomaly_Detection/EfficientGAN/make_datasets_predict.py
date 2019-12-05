import numpy as np
import os
import glob 
import re
import random
import cv2
from PIL import Image
from keras.preprocessing import image

class Make_datasets_predict():

    def __init__(self, test_data, img_width, img_height, seed):
        self.filename = test_data
        #self.true_data = true_data
        #self.false_data = false_data
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed
        x_test = self.read_DATASET(self.filename)
        #self.train_np = x_train #np.concatenate((y_train.reshape(-1,1), x_train), axis=1).astype(np.float32)
        #self.valid_true_np = np.concatenate((y_valid_true.reshape(-1,1), x_valid_true), axis=1).astype(np.float32)
        #self.valid_false_np = np.concatenate((y_valid_false.reshape(-1,1), x_valid_false), axis=1).astype(np.float32)
        #print("self.train_np.shape, ", self.train_np.shape)
        #print("self.valid_true_np.shape, ", self.valid_true_np.shape)
        #print("self.valid_false_np.shape, ", self.valid_false_np.shape)
        #print("np.max(x_train), ", np.max(x_train))
        #print("np.min(x_train), ", np.min(x_train))
        self.valid_data = x_test #np.concatenate((self.valid_true_np, self.valid_false_np))

        random.seed(self.seed)
        np.random.seed(self.seed)

    def read_DATASET(self, test_path):
        test_list = os.listdir(test_path)
        #y_train = np.ones(len(train_list))
        
        x_test = np.empty((0, self.img_width*self.img_height))
        for img in test_list:    
            path_name = test_path+img
            x_img = cv2.imread(path_name)
            # サイズを揃える
            x_img = cv2.resize(x_img, (self.img_width, self.img_height))
            # 3chを1chに変換
            x_img= cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
            # PIL.Image.Imageからnumpy配列へ
            x_img = np.array(x_img)
            # 正規化
            x_img = x_img / 255.0
            # axisの追加
            x_img = x_img.reshape((1,self.img_width, self.img_height))
            # flatten
            x_img = x_img.reshape(1, self.img_width*self.img_height)
            x_test = np.concatenate([x_test, x_img], axis = 0)
           
        print("x_test.shape, ", x_test.shape)
        #print("y_train.shape, ", y_train.shape)

        return x_test

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def divide_MNIST_by_digit(self, train_np, data1_num, data2_num):
        data_1 = train_np[train_np[:,0] == data1_num]
        data_2 = train_np[train_np[:,0] == data2_num]

        return data_1, data_2

    def read_data(self, d_y_np, width, height):
        #tars = []
        images = []
        for num, d_y_1 in enumerate(d_y_np):
            image = d_y_1.reshape(width, height, 1)
            #tar = d_y_1[0]
            images.append(image)
            #tars.append(tar)

        return np.asarray(images)#, np.asarray(tars)


    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = (data * 2.0) - 1.0 #applied for tanh

        return data_norm


    def make_data_for_1_epoch(self):
        self.filename_1_epoch = np.random.permutation(self.train_np)

        return len(self.filename_1_epoch)


    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.filename_1_epoch[i:i + batchsize]
        images, _ = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n

    def get_valid_data_for_1_batch(self, i, batchsize):
        filename_batch = self.valid_data[i:i + batchsize]
        images = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n#, tars

    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
        return target
