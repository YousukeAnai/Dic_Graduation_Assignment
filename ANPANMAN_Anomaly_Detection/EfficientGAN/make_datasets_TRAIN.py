import numpy as np
import os
import glob 
import re
import random
#import cv2
from PIL import Image
from keras.preprocessing import image

class Make_datasets_TRAIN():

    def __init__(self, filename, true_data, false_data, img_width, img_height, seed):
        self.filename = filename
        self.true_data = true_data
        self.false_data = false_data
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed
        x_train, x_valid_true, x_valid_false, y_train, y_valid_true, y_valid_false = self.read_DATASET(self.filename, self.true_data, self.false_data)
        self.train_np = np.concatenate((y_train.reshape(-1,1), x_train), axis=1).astype(np.float32)
        self.valid_true_np = np.concatenate((y_valid_true.reshape(-1,1), x_valid_true), axis=1).astype(np.float32)
        self.valid_false_np = np.concatenate((y_valid_false.reshape(-1,1), x_valid_false), axis=1).astype(np.float32)
        print("self.train_np.shape, ", self.train_np.shape)
        print("self.valid_true_np.shape, ", self.valid_true_np.shape)
        print("self.valid_false_np.shape, ", self.valid_false_np.shape)
        print("np.max(x_train), ", np.max(x_train))
        print("np.min(x_train), ", np.min(x_train))
        self.valid_data = np.concatenate((self.valid_true_np, self.valid_false_np))

        random.seed(self.seed)
        np.random.seed(self.seed)

    def read_DATASET(self, train_path, true_path, false_path):
        train_list = os.listdir(train_path)
        y_train = np.ones(len(train_list))
        
        x_train = np.empty((0, self.img_width*self.img_height))
        for img in train_list:    
            path_name = train_path+img
            x_img = Image.open(path_name)
            # サイズを揃える
            x_img = x_img.resize((self.img_width, self.img_height))
            # 3chを1chに変換
            x_img= x_img.convert('L')
            # PIL.Image.Imageからnumpy配列へ
            x_img = np.array(x_img)
            # 正規化
            x_img = x_img / 255.0
            # axisの追加
            x_img = x_img.reshape((1,self.img_width, self.img_height))
            # flatten
            x_img = x_img.reshape(1, self.img_width*self.img_height)
            x_train = np.concatenate([x_train, x_img], axis = 0)
           
        print("x_train.shape, ", x_train.shape)
        print("y_train.shape, ", y_train.shape)

        test_true_list = os.listdir(true_path)
        y_test_true = np.ones(len(test_true_list))
        
        x_test_true = np.empty((0, self.img_width*self.img_height))
        for img in test_true_list:    
            path_name = true_path+img
            x_img = Image.open(path_name)
            x_img = x_img.resize((self.img_width, self.img_height))
            x_img= x_img.convert('L')
            x_img = np.array(x_img)
            x_img = x_img / 255.0
            x_img = x_img.reshape((1,self.img_width, self.img_height))
            x_img = x_img.reshape(1, self.img_width*self.img_height)
            x_test_true = np.concatenate([x_test_true, x_img], axis = 0)
        
        print("x_test_true.shape, ", x_test_true.shape)
        print("y_test_true.shape, ", y_test_true.shape)
      
        test_false_list = os.listdir(false_path)
        y_test_false = np.zeros(len(test_false_list))
        
        x_test_false = np.empty((0, self.img_width*self.img_height))
        for img in test_false_list:    
            path_name = false_path+img
            x_img = Image.open(path_name)
            x_img = x_img.resize((self.img_width, self.img_height))
            x_img= x_img.convert('L')
            x_img = np.array(x_img)
            x_img = x_img / 255.0
            x_img = x_img.reshape((1,self.img_width, self.img_height))
            x_img = x_img.reshape(1, self.img_width*self.img_height)
            x_test_false = np.concatenate([x_test_false, x_img], axis = 0)
        
        print("x_test_false.shape, ", x_test_false.shape)        
        print("y_test_false.shape, ", y_test_false.shape)        

        return x_train, x_test_true, x_test_false, y_train, y_test_true, y_test_false

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    #def divide_MNIST_by_digit(self, train_np, data1_num, data2_num):
    #    data_1 = train_np[train_np[:,0] == data1_num]
    #    data_2 = train_np[train_np[:,0] == data2_num]

    #    return data_1, data_2



    def read_data(self, d_y_np, width, height):
        tars = []
        images = []
        for num, d_y_1 in enumerate(d_y_np):
            image = d_y_1[1:].reshape(width, height, 1)
            tar = d_y_1[0]
            images.append(image)
            tars.append(tar)

        return np.asarray(images), np.asarray(tars)


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
        images, tars = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n, tars

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
