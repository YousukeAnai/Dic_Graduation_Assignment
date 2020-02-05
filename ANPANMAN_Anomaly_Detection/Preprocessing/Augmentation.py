# coding: utf-8
import numpy as np
import cv2
import os
import scipy.stats as stats
from PIL import Image
from keras.preprocessing import image
        
def image_binarization(path_in, path_out, th_zero_num, width=100, height=100):
    """
    入力画像の輪郭を白黒で2値化して出力する。　2値化はnumpy化、標準化した数値を指定閾値で2値化。
    
    Input   : 画像ファイルが保存されているフォルダパス(終わりは/)　(フォルダには画像以外入れない)
    Output  : 2値化後の画像を指定フォルダに保存。 2値化後の0(輪郭線)の数を出力。
    
    Pramater
    path_in     : 入力画像群が入ったディレクトリパス
    path_out    : 出力ディレクトリパス
    th_zero_num : 画像の黒のドット数のMIN値
    width       : 画像の横幅サイズ
    height      : 画像の縦幅サイズ
    """
    list_in = os.listdir(path_in)
    im_np_out = np.empty((0, width*height))
    for img in list_in:    
        path_name = path_in+img
        x_img = cv2.imread(path_name)
        x_img = cv2.resize(x_img, (width, height))
        x_img= cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
        x_img = np.array(x_img)
        x_img = x_img / 255.0
        x_img = x_img.reshape((1, width, height))
        #np.savetxt(path_out+"_"+img+".txt", x_img2)
        x_img = x_img.reshape(1, width*height)
        #x_list = x_img.tolist()
        m = stats.mode(x_img)
        max_hindo = m.mode[0][0]
        for c in reversed(range(50)):
            th = (c+1)*0.01
            th_0_1 = max_hindo-th
            x_img_ = np.where(x_img>th_0_1, 1, 0)
            if (np.count_nonzero(x_img_ == 0))>th_zero_num:
                break   
        display(np.count_nonzero(x_img_ == 0))
        x_img = x_img_.reshape(width, height)
        x_img = (x_img * 2.0) - 1.0
        
        img_np_255 = (x_img + 1.0) * 127.5
        img_np_255_mod1 = np.maximum(img_np_255, 0)
        img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
        img_np_uint8 = img_np_255_mod1.astype(np.uint8)
        image = Image.fromarray(img_np_uint8)    
        image.save(path_out+img, quality=95)
        #cv2.imwrite(path_out+img, image)
        #im_np_out = np.concatenate([im_np_out, x_img], axis = 0)        

def image2np_binarization(x_img, th_zero_num, width, height):
    """
    画像を白黒2値化する。　2値化閾値は輪郭のドット数を外部から指定した数以上になる閾値に設定。
    
    input : image, 　2値化閾値を決めるための輪郭のドット数
    output: nd_array (2dim)
    """
    x_test_false = np.empty((0, width*height))
    x_img = cv2.resize(x_img, (width, height))
    x_img= cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
    x_img = np.array(x_img)
    x_img = x_img / 255.0
    x_img = x_img.reshape((1, width, height))
    x_img = x_img.reshape(1, width*height)
    m = stats.mode(x_img)
    max_hindo = m.mode[0][0]
    for c in reversed(range(50)):
        th = (c+1)*0.01
        th_0_1 = max_hindo-th
        x_img_ = np.where(x_img>th_0_1, 1, 0)
        if (np.count_nonzero(x_img_ == 0))>th_zero_num:
            break   
    x_img = x_img_.reshape(width, height)
    return x_img

def ary2image(np_2d):
    """
    画像から変換したnd_arrayを画像に戻す。
    
    input : image, 　2値化閾値を決めるための輪郭のドット数
    output: nd_array (2dim)
    """
    x_img = (np_2d * 2.0) - 1.0
    img_np_255 = (x_img + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    image = Image.fromarray(img_np_uint8)    
    #image.save(path2+img, quality=95)    
    return image

def horizontal_flip(np_2d):
    """
    画像から変換したnd_arrayを横反転させる。
    input  : nd_array(2dim (w,h))
    output : nd_array(2dim (w,h))
    """
    np_2d_out = np_2d[:, ::-1]
    return np_2d_out

def binarization_gradation_h_flip_augmantation(path_in, path_out, n_times_binarization_th, n_times_frip, th_zero_num_min, th_zero_num_max, width=100, height=100):
    """
    入力画像を2値化閾値、横反転について、それぞれ指定倍にAugmentationする。
    
    Input  : 入出力画像フォルダパス、　2値化閾値、横反転について何倍Augmentationするか、　2値化閾値のための輪郭ドット数範囲MIN,MAX、画像サイズ
    Output : 指定フォルダにAugmentationした画像を保存　(拡張子は入力画像と同じ)
    """    
    list_in = os.listdir(path_in)
    for name in list_in:
        path_name = path_in + name
        x_img = cv2.imread(path_name)
        binari_step = round( (th_zero_num_max-th_zero_num_min) / n_times_binarization_th )
        for i in range(n_times_binarization_th):
            zero_th = (binari_step * (i+1)) +th_zero_num_min
            img3 = image2np_binarization(x_img, zero_th, width,height)
            for j in range(n_times_frip):
                if j==1:
                    img3 = horizontal_flip(img3)
                image = ary2image(img3)
                image.save(path_out+str(zero_th)+"_"+str(j)+"_"+name, quality=95)
