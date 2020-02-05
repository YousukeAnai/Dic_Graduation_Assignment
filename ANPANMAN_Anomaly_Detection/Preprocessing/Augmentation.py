# coding: utf-8
import numpy as np
import cv2

def make_contour_image2(path):
    """
    イメージ画像を白黒化して指定ﾌｫﾙﾀﾞに保存
    
    input : jpg or png 
    output: inputと同じ拡張子の画像ファイル
    """
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite("gray.jpg", gray)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    #cv2.imwrite("dilated.jpg", dilated)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)
    #cv2.imwrite("diff.jpg", diff)

    # 白黒反転
    output = 255 - diff
    #cv2.imwrite(r"C:\Users\anai\dive\Graduation_Assignment\anpanman\test\0005_re.jpg", output)
    return output

def make_gray_scale(path_in, path_out):
    """
    イメージ画像を白黒化して指定ﾌｫﾙﾀﾞに保存
    
    input:jpg or png
    output:inputと同じ拡張子の画像ファイル
    """    
    x_list = os.listdir(path_in)
    for name in x_list:
        pathname = path_in+name
        img = make_contour_image2(pathname)
        cv2.imwrite(path_out+name, img)

        
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageChops
import numpy as np
#gradation_range=(220, 255)
def make_random_gray_grad(img, gradation_range=(230, 255)):
    """
    入力画像をランダムな度合いのグレイスケールに変換する
    Input  : 画像ファイル(カラーでも可)
    Output : 画像ファイル(グレイスケール変換後)
    """
    gra_range = np.random.randint(*gradation_range)
    gray = ImageOps.grayscale(img)
    output = ImageOps.colorize(gray, black=(0, 0, 0), white=(gra_range, gra_range, gra_range))
    return output

def make_random_scale_pil(image, scale_range=(50, 200)):
    """
    入力画像をランダムな度合いのサイズ(正方形)に変換する
    Input  : 画像ファイル
    Output : 画像ファイル(サイズ変換後)
    """    
    scale_size = np.random.randint(*scale_range)
    image = image.resize((scale_size, scale_size), Image.LANCZOS)
    return image

from PIL import Image, ImageOps, ImageFilter, ImageChops
import numpy as np
import cv2
def scale_gray_gradation_augmantation(path_in, path_out, n_times_size, n_times_grad):
    """
    入力画像を画像サイズ、グレイグラデーションにおいて、それぞれ指定倍にAugmentationする。
    
    Input  : 入出力画像フォルダパス　(中に画像ファイル(カラーでも可)のみ入れる)、画像サイズ,グレイグラデーションを何倍Augmentationするか
    Output : 指定フォルダにAugmentationした画像を保存　(拡張子は入力画像と同じ)
    """
    list_in = os.listdir(path_in)
    for name in list_in:
        img3 = Image.open(path_in+name)
        for i in range(n_times_size):
            img_3 = make_random_scale_pil(img3)
            for j in range(n_times_grad):
                img_3=make_random_gray_grad(img_3)
                img_3.save(path_out+str(i)+"_"+str(j)+"_"+name, quality=95)

        
from matplotlib import pyplot as plt
def make_histogram(in_np, out_path_filename):
    """
    histogramをpngで指定パスに保存、出力
    
    Input   : numpy shape(1, :)
    Output  : histogramを出力、histogramを保存(拡張子は関数の外で定義)
    """
    in_list = in_np.tolist()
    #plt.title("Histgram of Score")
    #plt.xlabel("Score")
    #plt.ylabel("freq")
    plt.hist(in_list, bins=40, alpha=0.3, histtype='stepfilled', color='r')
    #plt.legend(loc=1)
    plt.savefig(out_path_filename)
    plt.show()
    
import cv2
import os
def image2txt_histo(path_in, path_out, width, height):
    """
    入力画像をnumpy化、標準化した数値(txt)と、ヒストグラム(png)を指定パスに保存
    
    Input   : 画像ファイルが保存されているフォルダパス(終わりは/)　(フォルダには画像以外入れない)
    Output  : histogramを出力、histogramを保存(png)、txtファイルを保存
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
        x_img2 = x_img.reshape((width, height))
        np.savetxt(path_out+"_"+img+".txt", x_img2)
        x_img = x_img.reshape(1, width*height)
        make_histogram(x_img, path_out+"_"+img+".png")
        #im_np_out = np.concatenate([im_np_out, x_img], axis = 0)
        
        
import os        
import scipy.stats as stats
from PIL import Image
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

import scipy.stats as stats
from PIL import Image
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

from keras.preprocessing import image
import numpy as np
import cv2
#n_times_binarization_th=50
#n_times_frip=2
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

from PIL import Image
import sys, os, urllib.request, tarfile, cv2
from keras.preprocessing import image                
def load_resize_img(path_in, path_out, size):
    """
    フォルダ内のイメージ画像を読み込んで、任意サイズに変更し、標準化して出力
    
    input:jpg or png
    output:nd_array (shape:ファイル数、指定ｻｲｽﾞ、指定ｻｲｽﾞ、3    _PIL形式)
    """    
    x_list = os.listdir(path_in)
    for name in x_list:
        pathname = path_in+name
        x_img = Image.open(path_name)
        x_img = x_img.resize((size, size))
        x_img = np.array(x_img)
        x_img = x_img.reshape((1,size,size,3))
        x_img_array = np.concatenate([x_img_array, x_img], axis = 0)
    return x_img_array / 255.0