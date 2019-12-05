import numpy as np
# import os
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import csv
import seaborn as sns

def compute_precision_recall(score_A_np, ):
    array_1 = np.where(score_A_np[:, 1] == 1.0)
    array_0 = np.where(score_A_np[:, 1] == 0.0)
    print("len(array_1), ", len(array_1))
    print("len(array_0), ", len(array_0))
    array_1_np = score_A_np[array_1]
    array_0_np = score_A_np[array_0]
    mean_1 = np.mean((score_A_np[array_1])[:, 0])
    mean_0 = np.mean((score_A_np[array_0])[:, 0])
    medium = (mean_1 + mean_0) / 2.0
    print("mean_1, ", mean_1)
    print("mean_0, ", mean_0)
    print("medium, ", medium)
    #threshold_name = './score_threshold.npy'
    np.save('./score_threshold.npy', medium)
        
    array_upper = score_A_np[:, 0] >= medium
    array_lower = score_A_np[:, 0] < medium
    print("np.sum(array_upper.astype(np.float32)), ", np.sum(array_upper.astype(np.float32)))
    print("np.sum(array_lower.astype(np.float32)), ", np.sum(array_lower.astype(np.float32)))
    array_1_tf = score_A_np[:, 1] == 1.0
    array_0_tf = score_A_np[:, 1] == 0.0
    print("np.sum(array_1_tf.astype(np.float32)), ", np.sum(array_1_tf.astype(np.float32)))
    print("np.sum(array_0_tf.astype(np.float32)), ", np.sum(array_0_tf.astype(np.float32)))

    tn = np.sum(np.equal(array_lower, array_1_tf).astype(np.int32))
    tp = np.sum(np.equal(array_upper, array_0_tf).astype(np.int32))
    fp = np.sum(np.equal(array_upper, array_1_tf).astype(np.int32))
    fn = np.sum(np.equal(array_lower, array_0_tf).astype(np.int32))

    precision = tp / (tp + fp + 0.00001)
    recall = tp / (tp + fn + 0.00001)

    return tp, fp, tn, fn, precision, recall, array_1_np, array_0_np

def score_divide(score_A_np):
    array_1 = np.where(score_A_np[:, 1] == 1.0)
    array_0 = np.where(score_A_np[:, 1] == 0.0)
    print("len(array_1), ", len(array_1))
    print("len(array_0), ", len(array_0))
    array_1_np = score_A_np[array_1]
    array_0_np = score_A_np[array_0]

    return array_1_np, array_0_np

def save_graph(x, y, filename, epoch):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.title('ROC curve ' + filename + ' epoch:' + str(epoch))
    # x axis label
    plt.xlabel("FP / (FP + TN)")
    # y axis label
    plt.ylabel("TP / (TP + FN)")
    # save
    plt.savefig(filename + '_ROC_curve_epoch' + str(epoch) +'.png')
    plt.close()


def make_ROC_graph(score_A_np, filename, epoch):
    argsort = np.argsort(score_A_np, axis=0)[:, 0]
    value_1_0 = score_A_np[argsort][::-1].astype(np.float32)
    #value_1_0 = (np.where(score_A_np_sort[:, 1] == 7., 1., 0.)).astype(np.float32)
    # score_A_np_sort_0_1 = np.concatenate((score_A_np_sort, value_1_0), axis=1)
    sum_1 = np.sum(value_1_0)

    len_s = len(score_A_np)
    sum_0 = len_s - sum_1
    tp = np.cumsum(value_1_0[:, 1]).astype(np.float32)
    index = np.arange(1, len_s + 1, 1).astype(np.float32)
    fp = index - tp
    fn = sum_1 - tp
    tn = sum_0 - fp
    tp_ratio = tp / (tp + fn + 0.00001)
    fp_ratio = fp / (fp + tn + 0.00001)
    save_graph(fp_ratio, tp_ratio, filename, epoch)

    auc = sm.auc(fp_ratio, tp_ratio)
    return auc


def unnorm_img(img_np):
    img_np_255 = (img_np + 1.0) * 127.5
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8

def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL

def make_score_hist(array_1_np, array_0_np, epoch, LOGFILE_NAME, OUT_HIST_DIR):
    array_1_np = np.delete(array_1_np, 1, 1)
    array_1_np = array_1_np.flatten()
    array_0_np = np.delete(array_0_np, 1, 1)
    array_0_np = array_0_np.flatten()
    list_1 = array_1_np.tolist()
    list_0 = array_0_np.tolist()
    plt.figure(figsize=(7, 5))
    plt.title("Histgram of Score")
    plt.xlabel("Score")
    plt.ylabel("freq")
    plt.hist(list_1, bins=40, alpha=0.3, histtype='stepfilled', color='r', label="1")
    plt.hist(list_0, bins=40, alpha=0.3, histtype='stepfilled', color='b', label='0')
    plt.legend(loc=1)
    plt.savefig(OUT_HIST_DIR + "/resultScoreHist_"+ LOGFILE_NAME + '_' + str(epoch) + ".png")
    plt.show()    
    
def make_score_hist_test(array_1_np, array_0_np, score_th, LOGFILE_NAME, OUT_HIST_DIR):
    array_1_np = np.delete(array_1_np, 1, 1)
    array_1_np = array_1_np.flatten()
    array_0_np = np.delete(array_0_np, 1, 1)
    array_0_np = array_0_np.flatten()
    list_1 = array_1_np.tolist()
    list_0 = array_0_np.tolist()
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()
    ax.set_title("Histgram of Score")
    ax.set_xlabel("Score")
    ax.set_ylabel("freq")
    ax.hist(list_1, bins=40, alpha=0.3, histtype='stepfilled', color='r', label="1")
    ax.hist(list_0, bins=40, alpha=0.3, histtype='stepfilled', color='b', label='0')
    score_th_ylim = ax.get_ylim()
    ax.vlines(score_th, 0, score_th_ylim, colors = "red", linestyles='dotted')
    ax.legend(loc=1)
    plt.savefig(OUT_HIST_DIR + "/resultScoreHist_"+ LOGFILE_NAME + "_test.png")
    plt.show()   
def make_score_bar(score_a):
    
    score_a = score_a.tolist()
    list_images_PIL = []
    for score in score_a:
        x="score"
        plt.bar(x,score,label=score)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.bar(x,score,label=round(score,3))
        ax.legend(loc='center', fontsize=12)
        fig.canvas.draw()
        #im = np.array(fig.canvas.renderer.buffer_rgba()) # matplotlibが3.1より以降の場合
        im = np.array(fig.canvas.renderer._renderer) 
        image_1_PIL = Image.fromarray(im)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL 

def make_score_bar_predict(score_A_np_tmp):
    score_a = score_A_np_tmp.tolist()
    list_images_PIL = []
    for score in score_a:
        x="score"
        #plt.bar(x,score[0],label=score)
        fig, ax = plt.subplots(figsize=(1, 1))
        if score[1]==0:
            ax.bar(x,score[0], color='red',label=round(score[0],3))
        else:
            ax.bar(x,score[0], color='blue',label=round(score[0],3))
        ax.legend(loc='center', fontsize=12)
        fig.canvas.draw()
        #im = np.array(fig.canvas.renderer.buffer_rgba()) # matplotlibが3.1より以降の場合
        im = np.array(fig.canvas.renderer._renderer) 
        image_1_PIL = Image.fromarray(im)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL 

def make_output_img(img_batch_1, img_batch_0, x_z_x_1, x_z_x_0, score_a_0, score_a_1, epoch, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_1.shape

    img_batch_1_unn = np.tile(unnorm_img(img_batch_1), (1, 1, 3))
    img_batch_0_unn = np.tile(unnorm_img(img_batch_0), (1, 1, 3))
    x_z_x_1_unn = np.tile(unnorm_img(x_z_x_1), (1, 1, 3))
    x_z_x_0_unn = np.tile(unnorm_img(x_z_x_0), (1, 1, 3))
        
    diff_1 = img_batch_1 - x_z_x_1
    diff_1_r = (2.0 * np.maximum(diff_1, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    diff_1_b = (2.0 * np.abs(np.minimum(diff_1, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    diff_1_g = diff_1_b * 0.0 - 1.0
    diff_1_r_unnorm = unnorm_img(diff_1_r)
    diff_1_b_unnorm = unnorm_img(diff_1_b)
    diff_1_g_unnorm = unnorm_img(diff_1_g)
    diff_1_np = np.concatenate((diff_1_r_unnorm, diff_1_g_unnorm, diff_1_b_unnorm), axis=3)
    
    diff_0 = img_batch_0 - x_z_x_0
    diff_0_r = (2.0 * np.maximum(diff_0, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    diff_0_b = (2.0 * np.abs(np.minimum(diff_0, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    diff_0_g = diff_0_b * 0.0 - 1.0
    diff_0_r_unnorm = unnorm_img(diff_0_r)
    diff_0_b_unnorm = unnorm_img(diff_0_b)
    diff_0_g_unnorm = unnorm_img(diff_0_g)
    diff_0_np = np.concatenate((diff_0_r_unnorm, diff_0_g_unnorm, diff_0_b_unnorm), axis=3)

    img_batch_1_PIL = convert_np2pil(img_batch_1_unn)
    img_batch_0_PIL = convert_np2pil(img_batch_0_unn)
    x_z_x_1_PIL = convert_np2pil(x_z_x_1_unn)
    x_z_x_0_PIL = convert_np2pil(x_z_x_0_unn)
    diff_1_PIL = convert_np2pil(diff_1_np)
    diff_0_PIL = convert_np2pil(diff_0_np)
    score_a_1_PIL = make_score_bar(score_a_1)
    score_a_0_PIL = make_score_bar(score_a_0)
    
    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * 8 - 1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    
    for num, (ori_1, ori_0, xzx1, xzx0, diff1, diff0, score_1, score_0) in enumerate(zip(img_batch_1_PIL, img_batch_0_PIL ,x_z_x_1_PIL, x_z_x_0_PIL, diff_1_PIL, diff_0_PIL, score_a_1_PIL, score_a_0_PIL)):
        wide_image_PIL.paste(ori_1,                   (0,      num * (img1_h + 1)))
        wide_image_PIL.paste(xzx1,           (img1_w + 1,      num * (img1_h + 1)))
        wide_image_PIL.paste(diff1,         ((img1_w + 1) * 2, num * (img1_h + 1)))
        wide_image_PIL.paste(score_1, ((img1_w + 1) * 3, num * (img1_h + 1)))
        wide_image_PIL.paste(ori_0,         ((img1_w + 1) * 4, num * (img1_h + 1)))
        wide_image_PIL.paste(xzx0,          ((img1_w + 1) * 5, num * (img1_h + 1)))
        wide_image_PIL.paste(diff0,         ((img1_w + 1) * 6, num * (img1_h + 1)))
        wide_image_PIL.paste(score_0, ((img1_w + 1) * 7, num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + '_' + str(epoch) + ".png")

def make_output_img_test(img_batch_test, x_z_x_test, score_A_np_tmp, log_file_name, out_img_dir):
    (data_num, img1_h, img1_w, _) = img_batch_test.shape

    img_batch_test_unn = np.tile(unnorm_img(img_batch_test), (1, 1, 3))
    x_z_x_test_unn = np.tile(unnorm_img(x_z_x_test), (1, 1, 3))
        
    diff_test = img_batch_test - x_z_x_test
    diff_test_r = (2.0 * np.maximum(diff_test, 0.0)) - 1.0 #(0.0, 1.0) -> (-1.0, 1.0)
    diff_test_b = (2.0 * np.abs(np.minimum(diff_test, 0.0))) - 1.0 #(-1.0, 0.0) -> (1.0, 0.0) -> (1.0, -1.0)
    diff_test_g = diff_test_b * 0.0 - 1.0
    diff_test_r_unnorm = unnorm_img(diff_test_r)
    diff_test_b_unnorm = unnorm_img(diff_test_b)
    diff_test_g_unnorm = unnorm_img(diff_test_g)
    diff_test_np = np.concatenate((diff_test_r_unnorm, diff_test_g_unnorm, diff_test_b_unnorm), axis=3)

    img_batch_test_PIL = convert_np2pil(img_batch_test_unn)
    x_z_x_test_PIL = convert_np2pil(x_z_x_test_unn)
    diff_test_PIL = convert_np2pil(diff_test_np)
    
    score_a = score_A_np_tmp[:, 1:]
    #tars = score_A_np_tmp[:, 0]
    score_a_PIL = make_score_bar_predict(score_A_np_tmp)
    
    wide_image_np = np.ones(((img1_h + 1) * data_num - 1, (img1_w + 1) * 8 - 1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    
    for num, (ori_test, xzx_test, diff_test, score_test) in enumerate(zip(img_batch_test_PIL, x_z_x_test_PIL, diff_test_PIL, score_a_PIL)):
        wide_image_PIL.paste(ori_test,             (0,      num * (img1_h + 1)))
        wide_image_PIL.paste(xzx_test,    (img1_w + 1,      num * (img1_h + 1)))
        wide_image_PIL.paste(diff_test,  ((img1_w + 1) * 2, num * (img1_h + 1)))
        wide_image_PIL.paste(score_test, ((img1_w + 1) * 3, num * (img1_h + 1)))

    wide_image_PIL.save(out_img_dir + "/resultImage_"+ log_file_name + "_test.png")
    
def save_list_to_csv(list, filename):
    f = open(filename, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(list)
    f.close()
