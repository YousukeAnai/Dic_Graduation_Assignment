import numpy as np
import os
import tensorflow as tf
import utility as Utility
import argparse
import matplotlib.pyplot as plt
from model_BiGAN import BiGAN as Model

from make_datasets_predict import Make_datasets_predict as Make_datasets

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=300, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='anpanman', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='epoch')
    #parser.add_argument('--file_train_data', '-ftd', type=str, default='./mnist.npz', help='train data')
    #parser.add_argument('--test_true_data', '-ttd', type=str, default='./mnist.npz', help='test of true_data')
    #parser.add_argument('--test_false_data', '-tfd', type=str, default='./mnist.npz', help='test of false_data')
    parser.add_argument('--test_data', '-td', type=str, default='../Test_Data/200112/', help='test of false_data')
    parser.add_argument('--valid_span', '-vs', type=int, default=1, help='validation span')
    parser.add_argument('--score_th', '-st', type=float, default=np.load('./score_threshold.npy'), help='validation span')

    return parser.parse_args()

args = parser()

#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
#FILE_NAME = args.file_train_data
#TRUE_DATA = args.test_true_data
#FALSE_DATA = args.test_false_data
TEST_DATA = args.test_data
IMG_WIDTH = 100
IMG_HEIGHT = 100
IMG_CHANNEL = 1
BASE_CHANNEL = 32
NOISE_UNIT_NUM = 200
NOISE_MEAN = 0.0
NOISE_STDDEV = 1.0
TEST_DATA_SAMPLE = 5 * 5
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
SEED = 1234
SCORE_ALPHA = 0.9 # using for cost function
VALID_SPAN = args.valid_span
np.random.seed(seed=SEED)
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME
OUT_IMG_DIR = './out_images_BiGAN' #output image file
out_model_dir = './out_models_BiGAN/' #output model_ckpt file
#Load_model_dir = '../model_ckpt/' #Load model_ckpt file
OUT_HIST_DIR = './out_score_hist_BiGAN' #output histogram file
CYCLE_LAMBDA = 1.0
SCORE_TH = args.score_th

make_datasets = Make_datasets(TEST_DATA, IMG_WIDTH, IMG_HEIGHT, SEED)
model = Model(NOISE_UNIT_NUM, IMG_CHANNEL, SEED, BASE_CHANNEL, KEEP_PROB_RATE)

z_ = tf.placeholder(tf.float32, [None, NOISE_UNIT_NUM], name='z_') #noise to generator
x_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], name='x_') #image to classifier
d_dis_f_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_g_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_') #target of discriminator related to real image
is_training_ = tf.placeholder(tf.bool, name = 'is_training')

with tf.variable_scope('encoder_model'):
    z_enc = model.encoder(x_, reuse=False, is_training=is_training_)

with tf.variable_scope('decoder_model'):
    x_dec = model.decoder(z_, reuse=False, is_training=is_training_)
    x_z_x = model.decoder(z_enc, reuse=True, is_training=is_training_) # for cycle consistency

with tf.variable_scope('discriminator_model'):
    #stream around discriminator
    drop3_r, logits_r = model.discriminator(x_, z_enc, reuse=False, is_training=is_training_) #real pair
    drop3_f, logits_f = model.discriminator(x_dec, z_, reuse=True, is_training=is_training_) #real pair
    drop3_re, logits_re = model.discriminator(x_z_x, z_enc, reuse=True, is_training=is_training_) #fake pair

with tf.name_scope("loss"):
    loss_dis_f = tf.reduce_mean(tf.square(logits_f - d_dis_f_), name='Loss_dis_gen') #loss related to generator
    loss_dis_r = tf.reduce_mean(tf.square(logits_r - d_dis_r_), name='Loss_dis_rea') #loss related to real image

    #total loss
    loss_dis_total = loss_dis_f + loss_dis_r
    loss_dec_total = loss_dis_f
    loss_enc_total = loss_dis_r

with tf.name_scope("score"):
    l_g = tf.reduce_mean(tf.abs(x_ - x_z_x), axis=(1,2,3))
    l_FM = tf.reduce_mean(tf.abs(drop3_r - drop3_re), axis=1)
    score_A =  SCORE_ALPHA * l_g + (1.0 - SCORE_ALPHA) * l_FM

with tf.name_scope("optional_loss"):
    loss_dec_opt = loss_dec_total + CYCLE_LAMBDA * l_g
    loss_enc_opt = loss_enc_total + CYCLE_LAMBDA * l_g

tf.summary.scalar('loss_dis_total', loss_dis_total)
tf.summary.scalar('loss_dec_total', loss_dec_total)
tf.summary.scalar('loss_enc_total', loss_enc_total)
merged = tf.summary.merge_all()

# t_vars = tf.trainable_variables()
dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

with tf.name_scope("train"):
    train_dis = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5).minimize(loss_dis_total, var_list=dis_vars
                                                                                , name='Adam_dis')
    train_dec = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5).minimize(loss_dec_total, var_list=dec_vars
                                                                                , name='Adam_dec')
    train_enc = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5).minimize(loss_enc_total, var_list=enc_vars
                                                                                , name='Adam_enc')
    train_dec_opt = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5).minimize(loss_dec_opt, var_list=dec_vars
                                                                                , name='Adam_dec')
    train_enc_opt = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.5).minimize(loss_enc_opt, var_list=enc_vars
                                                                                , name='Adam_enc')

sess = tf.Session()

ckpt = tf.train.get_checkpoint_state(out_model_dir)
saver = tf.train.Saver()
if ckpt: # checkpointがある場合
    last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
    saver.restore(sess, last_model) # 変数データの読み込み
    print("load " + last_model)
else: # 保存データがない場合
    #init = tf.initialize_all_variables()    
    sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

log_list = []
log_list.append(['epoch', 'AUC'])
#training loop
for epoch in range(1):
    if epoch % VALID_SPAN == 0:
        score_A_np = np.zeros((0, 2), dtype=np.float32)
        val_data_num = len(make_datasets.valid_data)

        img_batch_test = make_datasets.get_valid_data_for_1_batch(0, val_data_num)
        score_A_ = sess.run(score_A, feed_dict={x_:img_batch_test, is_training_:False})
        score_A_re = np.reshape(score_A_, (-1, 1))
        tars_batch_re = np.where(score_A_re < SCORE_TH, 1, 0) #np.reshape(tars_batch, (-1, 1))

        score_A_np_tmp = np.concatenate((score_A_re, tars_batch_re), axis=1)

        x_z_x_test = sess.run(x_z_x, feed_dict={x_:img_batch_test, is_training_:False})
        #print(score_A_np_tmp)
        array_1_np, array_0_np = Utility.score_divide(score_A_np_tmp)
        
        Utility.make_score_hist_test(array_1_np, array_0_np, SCORE_TH, LOGFILE_NAME, OUT_HIST_DIR)
        Utility.make_output_img_test(img_batch_test, x_z_x_test, score_A_np_tmp, LOGFILE_NAME, OUT_IMG_DIR)    