'''
用来进行网络的使用，将定义的网络进行使用并通过loss函数进行计算
主要函数：

create_save_file(save_path,train_out_path):创建保存模型的文件夹
save_file(save_data,sess,log_dir,step):保存文件
get_trainning_result(trainning_3Dimage,gan_label,div_path,counter):将训练得到的矩阵进行输出到指定文件夹，得到训练的对比结果（先输出为.npy到指定文件夹,直接存放三维arr）
calcuate_lc_loss(gen_label,train_label,K,M):定义上下文的损失函数
clacuate_loss(train_label,gen_output,dis_output_fake,dis_output_real):计算总损失loss，并直接包含计算gen的loss
trainning_main():进行训练的主函数



'''
from __future__ import  absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import math
import os
import random
from random import shuffle
import cv2
import glob
import sys
import argparse


#网络结构组件
from net import *
#数据处理
from perpare_input_dataset import *
#剖面数据处理
from handel_Profile_data import *

'''
设定各类参数项及超参数项
'''
parser = argparse.ArgumentParser(description='')

args = parser.parse_args()#用来进行解析命令行的参数
parser.add_argument("--snapshot_dir",default='./model_save',help='path of snapshots')#模型保存路径
parser.add_argument("--out_dir",default='./train_out',help='path of training outputs')#训练后的生成文件输出路径
parser.add_argument("--out_trainningdata_dir",default='./dataset/train_3Dimagedata_out',help='path of trainning_out 3D data')#训练后的文件输出路径
parser.add_argument("--image_size", type=int, default=64, help="load image size") #网络输入的尺度
parser.add_argument("--image_size_z",type=int,default=64,help="load image size")#网络输入的z的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.002, help='initial learning rate for adam') #对生成器的学习率
parser.add_argument('--base_lr_dis', type=float, default=0.0002, help='initial learning rate for adam') #对判别器的学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  #训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=5, help="times to write.") #训练中每过多少step保存结果
parser.add_argument("--save_pred_every", type=int, default=1000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--lamda_lc_weight", type=float, default=0.0, help="Lc lamda") #训练中Lc_Loss前的乘数
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda") #训练中GAN_Loss前的乘数
parser.add_argument("--train_data_format", default='.txt', help="format of training datas.") #网络训练输入的数据的格式(图片在CGAN中被当做条件)    暂定为txt文件
parser.add_argument("--train_label_format", default='.txt', help="format of training labels.") #网络训练输入的标签的格式(标签在CGAN中被当做真样本)    暂定为txt文件
parser.add_argument("--train_data_path", default='./dataset/train_3Dimage/', help="path of training datas.") #网络训练输入的图片路径
parser.add_argument("--train_label_path", default='./dataset/train_label/', help="path of training labels.") #网络训练输入的标签路径
parser.add_argument("--profile_path",default='./dataset/profile_data/',help="parh of profile data")#进行上下文loss计算相关性的时候需要的剖面的数据
parser.add_argument("--profile_data_size",default=86400,help="num of profile data")#进行上下文loss计算相关性的时候需要的剖面的数据(有关位置的信息数据)
parser.add_argument("--profile_label_size",default=86400,help="num of profile label")#进行上下文loss计算相关性的时候需要的剖面的数据（有关对应位置的标签的数据）
parser.add_argument("--profile_data_format",default=".txt",help="format of profile file")#进行上下文loss计算相关性的时候需要的剖面的数据（有关对应位置的标签的数据）
parser.add_argument("--full_pixel_value", default=10, help="value to full no profile")  #填充没有剖面的地方的像素所使用的值


args = parser.parse_args()#解析命令行
EPS = 1e-12#用来保证log的参数不为负


'''
创建保存模型的文件夹
'''
def create_save_file(save_path,train_out_path):
    if not os.path.exists(save_path):#模型的保存路径
        os.makedirs(save_path)
    if not os.path.exists(train_out_path):#训练的输出路径（目前先定为txt形式输出）
        os.makedirs(train_out_path)

'''
保存文件
'''
def save_file(save_data,sess,log_dir,step):
    model_name = 'model' #定义前缀形式
    checkpoint_path = os.path.join(log_dir,model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_data.save(sess,checkpoint_path,step)
    print('checkpoint创建完毕已保存')



'''
得到训练的对比结果(直接存放txt文件）
'''
def get_trainning_result(trainning_3Dimage,gan_label,div_path,counter):#将训练得到的矩阵进行输出到指定文件夹
    # #进行保存为npy文件
    # trainning_3Dimage_path = div_path + "/out_trainning_3Dimage" + str(counter) + ".npy"
    # gan_label_path = div_path + "/gen_label" + str(counter) + ".npy"
    # np.save(trainning_3Dimage_path,trainning_3Dimage)
    # np.save(gan_label_path,gan_label)


    #进行保存为txt文件
    savepath = div_path + "/out_trainning_3Dimage" + str(counter) + ".txt"
    gan_label_path = div_path + "/gen_label" + str(counter) + ".txt"

    #进行归一化矩阵还原
    print("输出的生成矩阵的大小 = ",trainning_3Dimage.shape,gan_label.shape,type(gan_label))
    trainning_3Dimage = (trainning_3Dimage + 1.) * 5
    gan_label = (gan_label + 1.) * 5

    arr = trainning_3Dimage
    arr_gan = gan_label
    #进行trainning3Dimage的保存
    # num = arr.shape[0] * arr.shape[1] * arr.shape[2]
    # print("节点总数",num)
    f = open(savepath, 'a')
    f.write(str(arr.shape[0]) + '\t' + str(arr.shape[1]) + '\t' + str(arr.shape[2]) + '\n')
    f.write(str(1) + '\n')
    f.write(str("v"))
    f.close()
    f = open(savepath, 'a')
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            for k in range(0,arr.shape[2]):
                #value = abs(arr[0][i][j][k])
                value = arr[i][j][k]
                #print("其中的值的大小",value)
                if value - int(arr[i][j][k]) >= 0.5:
                    value = int(arr[i][j][k]) + 1
                else:
                    value = int(arr[i][j][k])

                f.write('\n' + str(value))
    f.close()

    #进行GAN生成结果的保存
    #num = arr_gan.shape[0] * arr_gan.shape[1] * arr_gan.shape[2]
    #print("节点总数",num)
    f = open(gan_label_path, 'a')
    f.write(str(arr_gan.shape[1]) + '\t' + str(arr_gan.shape[2]) + '\t' + str(arr_gan.shape[3]) + '\n')
    f.write(str(1) + '\n')
    f.write(str("v"))
    f.close()
    f = open(gan_label_path, 'a')
    for i in range(0,arr_gan.shape[1]):
        for j in range(0,arr_gan.shape[2]):
            for k in range(0,arr_gan.shape[3]):
                #value = abs(arr_gan[0][i][j][k])
                value = arr_gan[0][i][j][k]
                print("其中的值的大小",value)
                if value - int(arr_gan[0][i][j][k]) >= 0.5:
                    value = int(arr_gan[0][i][j][k]) + 1
                else:
                    value = int(arr_gan[0][i][j][k])

                f.write('\n' + str(value))
    f.close()


'''
定义第一层的l1_loss
'''
def l1_loss(src,dst):
    return tf.reduce_mean(tf.abs(src - dst))

'''

存在问题：如何将tensor当中数值转换成相应的可使用的索引以及value / 如何直接将tensorflow转换成numpy当中的数组
 
计算lc_loss（负责与元数据的契合程度）   不知道其中的钻进点数据如何得到
另外：（备用）计算上下文的loss函数，以后可能要加一个卷积块(就是解码器E）这样就变为了bicycle_GAN_
'''
#gen_label为生成的数据，train_label为训练的原始对应数据点，K为属性值，M为此属性的已知点的location，
#此loss的主要思想为将某一相的所有的已知钻井点和生成器生成的数据点之间的偏移距离总和
#或者为此方法使用的为剖面的数据，就可以将剖面的数据当作相应的二位数据和原有的剖面数据进行对应，从而使用gan生成的数据当中的相应剖面和原有的元数据的剖面零个二维数组之间计算相关性
'''
def calcuate_lc_loss(gen_label,image3D_data,image3D_label):
    # profile_location_data = image3D_data
    # profile_label_data = image3D_label
    # num_sum = profile_location_data.shape[0]
    # print("转换以后的格式 = ",gen_label.shape)
    # print("剖面的设计的控制点的总数 = ",num_sum)
    # print("输入的数据以及最终label的shape = ",profile_location_data.shape,profile_label_data.shape)
    # gen_profile_data = zeros(num_sum)
    # print("用于盛放gen数据的矩阵大小 = ",gen_profile_data.shape)
    # for i in range(num_sum):
    #     print("进行循环的回合数 = ",i)
    #     x_i = tf.gather_nd(profile_location_data,[i,0])
    #     y_i = tf.gather_nd(profile_location_data,[i,1])
    #     z_i = tf.gather_nd(profile_location_data,[i,2])
    #     print("包含的数据location信息 ： x_i, y_i, z_i", x_i, y_i, z_i)
    #     print("对应的数据点 gen_label[i]", gen_label[i])
    #     gen_profile_data[i] = gen_label[0,x_i - 1, y_i - 1, z_i - 1,0]  #提取出gen生成的三维模型的相应位置处的属性数值0/1，变为剖面矩阵
    #     print("gen生成的对应的剖面数据 = gen_profile_data[i]",gen_profile_data[i])
    # #进行两个矩阵之间的属性相关性计算并生成lc_loss(目前使用的为进行两个矩阵之间的对比并且将不同的值进行提取以后将loss函数定义为 不同的点的总数/已知剖面点的总数)
    # gen_profile_data = set(gen_profile_data)
    # image3D_label = set(image3D_label)
    #
    # different_point = gen_profile_data & image3D_label
    # different_num = len(different_point)
    #
    # lc_loss = different_num/len(image3D_label)

    # lc_loss = tf.reduce_mean(l1_loss(gen_label, train_label)
    lc_loss = 0

    return lc_loss
'''
'''
计算跳出每一个三维模型中的剖面行列数
'''
def find_col_row(train_data):
    #查看最初的train_data的大小尺寸
    print("查看最初的train_data的大小尺寸",train_data.shape)
    #初始化剖面类list
    pro_loc_list_x = []
    pro_loc_list_y = []
    pro_loc_list_z = []
    num = 0
    #找出x轴相应的剖面列（利用不是剖面列的填充地区的像素点为full_pixel_value)
    for i in range(64):
        if train_data[0,i,10,10,0] != 1:
            #print("每次挑选出的位置的value",train_data[0,i,1,1,0])
            pro_loc_list_x.append(i)
            num = num + 1
    print("最终取出的X方向的剖面列的列表 = ",pro_loc_list_x)
    #找出y轴相应的剖面列（利用不是剖面列的填充地区的像素点为full_pixel_value)
    for j in range(64):
        if train_data[0,10,j,10,0] != 1:
            pro_loc_list_y.append(j)
            num = num + 1
    print("最终取出的Y方向的剖面列的列表 = ", pro_loc_list_y)
    #找出z轴相应的剖面列（利用不是剖面列的填充地区的像素点为full_pixel_value)
    for k in range(64):
        if train_data[0,10,10,k,0] != 1:
            pro_loc_list_z.append(k)
            num = num + 1
    print("最终取出的Z方向的剖面列的列表 = ", pro_loc_list_z)

    print("取出的剖面列表一共有几个 = ",num)

    return pro_loc_list_x,pro_loc_list_y,pro_loc_list_z,num


'''
进行loss计算
'''
def cal_loss(gen_output,dis_output_fake,dis_output_real,train_data,pro_loc_list_x,pro_loc_list_y,pro_loc_list_z,num):
    # 计算loss
    # gen_loss_GAN = tf.reduce_mean(-tf.log(dis_output_fake + EPS))
    loss_list = tf.Variable(tf.zeros([1]))  # 用以存放计算出来的相关性loss
    no = 0
    #初始化其中的取出剖面作为放如ssim函数中的剖面
    # temp_pro_gen = tf.Variable(tf.zeros([64, 64, 1]))
    # temp_pro_label = tf.Variable(tf.zeros([64, 64, 1]))
    for i in range(len(pro_loc_list_x.get_shape().as_list())):
        if(len(pro_loc_list_x.get_shape().as_list())== 0):
            break
        temp_pro_gen = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_label = tf.Variable(tf.zeros([64, 64, 1]))

        # 进行相关性计算（其中的为取出相应的label中的已知原剖面矩阵数据，取出对应的gen_label中的对应剖面矩阵数据）
        # lc_loss = tf.image.ssim(gen_output[0:pro_loc_list_x[i], :, :, 0], train_data[0:pro_loc_list_x[i], :, :, 0], 1)
        temp_pro_gen = tf.assign(temp_pro_gen[:, :, 0],gen_output[0, tf.gather(pro_loc_list_x,i), :, :, 0])
        temp_pro_label = tf.assign(temp_pro_label[:, :, 0],train_data[0, tf.gather(pro_loc_list_x,i), :, :, 0])
        lc_loss = 1 - tf.reduce_mean(tf.image.ssim(temp_pro_label, temp_pro_gen, 1.0))
        print("lc_loss",lc_loss)
        loss_list = tf.concat([loss_list,tf.reshape(lc_loss,[1])],0)
        no = no + 1
    for j in range(len(pro_loc_list_y.get_shape().as_list())):
        if(len(pro_loc_list_y.get_shape().as_list()) == 0):
            break
        # 进行相关性计算（其中的为取出相应的label中的已知原剖面矩阵数据，取出对应的gen_label中的对应剖面矩阵数据）
        # lc_loss = tf.image.ssim(gen_output[0:pro_loc_list_y[j], :, :, 0], train_data[0:pro_loc_list_y[j], :, :, 0], 1)
        temp_pro_gen = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_label = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_gen = tf.assign(temp_pro_gen[:, :, 0],gen_output[0, :, tf.gather(pro_loc_list_y,j), :, 0])
        temp_pro_label = tf.assign(temp_pro_label[:, :, 0],train_data[0, :, tf.gather(pro_loc_list_y,j), :, 0])
        lc_loss = 1 - tf.reduce_mean(tf.image.ssim(temp_pro_label, temp_pro_gen, max_val=1.0))
        #loss_list = tf.assign(loss_list[no + j, 0],lc_loss)
        loss_list = tf.concat([loss_list,tf.reshape(lc_loss,[1])],0)
        no = no + 1
    for k in range(len(pro_loc_list_z.get_shape().as_list())):
        if(len(pro_loc_list_z.get_shape().as_list()) == 0):
            break
        # 进行相关性计算（其中的为取出相应的label中的已知原剖面矩阵数据，取出对应的gen_label中的对应剖面矩阵数据）
        # lc_loss = tf.image.ssim(gen_output[0,:, :, pro_loc_list_z[k], 0], train_data[0,:, :, pro_loc_list_z[k], 0], 1)
        temp_pro_gen = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_label = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_gen = tf.assign(temp_pro_gen[:, :, 0],gen_output[0, :, :, tf.gather(pro_loc_list_z,k), 0])
        temp_pro_label = tf.assign(temp_pro_label[:, :, 0],train_data[0, :, :, tf.gather(pro_loc_list_z,k), 0])
        lc_loss = 1 - tf.reduce_mean(tf.image.ssim(temp_pro_label, temp_pro_gen, max_val=1.0))
        #loss_list = tf.assign(loss_list[no + k, 0],lc_loss)
        loss_list = tf.concat([loss_list, tf.reshape(lc_loss,[1])], 0)

    print("最终输出的loss_list模块的值为", loss_list)
    #loss_list = tf.reduce_mean(lc_loss)

    # gen_loss_lcloss = lcloss
    # gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_lcloss * args.lamda_lc_weight
    # dis_loss = tf.reduce_mean(-(tf.log(dis_output_real + EPS) + tf.log(1 - dis_output_fake + EPS)))

    return loss_list,lc_loss#,gen_loss,dis_loss


'''
进行训练的主函数
'''


def trainning_main():
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True  # 设定显存不超量
    # sess = tf.Session(config=config)  # 建立会话
    # init = tf.global_variables_initializer()#新建初始化信息
    # sess.run(init)#初始化参数

    tf.set_random_seed(args.random_seed)  # 初始化随机数
    create_save_file(args.snapshot_dir, args.out_dir)  # 创建参数文件夹
    train_data_list = glob.glob(os.path.join(args.train_data_path, '*'))  # 训练输入的路径列表、

    # 进行三维数据的读取和tf占位
    train_data = tf.placeholder(tf.float32,
                                shape=[1, args.image_size, args.image_size, args.image_size_z, 3],
                                name='train_data')  # 输入训练图像
    train_label = tf.placeholder(tf.float32,
                                 shape=[1, args.image_size, args.image_size, args.image_size_z, 3],
                                 name='train_label')  # 输入训练图像标签

    pro_loc_list_x = tf.placeholder(tf.int32,shape=[None],name='pro_loc_list_x')
    pro_loc_list_y = tf.placeholder(tf.int32, shape=[None], name='pro_loc_list_y')
    pro_loc_list_z = tf.placeholder(tf.int32, shape=[None], name='pro_loc_list_z')


    # 生成器输出
    gen_output = Generator(image_3D=train_data)
    # 判别器的判别结果
    dis_output_real,_ = Discriminator(pro_loc_list_x, pro_loc_list_y,pro_loc_list_z,image_3D=train_data, targets=train_label, df_dim=64, reuse=False,
                                    name="discriminator")  # 返回真实标签结果
    print("*" * 50)
    print("gen_output = ", gen_output.shape)
    print("train_data = ", train_data.shape)
    print("dis_output_real = ", dis_output_real)
    dis_output_fake,loss_list = Discriminator(pro_loc_list_x, pro_loc_list_y,pro_loc_list_z,image_3D=train_data, targets=gen_output, df_dim=64, reuse=True,
                                    name="discriminator")  # 返回生成标签的对比结果、
    print("dis_output_fake = ", dis_output_fake)
    print("*" * 50)
    ###############################################################################################################################################################################################
    '''

            temp_pro_label[:,:,0].assign(image3D_data_new[pro_loc_list_x[j],:,:])#首先取出相应的label中的已知原剖面矩阵数据
            temp_pro_gen[:,:,0].assign(gen_output[pro_loc_list_x[j],:,:])#其次取出对应的gen_label中的对应剖面矩阵数据

            # gen_pro_data = np.concatenate((gen_pro_data,temp_pro_gen),axis=2)
            # label_pro_data = np.concatenate((label_pro_data,temp_pro_label),axis=2)

        print("其中的image3D_data_new【64，64，64】的值为 = ",image3D_data_new)
        print("最终取出的两个剖面结果 = ",temp_pro_gen,temp_pro_label)
        temp_pro_label = tf.Variable(tf.zeros([64, 64, 1]))
        temp_pro_gen = tf.Variable(tf.zeros([64, 64, 1]))

    # gen_pro_data = np.array([64,64,1])
    # label_pro_data = np.array([64,64,1])#初始化相应的存放数组
    '''
    #计算loss
    loss_list, lc_loss = cal_loss(gen_output, dis_output_fake, dis_output_real, train_data, pro_loc_list_x, pro_loc_list_y,pro_loc_list_z,10)

    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_output_fake + EPS))
    #gen_loss_lcloss = tf.reduce_mean(loss_list)
    gen_loss_lcloss = 0 #tf.reduce_mean(loss_list)
    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_lcloss * args.lamda_lc_weight
    dis_loss = tf.reduce_mean(-(tf.log(dis_output_real + EPS) + tf.log(1 - dis_output_fake + EPS)))

    ##################################################################################################################################################################################################
    print("@" * 100)
    print("计算出来的loss值 = ", gen_loss, dis_loss)
    print("@" * 100)
    gen_loss_sum = tf.summary.scalar('gen_loss', gen_loss)  # 显示标量信息
    dis_loss_sum = tf.summary.scalar('dis_loss', dis_loss)
    summary_write = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())  # 记录日志

    gen_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 载入已经经过训练的模型数据（此操作的原因是将所有的可训练变量加载进var中）
    dis_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

    # 进行梯度训练
    g_optim = tf.train.AdamOptimizer(args.base_lr, beta1=args.beta1)
    d_optim = tf.train.AdamOptimizer(args.base_lr_dis, beta1=args.beta1)
    g_grads_vars = g_optim.compute_gradients(gen_loss, var_list=gen_vars)  # 计算生成器训练的梯度
    g_train = g_optim.apply_gradients(g_grads_vars)  # 更新训练参数
    d_grads_vars = d_optim.compute_gradients(dis_loss, var_list=dis_vars)  # 计算判别器的梯度
    d_train = d_optim.apply_gradients(d_grads_vars) #将compute_gradients()返回的值作为输入参数对变量更新

    train_op = tf.group(d_train, g_train)  # 对多个操作进行分组

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True#设定显存不超量
    sess = tf.Session(config=config)#建立会话
    init = tf.global_variables_initializer()  # 新建初始化信息
    sess.run(init)  # 初始化参数

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)  # 保存模型

    counter = 0  # 记录训练步数

    for epoch in range(args.epoch):
        shuffle(train_data_list)
        for step in range(len(train_data_list)):
            counter += 1
            # 进行训练数据的读取
            data_name, _ = os.path.splitext(os.path.basename(train_data_list[step]))
            data_resize, label_resize = handle_3Ddata(trainning_data_path=args.train_data_path,
                                                      label_data_path=args.train_label_path,
                                                      trainning_data_name=data_name,
                                                      training_data_format=args.train_data_format,
                                                      label_data_format=args.train_label_format,
                                                      standard_x=args.image_size,
                                                      standard_y=args.image_size,
                                                      standard_z=args.image_size_z)
            # 进行剖面数据的读取
            pro_list_x, pro_list_y, pro_list_z, numsum = find_col_row(train_data=data_resize)
            # 计算loss
            # loss_list, lc_loss, gen_loss_v, dis_loss_v = cal_loss(gen_output, dis_output_fake, dis_output_real, train_data,pro_list_x, pro_list_y, pro_list_z, numsum)


            banch_data = data_resize
            banch_label = label_resize

            # ,pro_loc_list_x : pro_list_x,pro_loc_list_y : pro_list_y,pro_loc_list_z : pro_list_z,num : numsum

            feed_dict = {train_data: banch_data, train_label: banch_label ,pro_loc_list_x : pro_list_x,pro_loc_list_y : pro_list_y,pro_loc_list_z : pro_list_z}  # 进行项占位项的填补
            # feed_dict = {train_data: [[1,2,3],t],
            # print("输出的形状检查train_data",train_data.shape)
            # print("输出检查banch_data",banch_data.shape)
            # 计算每个生成器中的gen和抵受的loss   lcloss_list,lcloss,loss_list,lc_loss,
            lloossttlliisstt,listy,gen_loss_value, dis_loss_value, _ = sess.run([loss_list,pro_loc_list_y,gen_loss, dis_loss, train_op],
                                                         feed_dict=feed_dict)
            # print("()"*100)
            # print("计算生成的最终loss结果",gen_loss_value,dis_loss_value)
            # print("()" * 100)

            if counter % args.save_pred_every == 0:
                save_file(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0:
                gen_loss_sum_value, dis_loss_sum_value = sess.run([gen_loss_sum, dis_loss_sum],
                                                                  feed_dict=feed_dict)
                summary_write.add_summary(gen_loss_sum_value, counter)
                summary_write.add_summary(dis_loss_sum_value, counter)
            if counter % args.write_pred_every == 0:
                gen_label_value = sess.run(gen_output, feed_dict=feed_dict)
                # 进行训练数据的输出和保存
                get_trainning_result(label_resize, gen_label_value, args.out_dir, counter)
            print('epoch {:d} step {:d} '.format(epoch, step))
            #print("loss_test = ",llcctt)
            # print("loss_list = ",lc_list)
            print("最终计算出的上下下文损失 = ",lloossttlliisstt)
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step,
                                                                                                         gen_loss_value,
                                                                                                         dis_loss_value,
                                                                                                         ))
            writer = tf.summary.FileWriter("log_1", tf.get_default_graph())
            writer.close()


if __name__ == "__main__":
    #print(args.out_trainningdata_dir)
    trainning_main()

























