'''
用来进行基本的卷积和反卷积网络的定义以及构建生成器和判别器
其中的主要定义网络组件为;
1.3D卷积层:cov3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "conc3d",biased = False)
2.3D反卷积层:decon3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "conv3d")
3.归一化BN层：batch_norm(input_,name = 'batch_norm')
4.激活层：lrelu(x,leak = 0.2,name = 'relu'):
5.生成器结构：Generator(image_3D,gf_dim = 64,reuse = False,name = 'generator'):      #将生成的output_dim设定为64的倍数也就是chennels为64个filter
6.判别器结构：Discriminator(image_3D,targets,df_dim = 64,reuse = False,name = 'discriminator'):
'''


import tensorflow as tf
import numpy as np
import pandas as pan
import math
import matplotlib as plt


'''
构造可训练参数
'''
def make_var(name,shape,trainable = True):
    return tf.get_variable(name,shape,trainable=trainable)


'''
定义卷积层

使用卷积的时候应该注意进行3D卷积的时候三个维度的尺寸应该一样
tf.nn.conv3d(input, filter, strides, padding, data_format=None, name=None)
input:就是输入的数据，必须是float32，float64类型的。shape=[batch_size, in_depth, in_height, in_width, in_channels]batch是每次输入的视频样本数；in_depth是每个视频样本的帧数；in_height,in_width:视频中每个帧的长和宽，类似图像的分辨率。是一个5-D张量
filter:shape=[filter_depth, filter_height, filter_width, in_channels, out_channels],是一个Tensor，必须和input有一样的shape；
stride:shape=[strides_batch,strides_depth,strides_height,strides_width,strides_channels],是一个长度为五的一维张量，是输入张量每一维的滑动窗口跨度的大小；一般情况下strides_batch=strides_channels=1。
padding：参数有SAME or VALID，代表不同的填充；SAME表示使用0填充；VALID表示不用0填充
data_format:代表的是输入数据和输出数据每一维都指代的参数，有"NDHWC"，默认值是NDHWC，数据存储的顺序是[batch, in_depth, in_height, in_width, in_channels]; "HCDHW"则是，[batch,  in_channels, in_depth, in_height,in_width]
name = 取一个名字
Return:和输入参数一样shape的张量
'''
'''
定义卷积层（第三维度的数据可以类比于视频处理的关键帧）
'''
def conv3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "conv3d",biased = False):
    input_dim = input_.get_shape()[-1]#读出输入层的维度
    with tf.variable_scope(name):
        kernal = make_var(name = 'weights',shape = [2,kernel_size,kernel_size,input_dim,output_dim]) #定义卷积块
        output = tf.nn.conv3d(input_,
                              kernal,
                              [1,stride,stride,stride,1],
                              padding=padding)      #定义卷积过程
        if biased:
            biases = make_var(name = 'biases',shape=[output_dim])  #偏差
            output = tf.nn.bias_add(output,biases) #将偏差加到value上面

    return output

'''
定义反卷积层

'''
def deconv3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "deconv3d"):
    input_dim = input_.get_shape()[-1]
    input_x = int(input_.get_shape()[1])#x,y,z表示输入层相应维度的值x:height,y:width,z:chennel
    input_y = int(input_.get_shape()[2])
    input_z = int(input_.get_shape()[3])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights',shape = [2,kernel_size,kernel_size,output_dim,input_dim])
        print("input_ = ",input_)
        output = tf.nn.conv3d_transpose(input_,
                                        kernel,
                                        [1,input_z*2,input_x*2,input_y*2,output_dim],
                                        [1,2,2,2,1],
                                        padding=padding)

    return output


'''
定义归一化函数BN层(维度问题需要检查)
'''
def batch_norm(input_,name = 'batch_norm'):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        print("控制点，检查是否为后两项的计算失误",name,input_dim)
        scale = tf.get_variable("scale",
                                [input_dim],
                                initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
        offset = tf.get_variable("offset",
                                 [input_dim],
                                 initializer=tf.constant_initializer(0.0))
        print("offset,scale = ",offset,scale)
        mean,variance = tf.nn.moments(input_,axes=[1,2,3],keep_dims=True)
        print("mean,variance = ",mean,variance)
        epsilom = 1e-5
        inv = tf.rsqrt(variance + epsilom)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        test = scale * normalized
        print("各种变量的维度","scale",scale.shape,"offset",offset.shape,"normalized",normalized.shape,"相乘的维度",test.shape,"最终output的维度",output.shape)
        return output


'''
激活层：利用leakyrelu函数激活避免梯度爆炸和消失
'''
def lrelu(x,leak = 0.2,name = 'relu'):
    return tf.maximum(x,leak * x)  #relu函数本质上就是一个取大值的函数


'''
进行生成器的输出（128的数据两层卷积的U-Net）
'''
def Generator(image_3D,gf_dim = 64,reuse = False,name = 'generator'):
    input_dim = int(image_3D.get_shape()[-1])
    dropout_rate = 0.5
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        #卷积层的输入的维度为128的话就是将卷积两次最终的一层的维度为32

        #进行下采样
        print("第一个网络开始前的维度", image_3D)
        #第一个卷积层输出：（1*32*32*32*64）
        e1 = batch_norm(conv3D(input_=image_3D, output_dim=gf_dim, kernel_size=4, stride=2, name='g_conv_e1'),
                        name='g_bn_e1')
        print("第一层网络结束后的维度", e1.shape)
        # 第二个卷积层（1*16*16*16*128）
        e2 = batch_norm(conv3D(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_conv_e2'),
                        name='g_bn_e2')
        print("第二层网络结束后的维度", e2.shape)
        # 第三个卷积层(1*8*8*8*256)
        e3 = batch_norm(conv3D(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_conv_e3'),
                        name='g_bn_e3')
        print("第三层网络结束后的维度", e3.shape)
        # 第四个卷积层（1*4*4*4*512）
        e4 = batch_norm(conv3D(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e4'),
                        name='g_bn_e4')
        print("第四层网络结束后的维度", e4.shape)
        # 第五个卷积层（1*2*2*2*512）
        e5 = batch_norm(conv3D(input_=lrelu(e4), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e5'),
                        name='g_bn_e5')
        print("第五层网络结束后的维度", e5.shape)
        # 第六个卷积层（1*1*1*1*512）
        e6 = batch_norm(conv3D(input_=lrelu(e5), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e6'),
                        name='g_bn_e6')
        print("进行最终卷积以后的输出 = ", e6.shape)


        # 下采样结束，开始进行反卷积上采样
        #进行第一次反卷积（1*2*2*2*512）
        d1 = deconv3D(input_=tf.nn.relu(e6), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)  # 随机抛去无用层
        d1 = tf.concat([batch_norm(input_=d1, name='g_bn_d1'), e5], 4)
        print("反卷积第一层", d1.shape)

        #进行第二次反卷积（1*4*4*4*512）
        d2 = deconv3D(input_=tf.nn.relu(d1), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
        d2 = tf.concat([batch_norm(input_=d2, name='g_bn_d2'), e4], 4)
        print("反卷积第二层 = ", d2.shape)

        #进行第三次反卷积（1*8*8*8*256）
        d3 = deconv3D(input_=tf.nn.relu(d2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_deconv_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
        d3 = tf.concat([batch_norm(input_=d3, name='g_bn_d3'), e3], 4)
        print("反卷积第二层 = ", d3.shape)

        #进行第二次反卷积（1*16*16*16*128）
        d4 = deconv3D(input_=tf.nn.relu(d3), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_deconv_d4')
        d4 = tf.nn.dropout(d4, dropout_rate)  # 随机扔掉一般的输出
        d4 = tf.concat([batch_norm(input_=d4, name='g_bn_d4'), e2], 4)
        print("反卷积第二层 = ", d4.shape)

        #进行第二次反卷积（1*32*32*32*64）
        d5 = deconv3D(input_=tf.nn.relu(d4), output_dim=gf_dim, kernel_size=4, stride=2, name='g_deconv_d5')
        d5 = tf.nn.dropout(d5, dropout_rate)  # 随机扔掉一般的输出
        d5 = tf.concat([batch_norm(input_=d5, name='g_bn_d5'), e1], 4)
        print("反卷积第二层 = ", d5.shape)

        #反卷积最后一层（1*64*64*64*3）
        d_final = deconv3D(input_=tf.nn.relu(d5),output_dim=input_dim,kernel_size=4,stride=2,name='g_deconv_d6')
        print("最终反卷积的结果 = ",d_final)


        return tf.nn.tanh(d_final)


'''
定义generator：U-Net的生成结构（旧，网络开销过大）
'''
# def Generator(image_3D,gf_dim = 64,reuse = False,name = 'generator'):      #将生成的output_dim设定为64的倍数也就是chennels为64个filter
#     input_dim = int(image_3D.get_shape()[-1]) #获取输入的维度
#     dropout_rate = 0.5  #定义dropout的概率
#     with tf.variable_scope(name):
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse is False
#
#         #卷积的层数暂时按照256*256的输入来决定
#         #第一个卷积层
#         print("第一个网络开始前的维度",image_3D)
#         e1 = batch_norm(conv3D(input_=image_3D,output_dim=gf_dim,kernel_size=4,stride=2,name='g_conv_e1'),
#                         name='g_bn_e1')
#         print("第一层网络结束后的维度",e1.shape)
#         # 第二个卷积层
#         e2 = batch_norm(conv3D(input_=lrelu(e1), output_dim=gf_dim*2, kernel_size=4, stride=2, name='g_conv_e2'),
#                         name='g_bn_e2')
#         print("第二层网络结束后的维度",e2)
#         # 第三个卷积层
#         e3 = batch_norm(conv3D(input_=lrelu(e2), output_dim=gf_dim*4, kernel_size=4, stride=2, name='g_conv_e3'),
#                         name='g_bn_e3')
#         # 第四个卷积层
#         e4 = batch_norm(conv3D(input_=lrelu(e3), output_dim=gf_dim*8, kernel_size=4, stride=2, name='g_conv_e4'),
#                         name='g_bn_e4')
#         # 第五个卷积层
#         e5 = batch_norm(conv3D(input_=lrelu(e4), output_dim=gf_dim*8, kernel_size=4, stride=2, name='g_conv_e5'),
#                         name='g_bn_e5')
#         # 第六个卷积层
#         e6 = batch_norm(conv3D(input_=lrelu(e5), output_dim=gf_dim*8, kernel_size=4, stride=2, name='g_conv_e6'),
#                         name='g_bn_e6')
#         # 第七个卷积层
#         e7 = batch_norm(conv3D(input_=lrelu(e6), output_dim=gf_dim*8, kernel_size=4, stride=2, name='g_conv_e7'),
#                         name='g_bn_e7')
#         #第八个卷积层
#         e8 = batch_norm(conv3D(input_=lrelu(e7), output_dim=gf_dim*8, kernel_size=4, stride=2, name='g_conv_e8'),
#                         name='g_bn_e8')
#         print("最终卷积层",e8)
#         #下采样结束，开始进行反卷积上采样
#         d1 = deconv3D(input_=tf.nn.relu(e8),output_dim=gf_dim*8,kernel_size=4,stride=2,name='g_deconv_d1')
#         d1 = tf.nn.dropout(d1,dropout_rate)#随机抛去无用层
#         d1 = tf.concat([batch_norm(input_=d1,name='g_bn_d1'),e7],4)
#         print("e7的维度 = ",e7.shape)
#         print("asdfasdf",d1)
#         #把多个array沿着某一个维度接在一起,将反卷积的第一层和卷积的最后一层进行连接（原因为如果只进行上采样的话会丢失很多信息）
#         #可以理解为沿着chennels的方向进行的结合，因为前面的大小为1*1*1*1*512，有5位，以0开始就是最终的合并维度在最后一维
#         #反卷积第2层
#         print("反卷积第一层",d1)
#         d2 = deconv3D(input_=tf.nn.relu(d1), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d2')
#         d2 = tf.nn.dropout(d2, dropout_rate)
#         d2 = tf.concat([batch_norm(input_=d2, name='g_bn_d2'), e6], 4)
#         print("反卷积第2层", d2)
#         # 反卷积第3层
#         d3 = deconv3D(input_=tf.nn.relu(d2), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d3')
#         d3 = tf.nn.dropout(d3, dropout_rate)
#         d3 = tf.concat([batch_norm(input_=d3, name='g_bn_d3'), e5], 4)
#         print("反卷积第三层", d3)
#         # 反卷积第4层
#         d4 = deconv3D(input_=tf.nn.relu(d3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d4')
#         d4 = tf.concat([batch_norm(input_=d4, name='g_bn_d4'), e4], 4)
#         # 反卷积第5层
#         d5 = deconv3D(input_=tf.nn.relu(d4), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_deconv_d5')
#         d5 = tf.concat([batch_norm(input_=d5, name='g_bn_d5'), e3], 4)
#         # 反卷积第6层
#         d6 = deconv3D(input_=tf.nn.relu(d5), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_deconv_d6')
#         d6 = tf.concat([batch_norm(input_=d6, name='g_bn_d6'), e2], 4)
#         # 反卷积第7层
#         d7 = deconv3D(input_=tf.nn.relu(d6), output_dim=gf_dim, kernel_size=4, stride=2, name='g_deconv_d7')
#         d7 = tf.concat([batch_norm(input_=d7, name='g_bn_d7'), e1], 4)
#         # 反卷积第8层
#         d8 = deconv3D(input_=tf.nn.relu(d7),output_dim=input_dim,kernel_size=4,stride=2,name='g_deconv_d8')
#         print("#"*50)
#         print("最终的生成器输出维度",d8)
#         return  tf.nn.tanh(d8)   #双曲正切激活函数，用来保证网络的训练稳定性

'''
进行loss计算
'''
def cal_loss(gen_output,train_data,pro_loc_list_x,pro_loc_list_y,pro_loc_list_z):
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
定义discriminator：四层（暂定，不知道3D的生成精准度比2D高多少,层数不能太深避免训练失衡）
'''
#参数之中的image_3D为输入的原数据（train_data），targets是目标数据(train_label/gen_label)
def Discriminator(pro_loc_list_x,pro_loc_list_y,pro_loc_list_z,image_3D,targets,df_dim = 64,reuse = False,name = 'discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_input = tf.concat([image_3D,targets],4)#不是很明白为什么要进行合并，找其他的卷积神经网络看一下
        print("判别器输入的数据为 = ",dis_input.shape)
        #第一层卷积
        print("判别器第一层网络输入前的维度",dis_input)
        dis0 = lrelu(conv3D(input_=dis_input,output_dim=df_dim,kernel_size=4,stride=2,name='dis_con_0'))
        print("判别器第一层网络结束后的维度",dis0)
        #第2层卷积

        dis1 = lrelu(batch_norm(conv3D(input_=dis0,output_dim=df_dim*2,kernel_size=4,stride=2,name='dis_conv_1'),name='dis_bn_1'))
        print("判别器第二层网络后的维度", dis1)
        #第3层卷积

        dis2 = lrelu(batch_norm(conv3D(input_=dis1,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_2'),name='dis_bn_2'))
        print("第三层网络结束后的维度",dis2)
        #第4层卷积

        dis3 = lrelu(batch_norm(conv3D(input_=dis2,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_3'),name='dis_bn_3'))
        print("第三层网络结束后的维度",dis3)
        #最终层卷积

        dis_output = conv3D(input_=dis3,output_dim=1,kernel_size=4,stride=1,name='dis_conv_output')
        print("最终层网络结束后的维度",dis_output)
        #经过sigmoid层进行运算，作用为进行二分类运算，输出的结果为分类的结果
        dis_output = tf.sigmoid(dis_output)
        print("最终结果",dis_output)

        #计算loss
        loss_list,lc_loss = cal_loss(targets, image_3D, pro_loc_list_x, pro_loc_list_y, pro_loc_list_z)

        return dis_output,loss_list




























