
'''
进行转换，就是将.npy的文件直接转换为相应的txt文件，便于软件查看
'''

import numpy as np
import os
import argparse
from handel_Profile_data import *

argparseparser = argparse.ArgumentParser(description='')

parser.add_argument("--out_filename",default="D:/PycharmProjects/Generate_3D_GAN/test_output/",help="path of output test datas.")  #输出的文件夹路径
parser.add_argument("--input_npy_path1",default="D:/PycharmProjects/Generate_3D_GAN/train_out/gen_label1500.npy",help="path of input npyfiles")#输入的文件夹中的npy文件路径
parser.add_argument("--input_npy_path2",default="D:/PycharmProjects/Generate_3D_GAN/train_out/out_trainning_3Dimage5000.npy",help="path of input npyfiles")#输入的文件夹中的npy文件路径
parser.add_argument("--input_npy_path3",default="D:/PycharmProjects/Generate_3D_GAN/dataset/profile_data/after_handel2.npy",help="path of input npyfiles")#输入的文件夹中的npy文件路径
parser.add_argument("--input_npy_path4",default="D:/PycharmProjects/Generate_3D_GAN/dataset/profile_data/1.npy",help="path of input npyfiles")#输入的文件夹中的npy文件路径


args = parser.parse_args()  # 用来解析命令行参数


def loadnpy(filepath):
    arr = np.load(filepath)
    #print("未进行转换的原始array = ",arr)
    return arr

#行医输出路径和进行转换
def transorm_to_txt1(arr,filename,name_No):
    savepath = args.out_filename + str(name_No) + ".txt"
    num = arr.shape[0] * arr.shape[1] * arr.shape[2]
    print("节点总数",num)
    # 检查文件夹是否创建,并进行创建文件文件
    # if not os.path.exists(filename):
    #     os.makedirs(filename)

    f = open(savepath, 'a')
    f.write(str(arr.shape[0]) + '\t' + str(arr.shape[1]) + '\t' + str(arr.shape[2]))
    f.write("\n" + str(1))
    f.write("\n" + str("v"))
    f.close()
    f = open(savepath, 'a')
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            for k in range(0,arr.shape[2]):

                f.write('\n' + str(int(arr[i][j][k])))

    f.close()

#进行输出路径转换并且将预测值换为相应的0或1的值
def transorm_to_txt2(arr,filename,name_No):
    savepath = args.out_filename + str(name_No) + ".txt"
    num = arr.shape[0] * arr.shape[1] * arr.shape[2]
    print("节点总数",num)
    # 检查文件夹是否创建,并进行创建文件文件


    # if not os.path.exists(filename):
    #     os.makedirs(filename)

    f = open(savepath, 'a')
    f.write(str(arr.shape[1]) + '\t' + str(arr.shape[2]) + '\t' + str(arr.shape[3]) + '\n')
    f.write(str(1) + '\n')
    f.write(str("v"))
    f.close()
    f = open(savepath, 'a')
    for i in range(0,arr.shape[1]):
        for j in range(0,arr.shape[2]):
            for k in range(0,arr.shape[3]):
                value = abs(arr[0][i][j][k])
                print("其中的值的大小",value)
                # if value >= 0.5:
                #     value = 1
                # else:
                #     value = 0

                f.write('\n' + str(value * 100))
    f.close()



if __name__ == "__main__":
    # data_arr, data_label = read_profile("D:/PycharmProjects/Generate_3D_GAN/px.txt")
    # print(data_arr.shape, data_label.shape)
    # print(data_arr[0][1])
    # complete_profile_data = prepar_profile_data(data_arr, data_label)
    # print('填充给完毕后的含剖面训练准备数据：', complete_profile_data, complete_profile_data.shape)
    # transorm_to_txt(complete_profile_data,filename=args.out_filename)
    #arr = loadnpy(args.input_npy_path1)
    arr = loadnpy(args.input_npy_path4)
    #print("原始数据的大小 = ",arr.shape,type(arr))
    #arr = arr[:,:,:,:,-1]
    print("原始数据的大小 = ", arr.shape)
    #transorm_to_txt2(arr,args.out_filename,1500)
    transorm_to_txt1(arr,args.out_filename,"1")


































#重写循环

            #print("s删除以后order list的大小 = ",order_list.shape)
            #np.random.shuffle(order_list)#shuffle放到循环里面每一次都shuffle一次
            x = order_list[0,0]
            y = order_list[0,1]
            #print("此次进行模拟的点位置（左上角点） = ",x,y)

            need_fill_template = []#用来临时装一列的模板值，后面的时候直接进行concatenate构造总的数组并append到train_x上面

            #模板中心待模拟点的位置
            simulation_point_x = int(x) + int(template_size/2)
            simulation_point_y = int(y) + int(template_size/2)

            #进行判断带模拟的像素点区域上是否有原始的已知点，如果有的话就不再进行模拟值的填充
            if test_image[simulation_point_x,simulation_point_y] != args.full_pixel_value:
                #删除相应的已经进行过模拟的点
                order_list = np.delete(order_list,0,axis=0)
                #结束此次循环
                continue


            for n in range(template_size):
                for m in range(template_size):
                    #计算模板尚未知对应到原图像上的位置
                    x_location = int(x) + m
                    y_location = int(y) + n
                    value = input_image[x_location,y_location]
                    #print("最终的value = ",value)
                    need_fill_template.append(int(input_image[x_location,y_location]))
                    #print("显示出的dataset的数值每一项",need_fill_template)

            #构建测试数据集（就是相应的周围点的值并将中心点去除）
            test_tamplate_data = np.delete(need_fill_template,int(template_size/2),axis=0)


            # 如果待模拟点没有值的话就进行填充
            test_tamplate_data = test_tamplate_data.reshape([1, template_size * template_size - 1])
            # print("最终进入模型进行识别的数据组大小 = ",test_tamplate_data,test_tamplate_data.shape)
            belong_class, class_score_pro = model_classify(model, test_tamplate_data)
            simulation_point_value = belong_class
            # print("最终网络训练出的中心点的值为 = ",simulation_point_value)
            image_out[simulation_point_x, simulation_point_y] = simulation_point_value

            # 删除相应的已经进行过模拟的点
            order_list = np.delete(order_list, 0, axis=0)

'''
#print("s删除以后order list的大小 = ",order_list.shape)
            #np.random.shuffle(order_list)#shuffle放到循环里面每一次都shuffle一次
            x = order_list[0,0]
            y = order_list[0,1]
            #print("此次进行模拟的点位置（左上角点） = ",x,y)

            need_fill_template = []#用来临时装一列的模板值，后面的时候直接进行concatenate构造总的数组并append到train_x上面

            #模板中心待模拟点的位置
            simulation_point_x = int(x) + int(template_size/2)
            simulation_point_y = int(y) + int(template_size/2)

            #进行判断带模拟的像素点区域上是否有原始的已知点，如果有的话就不再进行模拟值的填充
            if test_image[simulation_point_x,simulation_point_y] != args.full_pixel_value:
                #删除相应的已经进行过模拟的点
                order_list = np.delete(order_list,0,axis=0)
                #结束此次循环
                continue


            for n in range(template_size):
                for m in range(template_size):
                    #计算模板尚未知对应到原图像上的位置
                    x_location = int(x) + m
                    y_location = int(y) + n
                    value = input_image[x_location,y_location]
                    #print("最终的value = ",value)
                    need_fill_template.append(int(input_image[x_location,y_location]))
                    #print("显示出的dataset的数值每一项",need_fill_template)

            #构建测试数据集（就是相应的周围点的值并将中心点去除）
            test_tamplate_data = np.delete(need_fill_template,int(template_size/2),axis=0)


            # 如果待模拟点没有值的话就进行填充
            test_tamplate_data = test_tamplate_data.reshape([1, template_size * template_size - 1])
            # print("最终进入模型进行识别的数据组大小 = ",test_tamplate_data,test_tamplate_data.shape)
            belong_class, class_score_pro = model_classify(model, test_tamplate_data)
            simulation_point_value = belong_class
            # print("最终网络训练出的中心点的值为 = ",simulation_point_value)
            image_out[simulation_point_x, simulation_point_y] = simulation_point_value

            # 删除相应的已经进行过模拟的点
            order_list = np.delete(order_list, 0, axis=0)
'''


















































