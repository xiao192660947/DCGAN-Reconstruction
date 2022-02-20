'''
进行新数据的处理，用来进行马鞍面的生成
'''

import numpy as np
import pandas as pd
import random as ran
import math
import os
import argparse
import matplotlib.pyplot as plt
import random as ran

parser = argparse.ArgumentParser(description='')
parser.add_argument("--out_filename",default="D:/PycharmProjects/Generate_3D_GAN/dataset/profile_data/",help='dirs of input image3Ddata')#进行输入的原始数据文件#D:/PycharmProjects/Generate_3D_GAN/fold_categorical_180x150x120.txt




args = parser.parse_args()  # 用来解析命令行参数



#进行曲线的拟合，画出一个面的拟合曲线
def create_line(x,y,mode = 0):
    x = np.array(x)
    y = np.array(y)
    x_fit = np.array(range(0,128))
    #用三次多项式进行拟合
    if mode == 0:
        #拟合y值
        f1 = np.polyfit(x,y,3)
        y_fit = np.polyval(f1,x_fit)
        print(y_fit)
    y_fit = y_fit.reshape([128,1])
    return y_fit


#进行属性赋值，将小于的值赋为一个颜色，大于的赋值为另一个属性
#这个输入值当中的y_fit为x，y方向上面的拟合曲线
def fill_color(y_fit_list):
    full_image = np.zeros([128,128])
    m,n,s = y_fit_list.shape
    print("列表的行列数 = ",m,n,s)

    nature_list = np.array(range(1,s))
    nature_list = nature_list

    nature_strart = 7



    for j in range(0,5):
        temp_band = []
        for i in range(0, 128):

            print("sdfa",s,i)
            bond1 = int(y_fit_list[i, 0, j])
            bond2 = int(y_fit_list[i, 0, j + 1])
            temp_band.append(bond2)
            #bond = int(y_fit[i, 0])
            # # full_image[i, 0:bond1] = nature0
            # if bond1 - bond2 <= 1:
            #     bond1 = bond2
            full_image[i, bond1:bond2] = nature_list[j]

    for i in range(0, 128):
        full_image[i, temp_band[i]:128] = nature_strart



    return full_image



#进行曲面的拟合，将拟合出的曲线沿着z轴进行扩充生成剖面
#这个输入当中的y_fit为z轴方向上面的拟合曲线
def create_surface(z_fit,y_fit_list):
    # full_3D_image = np.zeros([128,128,128])
    # for i in range(0,128):
    #     increase = z_fit[i,0]
    #     y_fit_i = y_fit[] + increase
    #     full_image = fill_color(y_fit_i)
    #     full_3D_image[:,:,i] = full_image

    full_3D_image = np.zeros([128,128,128])
    # for i in range(0, 128):
    #     increase = z_fit[i, 0]
    #     y_fit_list_i = y_fit_list + increase
    #     full_image = fill_color(y_fit_list_i)
    #     full_3D_image[:, :, i] = full_image

    for i in range(0, 128):
        increase = z_fit[i, :]
        y_fit_list_i = y_fit_list + increase
        full_image = fill_color(y_fit_list_i)
        full_3D_image[:, :, i] = full_image





    return full_3D_image



def transorm_to_txt1(arr,name_No):
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




if __name__ == "__main__":
    x = [0,30,50,60,80,120,128]
    y1 = [0,10,30,24,27,0,0]
    y2 = [20,30,37,40,44,45,50]
    y3 = [30,45,47,40,44,45,50]
    y4 = [50,60,77,80,70,80,70]
    y5 = [70,70,80,85,85,85,90]
    y6 = [80,110,120,110,117,117,117]

    y_fit_list = np.zeros([128,1,6])

    y_fit = create_line(x,y1,mode=0)
    y_fit_list[:,:,0] = y_fit
    y_fit = create_line(x,y2,mode=0)
    y_fit_list[:,:,1] = y_fit
    y_fit = create_line(x,y3, mode=0)
    y_fit_list[:,:,2] = y_fit
    y_fit = create_line(x,y4,mode=0)
    y_fit_list[:,:,3] = y_fit
    y_fit = create_line(x,y5,mode=0)
    y_fit_list[:,:,4] = y_fit
    y_fit = create_line(x,y6, mode=0)
    y_fit_list[:,:,5] = y_fit

    print(y_fit_list,y_fit_list.shape)

    x = [0, 30, 50, 60, 80, 120, 128]
    z1 = [0, 10,3, 15, 20, 3, 7]
    z2 = [0, 6,5, 8, 13, 3, 7]
    z3 = [0, 4,6, 9, 13, 3, 7]
    z4 = [0, 7,7, 10, 13, 3, 7]
    z5 = [0, 8,3, 1, 13, 3, 7]
    z6 = [0, 4,6, 0, 13, 3, 7]

    #z_fit_list = np.zeros([128, 1,6])
    z_fit_list = np.zeros([128, 6])

    z_fit = create_line(x, z1, mode=0)
    print(z_fit.shape)
    z_fit_list[:, 0] = z_fit.reshape([128])
    z_fit = create_line(x, z2, mode=0)
    z_fit_list[:, 1] = z_fit.reshape([128])
    z_fit = create_line(x, z3, mode=0)
    z_fit_list[:, 2] = z_fit.reshape([128])
    z_fit = create_line(x, z4, mode=0)
    z_fit_list[:, 3] = z_fit.reshape([128])
    z_fit = create_line(x, z5, mode=0)
    z_fit_list[:, 4] = z_fit.reshape([128])
    z_fit = create_line(x, z6, mode=0)
    z_fit_list[:, 5] = z_fit.reshape([128])
    # z_fit = create_line(x, z6, mode=0)
    # z_fit_list[:, :, 2] = z_fit

    print(z_fit_list, z_fit_list.shape)

    full_3D_image = create_surface(z_fit_list,y_fit_list)
    print(full_3D_image,full_3D_image.shape)
    transorm_to_txt1(full_3D_image,"newest2")






