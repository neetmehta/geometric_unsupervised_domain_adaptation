import os
from os.path import join as osp
import pandas as pd

splits_dict = {}
root = r"E:\Deep Learning Projects\datasets\kitti_raw"

def num_parser(num):
    num1 = str(num)
    num2 = '0'*(10-len(num1))
    return num2+num1


def make_csv(data_path, split_path, split_name='eigen_full_train'):

    with open(split_path, 'r') as f:
        list1 = f.readlines()
        img_dict = {}
        for i in list1:
            path, num, pos = i.split()
            # path.replace('/','\\')
            img_path = osp(path, f"image_02/data/{num_parser(num)}.png").replace('\\','/')

            if path not in img_dict:
                img_dict[path] = []

            img_dict[path].append(int(num))

    sample_list = []
    for key,value in img_dict.items():
        
        img_dict[key] = list(set(img_dict[key]))
        img_dict[key].sort()

        for i in range(len(img_dict[key])-1):
            if osp(data_path,f"image_02/data/{num_parser(img_dict[key][i]-1)}.png").replace('\\','/') not in os.listdir(osp(data_path,key,'image_02','data')) or osp(data_path,f"image_02/data/{num_parser(img_dict[key][i]+1)}.png").replace('\\','/') not in os.listdir(osp(data_path,key,'image_02','data')):
                print(osp(data_path,key,f"image_02/data/{num_parser(img_dict[key][i]-1)}.png").replace('\\','/')," not found")
                continue
            sample = [osp(key,f"image_02/data/{num_parser(img_dict[key][i])}.png").replace('\\','/'), 
                      osp(key,f"image_02/data/{num_parser(img_dict[key][i]-1)}.png").replace('\\','/'), 
                      osp(key,f"image_02/data/{num_parser(img_dict[key][i]+1)}.png").replace('\\','/')]
            sample_list.append(sample)

    dataframe = pd.DataFrame(columns=['target', 'source_minus_1', 'source_1'], data=sample_list)
    dataframe.to_csv(f'{split_name}.csv')

            