import h5py  #导入工具包  
import numpy as np  
#HDF5的读取：  
f = h5py.File('train_291_31_x234.h5','r')   #打开h5文件  
print(f.keys())                            #可以查看所有的主键  
#HDF5的读取：
# 可以查看所有的主键
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    #print(f[key].value)
a = f['data'][:]                    #取出主键为data的所有的键值  
f.close() 
