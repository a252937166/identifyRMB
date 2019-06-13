import os
import shutil
import csv


def my_move_file(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname=os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                # 创建路径
        shutil.move(srcfile, dstfile)          # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def my_copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname=os.path.split(dstfile)    # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                # 创建路径
        shutil.copyfile(srcfile, dstfile)      # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


csv_file = csv.reader(open('D:/competition/train_face_value_label.csv', 'r'))
iter_file = iter(csv_file)
next(iter_file)
for line in csv_file:
    print(line[0].strip()+'values:'+line[1].strip())
    my_copyfile('D:/competition/train_data/'+line[0].strip(), 'D:/competition/'+line[1].strip()+'/'+line[0].strip())

