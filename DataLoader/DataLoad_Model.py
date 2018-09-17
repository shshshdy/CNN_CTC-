'''
对数据进行处理类
'''

import os
import numpy as np
import cv2
import string

'''
处理文件系统上的文件
'''

DEBUG = False

import SuperParam
characters = SuperParam.characters
height = SuperParam.height
width = SuperParam.width
n_class = SuperParam.n_class
seq_len = SuperParam.seq_len
n_len = SuperParam.n_len
batch_size = SuperParam.batch_size
charset = SuperParam.characters
no_random = SuperParam.no_random

class DataLoad_Model_by_Path(object):

    def __init__(self,rootdir):

        self.rootdir = rootdir
        if not os.path.exists(rootdir):
            raise Exception(rootdir + ' is not exists')

        # 文件清单
        self.FilePathList = []
        self.labels = {}


    def get_path_list(self):
        '''
        :return: file list
        '''
        self.FilePathList = []
        for fpathe, dirs, fs in os.walk(self.rootdir):
            for f in fs:
                FilePath = os.path.join(fpathe, f)
                if os.path.isfile(FilePath):
                    self.FilePathList.append(FilePath)
        return self.FilePathList

    '''
    label就是文件名 多分类
    '''
    def gen_data_by_batch(self,batch_size):
        '''
        :param batch_size: 批次大小
        :param (height,width,n_class,n_len): input 图片 (height,width) n_class: onehot中几分类 n_len:最长识别数
        :return: [X,y]
        '''

        # 获取文件清单
        if self.FilePathList == []:
            self.get_path_list()
        # print(self.FilePathList)
        FilePathList = self.FilePathList
        FilePathList_Len = len(FilePathList)
        FileIndexs = np.random.randint(0, FilePathList_Len, batch_size)

        img_i = 0
        X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]

        for file_i in FileIndexs:
            img = cv2.imread(FilePathList[file_i])
            img_y = os.path.basename(FilePathList[file_i])
            img_y_labe = img_y.split('.')[0]

            X[img_i,:] = cv2.resize(img,(width,height))
            for j, ch in enumerate(img_y_labe):
                y[j][img_i, :] = 0
                y[j][img_i, characters.find(ch)] = 1
            img_i = img_i + 1
        return (X,y)


    def handle_numpy_img(self,numpy_file,pic_path):
        '''
        将numpy_file 转成图片保存到 pic_path
        :return:
        '''
        X = np.load(numpy_file)
        len = X.shape[0]
        for i in range(len):
            img = X[i,:,:,:]
            a_path = os.path.join(pic_path,str(i) + '.jpg')
            cv2.imwrite(a_path,img)

    def handle_numpy_label(self,numpy_file,label_file):
        '''
        将numpy_file 中label信息转成txt保存
        :param numpy_file:
        :param label_file:
        :return:
        '''
        X = np.load(numpy_file)
        len = X.shape[0]
        f = open(label_file,'w')
        for i in range(len):
            a_label = X[i, :,].tolist()
            a_label_str = '|'.join(str(e) for e in a_label)
            a_label_str = a_label_str.replace('11','')
            a_label_str = a_label_str.replace('|', '')
            a_filename = str(i)+'.jpg'
            f.writelines(a_filename+'|@|'+a_label_str+'\n')
        f.close()

    def _text_to_labels(self,text):
        '''
        将tabel-string 信息 转成ctc格式信息
        :param text:
        :param charset:
        :param seq_len: 最大长度
        :param label_count: 空白表征
        :return:
        '''
        lab = []
        label_count = n_class
        text = text.replace('.', '')
        for char in text:
            if char is not '\n':
                lab.append(charset.find(char))
        if len(lab) < seq_len:
            cur_seq_len = len(lab)
            for i in range(seq_len - cur_seq_len):
                lab.append(label_count)  # 补齐
        return lab

    def gen_image_data(self, label_file,img_name,label_name,img_path):
        '''
        根据label.txt 将 img 和 label 信息 保存为numpy
        :param label:
        :param img_name:
        :param label_name:
        :return:
        '''
        X = []
        Y = []
        for i, line in enumerate(open(label_file, encoding='utf-8')):
            words = line.split('|@|')
            file = os.path.join(img_path,words[0])
            # file = './imgs/' + words[0] + '.jpg'
            img = cv2.imread(file)
            X.append(img)
            Y.append(self._text_to_labels(words[1]))
            print('第' + str(i) + '张')
        np.save(img_name, np.asarray(X))
        np.save(label_name, np.asarray(Y))

    def gen_label_dict_ctc(self,label_file):
        '''
        获取label_file文件 生成ctc格式的label dict
        :param label_file:
        :return:
        '''
        with open(label_file,'r') as f:
            for line in f:
                a_label = line.split('|@|')
                key = a_label[0]
                value = self._text_to_labels(a_label[1])
                self.labels[a_label[0]] = value

    def gen_data_by_batch_ctc(self,batch_size,label_file,conv_shape):
        '''
        生成ctc风格数据
        :param batch_size:
        :param height:
        :param width:
        :param n_class:
        :param n_len:
        :param characters:
        :return:
        '''

        # 获取文件清单 加载lebel dict
        if self.labels == {}:
            self.gen_label_dict_ctc(label_file)
        if self.FilePathList == []:
            self.get_path_list()
        #print(self.FilePathList)
        FilePathList = self.FilePathList
        FilePathList_Len = len(FilePathList)
        FileIndexs = np.random.randint(0, FilePathList_Len, batch_size)

        img_i = 0
        X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        for file_i in FileIndexs:
            if DEBUG:
                print(FilePathList[file_i])
            #file_name = os.path.basename(FilePathList[file_i]).split('.')[0]
            file_name = os.path.basename(FilePathList[file_i])
            img_y_labe = self.labels[file_name]
            ####################################
            #img_y_labe = img_y_labe[0:n_len]
            ####################################
            if img_y_labe == '':
                raise  Exception(file_name + ' not have label')
            img = cv2.imread(FilePathList[file_i])

            X[img_i, :] = cv2.resize(img, (height, width))
            y[img_i] = img_y_labe
            img_i = img_i + 1

        return [y,X,np.ones(batch_size) * int(conv_shape[1]),
                np.ones(batch_size)*n_len], np.ones(batch_size)

    def gen_data_by_batch_ctc_norandom(self,batch_size,label_file,conv_shape):
        '''
        生成ctc风格数据
        :param batch_size:
        :param height:
        :param width:
        :param n_class:
        :param n_len:
        :param characters:
        :return:
        '''
        global no_random
        # 获取文件清单 加载lebel dict
        if self.labels == {}:
            self.gen_label_dict_ctc(label_file)
        if self.FilePathList == []:
            self.get_path_list()
        #print(self.FilePathList)
        FilePathList = self.FilePathList
        FilePathList_Len = len(FilePathList)
        FileIndexs=[]
        for i in range(batch_size):
            temp = ( i* batch_size + no_random) % (FilePathList_Len-1)
            FileIndexs.append ( temp )
        # print(FileIndexs)
        img_i = 0
        X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        for file_i in FileIndexs:
            if DEBUG:
                print(FilePathList[file_i])
            #file_name = os.path.basename(FilePathList[file_i]).split('.')[0]
            file_name = os.path.basename(FilePathList[file_i])
            img_y_labe = self.labels[file_name]
            ####################################
            #img_y_labe = img_y_labe[0:n_len]
            ####################################
            if img_y_labe == '':
                raise  Exception(file_name + ' not have label')
            img = cv2.imread(FilePathList[file_i])

            X[img_i, :] = cv2.resize(img, (height, width))
            y[img_i] = img_y_labe
            img_i = img_i + 1

        no_random = no_random + 1

        return [y,X,np.ones(batch_size) * int(conv_shape[1]),
                np.ones(batch_size)*n_len], np.ones(batch_size)


    def gen_data_by_batch_ctc_hw(self,batch_size,label_file,conv_shape):
        '''
        生成ctc风格数据
        :param batch_size:
        :param height:
        :param width:
        :param n_class:
        :param n_len:
        :param characters:
        :return:
        '''

        # 获取文件清单 加载lebel dict
        if self.labels == {}:
            self.gen_label_dict_ctc(label_file)
        if self.FilePathList == []:
            self.get_path_list()
        #print(self.FilePathList)
        FilePathList = self.FilePathList
        FilePathList_Len = len(FilePathList)
        FileIndexs = np.random.randint(0, FilePathList_Len, batch_size)

        img_i = 0
        X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        for file_i in FileIndexs:
            if DEBUG:
                print(FilePathList[file_i])
            #file_name = os.path.basename(FilePathList[file_i]).split('.')[0]
            file_name = os.path.basename(FilePathList[file_i])
            img_y_labe = self.labels[file_name]
            ####################################
            #img_y_labe = img_y_labe[0:n_len]
            ####################################
            if img_y_labe == '':
                raise  Exception(file_name + ' not have label')
            img = cv2.imread(FilePathList[file_i])

            X[img_i, :] = cv2.resize(img, (width, height))
            y[img_i] = img_y_labe
            img_i = img_i + 1

        return [y,X,np.ones(batch_size) * int(conv_shape[1]),
                np.ones(batch_size)*n_len], np.ones(batch_size)

    def gen_data_by_batch_ctc_yield(self,batch_size,label_file,conv_shape):
        while True:
            yield self.gen_data_by_batch_ctc(batch_size,label_file ,conv_shape)

    def gen_data_by_batch_ctc_yield_hw(self,batch_size,label_file,conv_shape):
        while True:
            yield self.gen_data_by_batch_ctc_hw(batch_size,label_file ,conv_shape)

    def gen_data_by_batch_ctc_yield_norandom(self,batch_size,label_file,conv_shape):
        while True:
            yield self.gen_data_by_batch_ctc_norandom(batch_size,label_file ,conv_shape)

class A_Testing():
    def __init__(self,rootpath):
        self.a_object = DataLoad_Model_by_Path(rootpath)

    def atest_get_path_list(self):
        print(os.getcwd())
        print(self.a_object.get_path_list())

    def atest_get_data_by_batch(self):

        characters = string.digits + string.ascii_uppercase
        result = self.a_object.gen_data_by_batch(1)
        print(result)

    def atest_handle_numpy_img(self):
        self.a_object.handle_numpy_img('../DATA/training_2/train_image.npy','../DATA/training_4')

    def atest_handle_numpy_label(self):
        self.a_object.handle_numpy_label('../DATA/label_2/train_label.npy','../DATA/label_4/label.txt')

    def atest_gen_image_data(self):
        characters = string.digits + string.ascii_uppercase
        self.a_object.gen_image_data('../DATA/label_3/label.txt',
                            '../DATA/training_2/val_image_test.npy',
                            '../DATA/training_2/val_label_test.npy',
                            '../DATA/training_3'
                            )

    def atest_gen_data_by_batch_ctc(self):
        characters = string.digits + string.ascii_uppercase
        return self.a_object.gen_data_by_batch_ctc(batch_size=1,
                                        label_file = '../DATA/label_3/label.txt',
                                        conv_shape = [17,12]
                                        )

    def atest_gen_data_by_batch_ctc_hw(self):
        characters = string.digits + string.ascii_uppercase
        return self.a_object.gen_data_by_batch_ctc_hw(batch_size=1,
                                        label_file = '../DATA/label_3/label.txt',
                                        conv_shape = [17,12]
                                        )

    def atest_gen_data_by_batch_ctc_norandom(self):
        characters = string.digits + string.ascii_uppercase
        return self.a_object.gen_data_by_batch_ctc_norandom(batch_size=32,
                                        label_file = '../DATA/label_3/label.txt',
                                        conv_shape = [17,12]
                                        )

if __name__ == '__main__':

    # test get_path_list
    a = A_Testing('../DATA/training_3')
    # a.atest_get_path_list()
    # a.atest_get_data_by_batch()
    # a.atest_handle_numpy_img()
    # a.atest_handle_numpy_label()
    # a.atest_gen_image_data()
    print(a.atest_gen_data_by_batch_ctc_hw())

    # for i in range(100):
    #     a.atest_gen_data_by_batch_ctc_norandom()

