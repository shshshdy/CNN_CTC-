'''
DenseNet_CTC 模型
'''

from keras.models import load_model
from keras import backend as K
from Model.Timer import Timer
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Lambda, Reshape
from keras.models import Model
import datetime as dt
from keras.callbacks import *
import numpy as np
from DataLoader.DataLoad_Model import DataLoad_Model_by_Path
import Model.densenet as densenet
import os

import SuperParam
characters = SuperParam.characters
height = SuperParam.height
width = SuperParam.width
n_class = SuperParam.n_class
seq_len = SuperParam.seq_len
n_len = SuperParam.n_len
batch_size = SuperParam.batch_size

train_path = SuperParam.train_path
val_path = SuperParam.train_path
train_label = SuperParam.train_label
val_label = SuperParam.train_label



class Model_DenseNet_CTC():

    def __init__(self):
        # self.model 初始
        # self.input_tensor = Input((SuperParam.height, SuperParam.width, 3))
        self.model = None
        self.basemodel = None
        self.conv_shape = None
        self.modelweight = None

    def load_model(self, filepath):
        # 模型导入
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def load_model_weight(self,filepath):
        # 模型权重导入
        print('[Model] Loading model Weight from file %s' % filepath)
        if os.path.exists(filepath):
            self.model.load_weights(filepath)

    def ctc_lambda_func(self,args):
        labels, y_pred, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(self):
        '''
        :param height:
        :param width:
        :param n_class: 字符种类
        :param seq_len: 预测字符最大长度
        :return:
        '''
        # 模型层次构建
        # configs 使用配置文件方式构建
        timer = Timer()
        timer.start()
        print('[Model] Model begin')

        ########
        input = Input((width, height, 3))
        pred_conv = densenet.dense_cnn(input, n_class)

        conv_shape = pred_conv.get_shape()
        self.conv_shape = conv_shape
        # x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        # pred_conv = Dense(n_class, activation='softmax')(x)
        self.basemodel = Model(inputs=input, outputs=pred_conv)

        print('-------------- base model -------------')
        print(self.basemodel.summary())

        # (*,11)
        labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
        # (*,1)
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        # (*,1)
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([labels, pred_conv, input_length, label_length])
        self.model = Model(input=[labels, input, input_length, label_length], output=[loss_out])
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

        print('-------------- model -------------')
        print(self.model.summary())
        #########

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size):
        # 模型训练
        pass

    def evaluate(self,base_model,batch_num=10):
        '''
        预测结果
        :param batch_num:
        :return:
        '''
        size = 8
        error_count = 0
        data = DataLoad_Model_by_Path(val_path)
        for i in range(batch_num):
            [y_test,X_test, _, _], _ = next(data.gen_data_by_batch_ctc_yield(batch_size=size,
                                    label_file=val_label,
                                    conv_shape=self.conv_shape
                                   ))

            y_pred = base_model.predict(X_test)
            shape = y_pred[:, :, :].shape
            ctc_decode = K.ctc_decode(y_pred[:, :, :],
                                      input_length=np.ones(shape[0]) * shape[1])[0][0]
            out = K.get_value(ctc_decode)
            for i in range(len(X_test)):
                str_src = y_test[i]
                str_out = ''.join([str(x) for x in out[i] if x != -1])
                # print('str_src={}  str_out={}'.format(str_src, str_out))
                if str_src != str_out:
                    error_count += 1
                else:
                    print('str_src={}  str_out={}'.format(str_src, str_out))
        return (batch_num*size - error_count) / (batch_num*size)


    def train_generator(self, data_gen_train, data_gen_val,epochs, batch_size, steps_per_epoch,validation_steps):
        # 生成方式训练
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = 'saved_models-%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs))
        checkpoint = ModelCheckpoint(save_fname,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=1)
        print('=================')
        print(type(self))
        evaluator = Evaluate(self)



        lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(10)])

        def step_decay(epoch):
            new_lrate = float(learning_rate[epoch])
            print('new_lrate:' + str(new_lrate))
            return new_lrate
        changelr = LearningRateScheduler(step_decay)  # 动态学习率

        callbacks = [
            evaluator,EarlyStopping(patience=10),checkpoint,changelr
        ]

        self.model.fit_generator(
            data_gen_train(batch_size=batch_size,label_file=train_label,conv_shape=self.conv_shape),
            steps_per_epoch = steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=data_gen_val(batch_size=batch_size,label_file=val_label,conv_shape=self.conv_shape),
            validation_steps=validation_steps
        )

        self.model.save('final_model.h5')

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()



class Evaluate(Callback):
    def __init__(self,ctc):
        self.accs = []
        self.ctc = ctc
        print(type(self.ctc))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            acc = self.ctc.evaluate(self.ctc.basemodel) * 100
            self.accs.append(acc)
            print('---------------------')
            print('acc: %f%%' % acc)



def training():
    pass
    a_model = Model_DenseNet_CTC()
    data_train = DataLoad_Model_by_Path(train_path)
    data_val = DataLoad_Model_by_Path(val_path)
    a_model.build_model()
    a_model.load_model_weight('../saved_models-16092018-172058-e100.h5')
    # a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
    #                         data_gen_val = data_val.gen_data_by_batch_ctc_yield,
    #                         epochs=8,
    #                         batch_size=4,
    #                         steps_per_epoch=1,
    #                         validation_steps = 1)

    a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield_norandom,
                            data_gen_val = data_val.gen_data_by_batch_ctc_yield_norandom,
                            epochs=100,
                            batch_size=8,
                            steps_per_epoch=900//8,
                            validation_steps = 10)

if __name__ == '__main__':
    # a_Model_CTC = Model_DenseNet_CTC()
    # a_Model_CTC.build_model()
    # print(a_Model_CTC.conv_shape)
    training()

