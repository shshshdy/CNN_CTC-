from Model.Model_CNN_CTC import Model_CNN_CTC
from Model.Model_DenseNet_CTC import Model_DenseNet_CTC
from DataLoader.DataLoad_Model import DataLoad_Model_by_Path
import SuperParam
from Model.Model_DenseNet_CTC_HW import Model_DenseNet_CTC_HW

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

def training():
    pass
    a_model = Model_CNN_CTC()
    data_train = DataLoad_Model_by_Path(train_path)
    data_val = DataLoad_Model_by_Path(val_path)
    a_model.build_model()
    # a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
    #                         data_gen_val = data_val.gen_data_by_batch_ctc_yield,
    #                         epochs=8,
    #                         batch_size=4,
    #                         steps_per_epoch=1,
    #                         validation_steps = 1)

    a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
                            data_gen_val = data_val.gen_data_by_batch_ctc_yield,
                            epochs=100,
                            batch_size=32,
                            steps_per_epoch=900//32,
                            validation_steps = 20)

def training_densenet():
    pass
    a_model = Model_DenseNet_CTC()
    data_train = DataLoad_Model_by_Path(train_path)
    data_val = DataLoad_Model_by_Path(val_path)
    a_model.build_model()
    a_model.load_model_weight('saved_models-15092018-221604-e100.h5')
    # a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
    #                         data_gen_val = data_val.gen_data_by_batch_ctc_yield,
    #                         epochs=8,
    #                         batch_size=4,
    #                         steps_per_epoch=1,
    #                         validation_steps = 1)

    a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
                            data_gen_val = data_val.gen_data_by_batch_ctc_yield,
                            epochs=100,
                            batch_size=8,
                            steps_per_epoch=900//8,
                            validation_steps = 20)


def training_1():
    pass
    a_model = Model_DenseNet_CTC_HW()
    data_train = DataLoad_Model_by_Path(train_path)
    data_val = DataLoad_Model_by_Path(val_path)
    a_model.build_model()
    # a_model.load_model_weight('../saved_models-16092018-172058-e100.h5')
    # a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield,
    #                         data_gen_val = data_val.gen_data_by_batch_ctc_yield,
    #                         epochs=8,
    #                         batch_size=4,
    #                         steps_per_epoch=1,
    #                         validation_steps = 1)

    a_model.train_generator(data_gen_train = data_train.gen_data_by_batch_ctc_yield_hw,
                            data_gen_val = data_val.gen_data_by_batch_ctc_yield_hw,
                            epochs=100,
                            batch_size=8,
                            steps_per_epoch=900//8,
                            validation_steps = 10)

if __name__ == '__main__':
    # a_Model_CTC = Model_CNN_CTC()
    # a_Model_CTC.build_model()
    # print(a_Model_CTC.conv_shape)
    training_1()
