'''
超参数信息
'''


import string
import platform

characters = '0123456789'
height = 85
width = 350
n_class = 11  # 36种字符
batch_size = 32
seq_len = 11 # 验证码长度 4self
n_len = 11  # 验证码长度 4self
no_random = 1

whatos = platform.architecture()

if whatos[1] == 'WindowsPE':
    train_path = r'C:\004_project\007-ocr\ai_model\DATA\training_4'
    val_path = r'C:\004_project\007-ocr\ai_model\DATA\training_3'
    train_label = r'C:\004_project\007-ocr\ai_model\DATA\label_4\label.txt'
    val_label = r'C:\004_project\007-ocr\ai_model\DATA\label_3\label.txt'
else:
    train_path = r'/root/home/ai/ai_model/DATA/training_3'
    val_path = r'/root/home/ai/ai_model/DATA/training_3'
    train_label = r'/root/home/ai/ai_model/DATA/label_3/label.txt'
    val_label = r'/root/home/ai/ai_model/DATA/label_3/label.txt'

class SuperParam():
    def __init__(self):
        pass

if __name__ == '__main__':
    print(platform.architecture())