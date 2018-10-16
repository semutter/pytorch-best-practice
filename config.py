#coding:utf8
import warnings

class DefaultConfig(object):
    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class DogsCatsConfig(DefaultConfig):
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/dogscats/train' # 训练集存放路径
    val_data_root = './data/dogscats/valid'   # 验证集存放路径
    test_data_root = './data/dogscats/test1' # 测试集存放路径
    train_max_num = 1000  # 配置
    val_max_num = 1000
    test_max_num = 1000
    load_model_path = None # 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128 # batch size
    use_gpu = False # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数


class MnistConfig(DefaultConfig):
    env = 'mnist'  # visdom 环境
    model = 'MLPNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    load_model_path = None  # 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb

    max_epoch = 30
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

