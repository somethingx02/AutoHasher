# -*- coding: utf8 -*-

#ROOT_DIR = '.' #the root path, currently is the code execution path
ROOT_DIR = '/media/data1/lixing/2017_2018_2_Aston/ElmoProj/FeatureHashingSenti'

TRAIN_SET_PATH = '%s/datasets/clinical_reviews_train.csv'%ROOT_DIR
TEST_SET_PATH = '%s/datasets/clinical_reviews_test.csv'%ROOT_DIR

SAVE_DIR = '%s/save'%ROOT_DIR

TRAINING_INSTANCES = 10000
TESTING_INSTANCES = 5739
DOCUMENT_SEQ_LEN = 1209
CHARACTER_EMBEDDING_LEN = 50

CLASS_COUNT = 5 # number of classes for classification  

class DefaultConfig():
    '''
    default config for training parameters
    '''
    batch_size = 16 # 256 best for 01 #if if 2048 then cuda memory exceeds
    epochs = 8 # since the batch size is 16, then 16 epoch is enough
    learning_rate = 0.0005 # 0.0005 initial best, learning rate initialize should depend on the batch size
    lr_decay = 0.9
    weight_decay = 1e-4
    model = 'BiLSTMAttention'
    
    #on_cuda = False # if this is false then run CPU
    on_cuda = True # if this is True then run GPU

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # tdnn_kernel=[(1,25),
    #             (2,50),
    #             (3,75),
    #             (4,100),
    #             (5,125),
    #             (6,150),
    #             (7,175)],
    # highway_size=700,
    # rnn_hidden_size=650,
    # dropout':0.0
    def set_attrs(self,kwargs):
        '''
        kwargs is a dict
        '''
        for k,v in kwargs.items():
            setattr(self,k,v)#inbuilt function of python, set the attributes of a class object. For example: setattr(oDefaultConfig,epochs,50) <=> oDefaultConfig.epochs = 50
    
    def get_attrs(self):
        '''
        the enhanced getattr, returns a dict whose key is public items in an object
        '''
        
        attrs = {} #attrs = dict()
        for k , v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'set_attrs' and k != 'get_attrs' :
                attrs[k] = getattr( self , k) #get the attr in an object
        return attrs

if __name__=='__main__':
    config=DefaultConfig()
    print(config.get_attrs())
    config.set_attrs( { 'epochs':100 , 'batch_size' : 16 } )
    print(config.get_attrs())