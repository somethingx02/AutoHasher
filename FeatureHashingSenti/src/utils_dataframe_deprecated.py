# -*- coding: utf8 -*- 

import logging
from settings import CLASS_LEVEL1
from settings import * #import all the variables and classes in .py
import settings #you shall write setting.CLASS_LEVEL1 to use the variable in .py
import os,sys
import pandas
import numpy

def get_logger(name):
    '''
    set and return the logger modula,
    output: std
    '''
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def latest_save_num():
    if not os.path.exists( SAVE_DIR ):
        os.makedirs( SAVE_DIR )
    files = os.listdir( SAVE_DIR ) #list all the files in save dir
    maxno = -1
    # print(files)
    for f in files:
        if os.path.isdir(SAVE_DIR+'/'+f):
            # print(f,'true')
            try:
                # print(name)
                no=int(f)
                maxno=max(maxno,no)
                # print(maxno)
            except Exception as e:
                print( e.message ,file = sys.stderr)
                pass
    return maxno

class DataManager():
    '''
    data loading module
    '''
    #__df = None # not use this format of data initialization, possible confusion
    #__cursor = 0

    def __init__(self , param_batch_size):
        #self.df=None
        #self.cursor=0
        #self.batch_size=batch_size 

        #global private variable initialized here
        self.__dataframe_of_pandas = None #dataframe created by pandas, load from file once
        self.__current_cursor_in_dataframe = 0 #the cursor shifted in pandas
        self.__batch_size = param_batch_size

        #self.__batch_size = param_batch_size
        #self.__current_cursor = 0 #the current cursor in file, is the line num
        #self.__current_file_pointer = None list_all_tweets_of_auser#the current file pointer

    def generate_csv_from_wordembbed( self, param_filepath_in ):
        '''
        use this if you only have wordembbed file and TRAIN_SET_PATH = '.wordembbed'
        '''
        fp_out_csv = open( param_filepath_in + '.csv', 'wt', encoding = 'utf8')
        fp_in_wordembbed = open( param_filepath_in , 'rt', encoding = 'utf8')

        for aline in fp_in_wordembbed:
            aline = aline.strip()
            ( user_id, level1_allocation, level2_allocation ) = aline.split('\t')
            fp_out_csv.write( level1_allocation + '\t' + level2_allocation)
            for i in range(0,1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT):
                atwitterline = fp_in_wordembbed.readline()
                atwitterline = atwitterline.strip()
                fp_out_csv.write( '\t' + atwitterline )
            fp_out_csv.write('\n')

        fp_out_csv.close()
        fp_in_wordembbed.close()

    def load_dataframe_from_file( self, param_filepath_in):
        '''
        read once, initialize dataframe_of_pandas
        '''
        chunk_size = self.__batch_size #chunk size is the batch size

        self.__dataframe_of_pandas = pandas.read_csv( param_filepath_in, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c', chunksize= chunk_size)
        
        #self.__dataframe_of_pandas = pandas.read_csv( param_filepath_in, header = None, encoding = 'utf8',  sep = '\t' , engine = 'c') # you can use regular expression in sep by setting engine = 'python'
        #engine = 'c' will face error, may be 'c' needs 0 0 0 to be 0.0 0.0 0.0

        #c do not support \t\n
        #print(len(self.__dataframe_of_pandas) )
        #currently every 9 lines describe a user

    def dataframe_size(self):
        if isinstance( self.__dataframe_of_pandas , pandas.DataFrame ):
            return len(self.__dataframe_of_pandas)

    def next_batch(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        '''

        batch_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        batch_x = numpy.zeros( ( batch_size, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )
        for user_i in range(s,t):
            #each user's time serial
            label_shift = 1 #one col for label
            for timeserial_i in range( 0, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT):
                timeserial_shift = timeserial_i* ( 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT)* (TWITTER_LENGTH)* (WORD_EMBEDDING_DIMENSION) # timeserial shift in a user timeserials
                for twitter_i in range( 0, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT):
                    twitter_shift = twitter_i * (TWITTER_LENGTH)* (WORD_EMBEDDING_DIMENSION) #twitter shift in a timeserial
                    for word_i in range( 0, TWITTER_LENGTH):
                        word_shift = word_i* (WORD_EMBEDDING_DIMENSION) 
                        for embeddim_i in range( 0, WORD_EMBEDDING_DIMENSION):
                            embeddim_shift = embeddim_i
                            batch_x[ user_i%batch_size][ timeserial_i][ twitter_i][ word_i][ embeddim_i] = self.__dataframe_of_pandas.iloc[ user_i][ label_shift+timeserial_shift+twitter_shift+word_shift+embeddim_shift]

            batch_y[ user_i%batch_size] = self.__dataframe_of_pandas.iloc[ user_i, 0]
            
            # list_labels = self.__dataframe_of_pandas.iloc[ user_i , 0 ]
            # batch_y[ user_i%batch_size ][ 0 ] = list_labels[ 0 ]
            #batch_y[ user_i%batch_size ][ 1 ] = list_labels[ 1 ]

        self.__currparam_batch_sizeent_cursor_in_dataframe = t

        return batch_x,batch_y
        #remenber to return! or will error: noneType not iterable

    def tail_batch(self):
        '''
        the tail_batch, complemented by the head if not long enough
        '''
        batch_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        batch_x = numpy.zeros( ( batch_size, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )
        for user_i in range(s,t):
            if user_i >= self.dataframe_size(): # when exceeded, complemented by head
                user_i = user_i - self.dataframe_size()
            #list_all_tweets_of_auser = self.__dataframe_of_pandas.iloc[ i , 2:(2+1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT) ].values.tolist()
            #list_a            label_shift = 1 #one col for label
            label_shift = 1 #one col for label
            for timeserial_i in range( 0, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT):
                timeserial_shift = timeserial_i* ( 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT)* (TWITTER_LENGTH)* (WORD_EMBEDDING_DIMENSION) # timeserial shift in a user timeserials
                for twitter_i in range( 0, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT):
                    twitter_shift = twitter_i * (TWITTER_LENGTH)* (WORD_EMBEDDING_DIMENSION) #twitter shift in a timeserial
                    for word_i in range( 0, TWITTER_LENGTH):
                        word_shift = word_i* (WORD_EMBEDDING_DIMENSION) 
                        for embeddim_i in range( 0, WORD_EMBEDDING_DIMENSION):
                            embeddim_shift = embeddim_i
                            batch_x[ user_i%batch_size][ timeserial_i][ twitter_i][ word_i][ embeddim_i] = self.__dataframe_of_pandas.iloc[ user_i][ label_shift+timeserial_shift+twitter_shift+word_shift+embeddim_shift]

            batch_y[ user_i%batch_size ] =  self.__dataframe_of_pandas.iloc[ user_i, 0 ]       

        self.__current_cursor_in_dataframe = 0

        return batch_x,batch_y

if __name__ == '__main__':
    #print(os.getcwd())  
    oDataManager = DataManager( param_batch_size = 64 )
    #oDataManager.generate_csv_from_wordembbed( TRAIN_SET_PATH )
    oDataManager.load_dataframe_from_file( TRAIN_SET_PATH)
    print(oDataManager.dataframe_size()//64)
    print(oDataManager.dataeframe_size()%64)
    
    for i in range(0,oDataManager.dataframe_size()//64):
        (batch_x,batch_y) = oDataManager.next_batch()
        print(i)
    (batch_x,batch_y) = oDataManager.tail_batch()
    print(batch_x.shape) #size of torch in numpy
    print(batch_y.shape)
