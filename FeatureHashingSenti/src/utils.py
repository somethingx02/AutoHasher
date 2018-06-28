# -*- coding: utf8 -*- 

import logging
#from settings import CLASS_LEVEL1
from settings import * #import all the variables and classes in .py
import settings #you shall write setting.CLASS_LEVEL1 to use the variable in .py
import os,sys
import pandas
import numpy

from math import ceil

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

    def __init__(self , param_batch_size , param_training_instances_size ):
        #self.df=None
        #self.cursor=0
        #self.batch_size=batch_size 

        #global private variable initialized here
        
        self.__current_dataframe_of_pandas = None #dataframe read by a chunk
        self.__current_cursor_in_dataframe = 0 #the cursor shifted in pandas, it's the global cursor shift in dataframe
        self.__batch_size = param_batch_size
        self.__training_instances_size = param_training_instances_size

        #self.__batch_size = param_batch_size
        #self.__current_cursor = 0 #the current cursor in file, is the line num
        #self.__current_file_pointer = None list_all_tweets_of_auser#the current file pointer

    def load_dataframe_from_file( self, param_filepath_in):
        '''
        read once, initialize dataframe_of_pandas
        '''

        print('Loading dataframe...')
        self.__current_dataframe_of_pandas = pandas.read_csv( param_filepath_in, dtype = numpy.int32, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c')
        
        #self.__dataframe_of_pandas = pandas.read_csv( param_filepath_in, header = None, encoding = 'utf8',  sep = '\t' , engine = 'c') # you can use regular expression in sep by setting engine = 'python'
        #engine = 'c' will face error, may be 'c' needs 0 0 0 to be 0.0 0.0 0.0

        #c do not support \t\n
        #print(len(self.__dataframe_of_pandas) )
        #currently every 9 lines describe a user
    
    def reshuffle_dataframe(self):
        self.__current_dataframe_of_pandas.sample( frac=1 )

    def next_batch(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        use reshape
        '''

        batch_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        batch_index = s // batch_size

        #print( 'get_chunk: '+ str( s//batch_size) )
        batch_x = numpy.zeros( ( batch_size, DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
        batch_y = numpy.zeros( ( batch_size) )
        for user_i in range(s,t):

            label_shift = 1 #one col for label

            batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
            batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ 0] #iloc is absolute shift, loc is access by row name. if header==None, then the row name is the int index, which is inherented by splitted chunks

        self.__current_cursor_in_dataframe = t

        return batch_x,batch_y

    def set_current_cursor_in_dataframe_zero(self):
        '''
        if INSTANCE % batch_size == 0, then the tail_batch won't be called, so call this function to reset the __cursor_in_current_frame
        '''
        self.__current_cursor_in_dataframe = 0

    def tail_batch( self):

        batch_size= self.__batch_size

        s= self.__current_cursor_in_dataframe
        t= s + batch_size

        batch_index = s // batch_size

        batch_x = numpy.zeros( ( batch_size, DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN) )
        batch_y = numpy.zeros( ( batch_size) )

        #complement the last chunk with the initial last chunk
        assert len(self.__current_dataframe_of_pandas) == self.__training_instances_size
        last_batch_size = len( self.__current_dataframe_of_pandas) % batch_size
        
        append_times = batch_size // last_batch_size
        append_tail = batch_size % last_batch_size

        for user_i in range(s,t):
            label_shift = 1 #one col for label

            if (user_i % batch_size) < last_batch_size:    
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
                batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][0]
            else:
                shift_in_last_batch_size = (user_i % batch_size) % last_batch_size
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i - user_i % batch_size + shift_in_last_batch_size][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
                batch_y[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i - user_i % batch_size + shift_in_last_batch_size][0]
        
        self.__current_cursor_in_dataframe = 0

        return batch_x,batch_y

    def n_batches(self ):
        return ceil( self.__training_instances_size / self.__batch_size )

    def next_batch_nolabel(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        use reshape
        '''

        batch_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        batch_index = s // batch_size

        #print( 'get_chunk: '+ str( s//batch_size) )
        batch_x = numpy.zeros( ( batch_size, DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
        for user_i in range(s,t):

            label_shift = 0 #one col for label

            batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )

        self.__current_cursor_in_dataframe = t

        return batch_x

    def tail_batch_nolabel( self):

        batch_size= self.__batch_size

        s= self.__current_cursor_in_dataframe
        t= s + batch_size

        batch_index = s // batch_size

        batch_x = numpy.zeros( ( batch_size, DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN) )

        #complement the last chunk with the initial last chunk
        assert len(self.__current_dataframe_of_pandas) == self.__training_instances_size
        last_batch_size = len( self.__current_dataframe_of_pandas) % batch_size
        
        append_times = batch_size // last_batch_size
        append_tail = batch_size % last_batch_size

        for user_i in range(s,t):
            label_shift = 0 #one col for label

            if (user_i % batch_size) < last_batch_size:    
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
            else:
                shift_in_last_batch_size = (user_i % batch_size) % last_batch_size
                batch_x[ user_i%batch_size] = self.__current_dataframe_of_pandas.iloc[ user_i - user_i % batch_size + shift_in_last_batch_size][ label_shift: ].values.reshape( ( DOCUMENT_SEQ_LEN, CHARACTER_EMBEDDING_LEN ) )
        
        self.__current_cursor_in_dataframe = 0

        return batch_x


if __name__ == '__main__':
    #print(os.getcwd())  
    oDataManager = DataManager( param_batch_size = 128 , param_training_instances_size = TRAINING_INSTANCES )
    #oDataManager.generate_csv_from_wordembbed( TRAIN_SET_PATH )
    oDataManager.load_dataframe_from_file( TRAIN_SET_PATH)

    #print(oDataManager.dataframe_size()//64)
    #print(oDataManager.dataeframe_size()%64)
    print('Hello')
    for i in range(0, TRAINING_INSTANCES// 128) :
        print(i)
        (batch_x,batch_y) = oDataManager.next_batch()
        print('shape_x:' , batch_x.shape , '--shape_y:' , batch_y.shape ) 
    ( batch_x, batch_y ) = oDataManager.tail_batch()
    print( batch_x.shape ) #size of torch in numpy
    print( batch_y.shape )