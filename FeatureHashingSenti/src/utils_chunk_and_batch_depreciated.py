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
        
        self.__chunks_of_pandas = None #chunks created by pandas, load from file once
        self.__chunk_multiplier_for_chunk_size = CHUNK_SIZE_MULTIPLIER_BATCH_SIZE
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
        chunk_size = self.__chunk_multiplier_for_chunk_size * self.__batch_size #chunk size is the multiply of batch size

        self.__chunks_of_pandas = pandas.read_csv( param_filepath_in, dtype = numpy.float32, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c', chunksize= chunk_size)
        
        #self.__dataframe_of_pandas = pandas.read_csv( param_filepath_in, header = None, encoding = 'utf8',  sep = '\t' , engine = 'c') # you can use regular expression in sep by setting engine = 'python'
        #engine = 'c' will face error, may be 'c' needs 0 0 0 to be 0.0 0.0 0.0

        #c do not support \t\n
        #print(len(self.__dataframe_of_pandas) )
        #currently every 9 lines describe a user
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

        #dataframe_of_pandas = pandas.DataFrame( data= self.__chunks_of_pandas.get_chunk( chunk_size), dtype= numpy.float32) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        #get the chunk_index and batch_index, if multiplied, get a chunk
        chunk_size = self.__chunk_multiplier_for_chunk_size * self.__batch_size
        chunk_index = s // chunk_size # infact it's a dataframe size
        batch_index = s // batch_size

        if chunk_index * self.__chunk_multiplier_for_chunk_size == batch_index:
            print("Loading chunk: %d"%(chunk_index))
            self.__current_dataframe_of_pandas = self.__chunks_of_pandas.get_chunk( chunk_size ) # if chunk_size == None then using the param specified in read_csv
            # you can also use self.__chunks_of_pandas.get_chunk()
            # or you can use next( self.__chunks_of_pandas)
            self.__current_dataframe_of_pandas.sample( frac=1 ) # reshuffle the dataframe
        print("Loading batch: %d"%(batch_index))
        dataframe_of_pandas = self.__current_dataframe_of_pandas#.get_chunk( chunk_size) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        #print( 'get_chunk: '+ str( s//batch_size) )
        batch_x = numpy.zeros( ( batch_size, USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )
        for user_i in range(s,t):
            #print( 'user_i: '+ str(user_i) )
            #each user's time serial
            label_shift = 1 #one col for label
            #batch_x[ user_i%batch_size] = numpy.reshape( dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ], (1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )
            #batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ].values.reshape( ( 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )

            # dataframe_of_pandas infact is a dataframe of a chunk_size
            batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% chunk_size][ label_shift: ].values.reshape( ( USER_SELF_TWEETS, NEIGHBOR_TWEETS+1, TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
            batch_y[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% chunk_size][ 0] #iloc is absolute shift, loc is access by row name. if header==None, then the row name is the int index, which is inherented by splitted chunks
            #print(batch_x.shape)

            # list_labels = self.__dataframe_of_pandas.iloc[ user_i , 0 ]
            # batch_y[ user_i%batch_size ][ 0 ] = list_labels[ 0 ]
            #batch_y[ user_i%batch_size ][ 1 ] = list_labels[ 1 ]

        self.__current_cursor_in_dataframe = t

        return batch_x,batch_y

    def next_batch_slow(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        '''

        batch_size = self.__batch_size
        chunk_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        #dataframe_of_pandas = pandas.DataFrame( data= self.__chunks_of_pandas.get_chunk( chunk_size), dtype= numpy.float32) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same
        dataframe_of_pandas = self.__chunks_of_pandas.get_chunk( chunk_size) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        print( 'get_chunk: '+ str( s//batch_size) )
        batch_x = numpy.zeros( ( batch_size, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )
        for user_i in range(s,t):
            print( 'user_i: '+ str(user_i) )
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
                            #chunk_size must == batch_size
                            #print( 'embeddim_shift: '+ str(embeddim_shift) )
                            batch_x[ user_i%batch_size][ timeserial_i][ twitter_i][ word_i][ embeddim_i] = dataframe_of_pandas.iloc[ user_i%batch_size][ label_shift+timeserial_shift+twitter_shift+word_shift+embeddim_shift]
                            #batch_x[ user_i%batch_size][ timeserial_i][ twitter_i][ word_i][ embeddim_i] = dataframe_of_pandas.iloc[ user_i%batch_size][ label_shift+timeserial_shift+twitter_shift+word_shift+embeddim_shift]
            batch_y[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i%batch_size][ 0] #iloc is absolute shift, loc is access by row name. if header==None, then the row name is the int index
            
            # list_labels = self.__dataframe_of_pandas.iloc[ user_i , 0 ]
            # batch_y[ user_i%batch_size ][ 0 ] = list_labels[ 0 ]
            #batch_y[ user_i%batch_size ][ 1 ] = list_labels[ 1 ]

        self.__current_cursor_in_dataframe = t

        return batch_x,batch_y
        #remenber to return! or will error: noneType not iterable

    def tail_batch( self):
        batch_size= self.__batch_size

        s= self.__current_cursor_in_dataframe
        t= s + batch_size

        #get the chunk_index and batch_index, if multiplied, get a chunk
        chunk_size = self.__chunk_multiplier_for_chunk_size * self.__batch_size
        chunk_index = s // chunk_size # infact it's a dataframe size
        batch_index = s // batch_size

        if chunk_index * self.__chunk_multiplier_for_chunk_size == batch_index:
            print("Loading chunk: %d"%(chunk_index))
            self.__current_dataframe_of_pandas = self.__chunks_of_pandas.get_chunk( chunk_size ) # if chunk_size == None then using the param specified in read_csv
            # you can also use self.__chunks_of_pandas.get_chunk()
            # or you can use next( self.__chunks_of_pandas)
            self.__current_dataframe_of_pandas.sample( frac=1 ) # reshuffle the dataframe
        print("Loading batch: %d"%(batch_index))
        dataframe_of_pandas_last_chunk = self.__current_dataframe_of_pandas#.get_chunk( chunk_size) #get the current chunk, chunk is a default dataframe, should be transformed to float32, or the dataformat will not be the same

        batch_x = numpy.zeros( ( batch_size, USER_SELF_TWEETS, 1 + NEIGHBOR_TWEETS , TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )

        #complement the last chunk with the initial last chunk
        last_chunk_size = len( dataframe_of_pandas_last_chunk)
        
        append_times = chunk_size// last_chunk_size
        append_tail = chunk_size% last_chunk_size

        dataframe_of_pandas = pandas.DataFrame(data = None, columns = dataframe_of_pandas_last_chunk.axes[1] )
        for i in range( 0, append_times) :
            dataframe_of_pandas= dataframe_of_pandas.append( dataframe_of_pandas_last_chunk, ignore_index=True) # for ignore_index refer to the manual

        dataframe_of_pandas= dataframe_of_pandas.append( dataframe_of_pandas_last_chunk.iloc[ 0: append_tail], ignore_index=True)

        for user_i in range(s,t):
            #list_all_tweets_of_auser = self.__dataframe_of_pandas.iloc[ i , 2:(2+1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT) ].values.tolist()
            #list_a            label_shift = 1 #one col for label
            label_shift = 1 #one col for label

            #batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift: ].reshape( (1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION) )
            batch_x[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% chunk_size][ label_shift: ].values.reshape( ( USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH*WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION ) )
            batch_y[ user_i%batch_size] = dataframe_of_pandas.iloc[ user_i% chunk_size][0]
        self.__current_cursor_in_dataframe = 0

        return batch_x,batch_y

    def tail_batch_slow(self):
        '''
        the tail_batch, complemented by the head if not long enough
        '''
        batch_size = self.__batch_size
        chunk_size = self.__batch_size

        s = self.__current_cursor_in_dataframe #start cursor
        t = s + batch_size #end cursor

        dataframe_of_pandas_last_chunk = self.__chunks_of_pandas.get_chunk( chunk_size) #get the current chunk
        batch_x = numpy.zeros( ( batch_size, 1+USER_TWITTER_COUNT+NEIGHBOR_TWITTER_COUNT, 1 + USER_TWITTER_COUNT + NEIGHBOR_TWITTER_COUNT , TWITTER_LENGTH, WORD_EMBEDDING_DIMENSION ) )
        batch_y = numpy.zeros( ( batch_size) )
        
        #complement the last chunk with the initial last chunk
        last_chunk_size = len( dataframe_of_pandas_last_chunk)
        append_times = chunk_size// last_chunk_size
        append_tail = chunk_size% last_chunk_size
        dataframe_of_pandas = pandas.DataFrame(data = None, columns = dataframe_of_pandas_last_chunk.axes[1] )
        for i in range( 0, append_times) :
            dataframe_of_pandas = dataframe_of_pandas.append( dataframe_of_pandas_last_chunk, ignore_index=True)

        dataframe_of_pandas = dataframe_of_pandas.append( dataframe_of_pandas_last_chunk.iloc[ 0: append_tail], ignore_index=True)
        
        for user_i in range(s,t):
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
                            batch_x[ user_i%batch_size][ timeserial_i][ twitter_i][ word_i][ embeddim_i] = dataframe_of_pandas.iloc[ user_i% batch_size][ label_shift+timeserial_shift+twitter_shift+word_shift+embeddim_shift]

            batch_y[ user_i%batch_size ] = dataframe_of_pandas.iloc[ user_i%batch_size][0]       

        self.__current_cursor_in_dataframe = 0

        return batch_x,batch_y

    def n_batches(self ):
        if self.__chunks_of_pandas == None:
            print( 'Error: __chunks_of_pandas == None' , file = sys.stderr)
        else:
            return ceil( self.__training_instances_size / self.__batch_size )

if __name__ == '__main__':
    #print(os.getcwd())  
    oDataManager = DataManager( param_batch_size = 4 , param_training_instances_size = TRAINING_INSTANCES )
    #oDataManager.generate_csv_from_wordembbed( TRAIN_SET_PATH )
    oDataManager.load_dataframe_from_file( TRAIN_SET_PATH)

    #print(oDataManager.dataframe_size()//64)
    #print(oDataManager.dataeframe_size()%64)
    print('Hello')
    for i in range(0, TRAINING_INSTANCES// 4) :
        print(i)
        (batch_x,batch_y) = oDataManager.next_batch()
        print('shape_x:' , batch_x.shape , '--shape_y:' , batch_y.shape ) 
    ( batch_x, batch_y ) = oDataManager.tail_batch()
    print( batch_x.shape ) #size of torch in numpy
    print( batch_y.shape )