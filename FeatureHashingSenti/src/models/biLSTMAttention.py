# -*- coding: utf8 -*-

import torch

import torch.nn #torch.nn and torch two different module from torch
import torch.nn.functional

import torch.autograd

from allennlp.modules.elmo import Elmo
from models.attention import Attention
# when debugging, should change to another absolute path import from attention import Attention
#from attention import Attention #Debugging

import os


class BiLSTMAttention(torch.nn.Module):
        '''
        You have to inherent nn.Module to define your own model
        two virtual functions, loda and save must be instantialized
        '''
        def __init__(self,
            param_document_seq_len,# 300 in our model
            param_character_embedding_len, #it depends on the setting
            param_bilstm_hidden_size, # the size of tweet level rnn, since it's biLSTM should be divided twice
            param_attention_size, # attention size should be a smoothed representation of character-emb
            param_class_count,
            param_options_file, # elmo file for options
            param_weight_file, # elmo file for weight
            param_on_cuda): # class of user labels, in fact it is the class of level 1 labels

            super(BiLSTMAttention,self).__init__()
            self.modelname='BiLSTMAttention' #same with the class name
        
            self.document_seq_len = param_document_seq_len
            self.character_embedding_len = param_character_embedding_len
            self.bilstm_hidden_size = param_bilstm_hidden_size
            self.attention_size = param_attention_size
            self.class_count = param_class_count
            # if param_on_cuda:
            #     self.ids_only_wordembed_dim = torch.autograd.Variable( torch.LongTensor( [ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ] ) ).cuda()
            #     self.ids_only_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embed_size ) ] ) ).cuda()
            # else:
            #     self.ids_only_wordembed_dim = torch.autograd.Variable( torch.LongTensor( [ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ] ) ).cpu()
            #     self.ids_only_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embed_size ) ] ) ).cpu()

            self.elmo_layer = Elmo( param_options_file, param_weight_file, num_output_representations = 1, dropout = 0 )
            self.elmo_hiddensize = 1024 # this is fixed, after elmo_layer, the CharEmbLen should be transferred to ElmoHiddensize

            self.bilstm_document_layer_count = 2 # 2 BiLSTM layers
            self.bilstm_document = torch.nn.LSTM( self.elmo_hiddensize, self.bilstm_hidden_size, self.bilstm_document_layer_count, dropout = 0.0, bidirectional = True ) #, default batch_first = False, the batch_size = second
            
            # if param_on_cuda:
            #     self.ids_seq_last = torch.autograd.Variable( torch.LongTensor( [ self.user_self_tweets -1 ] ) ).cuda()
            # else:
            #     self.ids_seq_last = torch.autograd.Variable( torch.LongTensor( [ self.user_self_tweets -1 ] ) ).cpu()

            # self.linear_tweet = torch.nn.Linear( self.rnn_tweet_hidden_size, self.word_embed_size)
            # # mean pooling is functioned as mean after the LSTM

            # self.A_alpha = torch.nn.Parameter( torch.Tensor( ( self.fix_sentence_length ) ) )
            # # self.B_alpha = torch.nn.Parameter( torch.Tensor( ( self.word_embed_size ) ) ) # you can choose this to be a vector, but this may cause Curse of Dimensionality
            # self.B_alpha = torch.nn.Parameter( torch.Tensor( ( 1 ) ) ) # can use a single value or matrix
            # self.A_tweets = torch.nn.Parameter( torch.Tensor( ( self.neighbor_tweets + 1 ) ) ) # attention for twitter embed
            # self.B_tweets = torch.nn.Parameter( torch.Tensor( ( 1 ) ) ) # attention for twitter embed

            # self.A_alpha.data.normal_(std=0.1)
            # self.B_alpha.data.normal_(std=0.1)
            # self.A_tweets.data.normal_(std=0.1)
            # self.B_tweets.data.normal_(std=0.1)

            #the attention size can be batch-dependent, that is the case in emoj_LSTM, but there we think for all the input the attention is the same

            # self.rnn_hidden_size = param_rnn_hidden_size # hidden size: the vector size of output for each time, It's a vector!
            # self.rnn_layer_count = 1 #constant, I haven't implemented the dynamic programming yet, but it defines the RNN LayerNum
            # self.rnn = torch.nn.GRU( self.word_embed_size + self.topic_embed_size , self.rnn_hidden_size ,self.rnn_layer_count , dropout = 0.0 , batch_first = True , bidirectional = False ) # batchfirst: the datasize is [ batch_size, seq , feature ]
            
            self.attention_over_seq = Attention( self.attention_size, self.bilstm_hidden_size * 2 ) # to handle biLSTM output

            self.linear = torch.nn.Linear( self.elmo_hiddensize , self.class_count )


        def load(self , path):
            self.load_state_dict( torch.load(path) )

        def save(self, path):
            save_result = torch.save( self.state_dict() , path )
            return save_result

        def forward( self , param_input ):
            '''
            from input to output
            '''
            ( batch_size , doc_seq_len , char_emb_len ) = param_input.size()
            assert self.document_seq_len == doc_seq_len
            assert self.character_embedding_len == char_emb_len

            list_elmo_rep = self.elmo_layer( param_input )['elmo_representations'][0]
            # since num_output_representations = 1, so len(list_elmo_rep) = 1, 
            # if num_output_representations == 2, then will produce 2 same elmo_representations of [batch_size, seq_len, wordembedding_len]
            var_elmo_rep = torch.cat( list_elmo_rep, dim = 0 ) # concatenate seq of tensors
            #print( var_elmo_rep.size() )
            var_elmo_rep = var_elmo_rep.permute( 1, 0, 2 ) # not batch_first
            
            var_bilstm_document_output, (var_bilstm_document_output_h, var_bilstm_document_output_c) = self.bilstm_document( var_elmo_rep )
            var_bilstm_document_output = var_bilstm_document_output.permute( 1, 0, 2 ) # batch_first again
            # #output is (batch , seq , hiddesize * 2 ) # it's concatenated automatically
            # #var_bilstm_document_output = torch.cat( ( var_twitter_embedded , var_only_topic_embedding ) ,dim = 3 ) 
            # #now is batch * seq * hiddesize * 2

            # batch_size , seq , hiddesize * 2
            #print( var_bilstm_document_output.size() )
            var_attentioned_output = self.attention_over_seq( var_bilstm_document_output )

            # var_attentioned_output is batch , hiddensize * 2
            var_attentioned_output = self.linear( var_attentioned_output )

            #print( var_attentioned_output.size() )

            return var_attentioned_output

        def forward_obtainTrainedElmoRep(self , param_input):
            '''
            compute the ElmoRep after training
            '''
            ( batch_size , doc_seq_len , char_emb_len ) = param_input.size()
            assert self.document_seq_len == doc_seq_len
            assert self.character_embedding_len == char_emb_len

            dict_elmo = self.elmo_layer( param_input )

            return dict_elmo
            


if __name__ == '__main__':
    # m = torch.randn(4,5,6)
    # print(m)userandneighbor_size
    # m_var = torch.autograd.Variable( m )
    # #ids = torch.Tensor([1,1,0,0]).long() #autograd = false by acquiescence
    # #var2 =  m.gather(1, ids.view(-1,1))
    # ids = torch.LongTensor( [ 2 , 4 ] )

    # ids_var = torch.autograd.Variable( ids )
    
    #they have the same function
    # var2 = m.index_select( 2 , ids  )
    # print( var2 )
    # var3 = torch.index_select( m , 2 , ids )
    # print( var3 )var_only_userprev_dim
    # var2_var = m_var.index_select( 2 , ids_var )
    # print(var2_var)

    # #model=model.cpu() #load the model to the CPU, 2 different ways to load the model
    # #model=model.cuda() #load the model to the GPU

    # var_test_expand = torch.autograd.Variable( torch.Tensor( [ [1 ,2 ,3 ,4 , 5, 6] , [7,8,9,10,11,12] , [ 13,14,15,16,17,18 ] ] ) )
    # var_test_expanded = var_test_expand.expand( 2, 3, 6 ) # expand the (3 , 6) into higher dimensions
    # print(var_test_expanded)

    # var_test_mult = torch.autograd.Variable( torch.Tensor( [ [ 1 ] , [ 2 ] ] ) )
    # var_test_fac = torch.autograd.Variable( torch.Tensor( [ 2 ] )  )
    # var_test_mult = var_test_mult.mul( var_test_fac )
    # print( var_test_mult )
    # var_test_mult = var_test_mult * 2 + 10
    # print( var_test_mult )

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2" #cuda device id

    var_input = torch.autograd.Variable( torch.ones( [16, 1219, 50], dtype = torch.long ) ) #torch.LongTensor( 128, 23, 50 ) )
    var_input = var_input * 261

    att_model_test = BiLSTMAttention(
            param_document_seq_len = 1219,# 300 in our model
            param_character_embedding_len = 50, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # the size of tweet level rnn, since it's biLSTM should be divided twice
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            param_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            param_on_cuda = True)

    #res=att_model_test( var_input )
    att_model_test = att_model_test.cuda()
    var_input = var_input.cuda()

    att_model_test( var_input )