import torch #torch is a file named torch.py
from torch.autograd import Variable # torch here is a folder named torch
from torchnet import meter
from models import biLSTMAttention #this is filename, once imported, you can use the classes in it
# equal to from models.topicalAttentionGRU import TopicalAttentionGRU

from settings import *
from utils import *

import os

# for realtime tokenization and charEncoding
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from allennlp.modules.elmo import batch_to_ids

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2" #cuda device id

# loss_func=nn.CrossEntropyLoss()

def train(**kwargs):
    '''
    begin training the model
    *kwargs: train(1,2,3,4,5)=>kwargs[0] = 1 kwargs[1] = 2 ..., kwargs is principally a tuple
    **kwargs: train(a=1,b=2,c=3,d=4)CustomPreProcessor=>kwargs[a] = 1, kwargs[b] = 2, kwargs[c] = 3, kwargs[d] = 4, kwargs is principally a dict
    function containing kwargs *kwargs **kwargs must be written as: def train(args,*args,**args)
    '''

    saveid = latest_save_num() + 1
    save_path = '%s/%d' % ( SAVE_DIR , saveid ) #the save_path is 
    print("logger save path: %s"%(save_path) )
    if not os.path.exists( save_path ):
        os.makedirs( save_path )
    log_path_each_save = '%s/log.txt' % save_path
    model_path_each_save = '%s/model' % save_path
    logger = get_logger(log_path_each_save)


    config = DefaultConfig()
    config.set_attrs(kwargs) # settings here, avalid_data_utillso about whether on cuda
    # print(config.get_attrs())
    epochs = config.epochs
    batch_size = config.batch_size

    if config.on_cuda: # determine whether to run on cuda        
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            logger.info('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.model == 'BiLSTMAttention':
        model = biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file,
            param_on_cuda = False)

    if config.on_cuda:
        logger.info('Model run on GPU')
        model = model.cuda()
        logger.info('Model initialized on GPU')
    else:
        logger.info('Model run on CPU')
        model = model.cpu()
        logger.info('Model initialized on CPU')


    #print('logger-setted',file=sys.stderr)
    logger.info( model.modelname ) #output the string informetion to the log
    logger.info( str( config.get_attrs() ) ) #output the string information to the log

    #read in the trainset and the trial set
    train_data_manager = DataManager( batch_size , TRAINING_INSTANCES ) #Train Set
    train_data_manager.load_dataframe_from_file( TRAIN_SET_PATH )
    
    #set the optimizer parameter, such as learning rate and weight_decay, function Adam, a method for Stochastic Optizimism
    lr = config.learning_rate#load the learning rate in config, that is settings.py
    # params_iterator_requires_grad can only be iterated once
    params_iterator_requires_grad = filter( lambda trainingParams: trainingParams.requires_grad, model.parameters() )
    # print( len(list(params_iterator_requires_grad) ) ) # 25 parameters
    optimizer = torch.optim.Adam(
                                params_iterator_requires_grad, 
                                lr=lr,
                                weight_decay=config.weight_decay #weight decay that is L2 penalty that is L2 regularization, usually added after a cost function(loss function), for example C=C_0+penalty, QuanZhongShuaiJian, to avoid overfitting
                                )

    # By default, the losses are averaged over observations for each minibatch. 
    # However, if the field size_average is set to False, the losses are instead 
    # summed for each minibatch

    criterion = torch.nn.CrossEntropyLoss( size_average = False )#The CrossEntropyLoss, My selector in my notebook = loss + selecting strategy(often is selecting the least loss)
    #once you have the loss function, you also have to train the parameters in g(x), which will be used for prediction
    loss_meter = meter.AverageValueMeter() #the loss calculated after the smooth method, that is L2 penalty mentioned in torch.optim.Adam
    confusion_matrix = meter.ConfusionMeter( CLASS_COUNT ) #get confusionMatrix, the confusion matrix is the one show as follows:
    '''                    class1 predicted class2 predicted class3 predicted
    class1 ground truth  [[4,               1,               1]
    class2 ground truth   [2,               3,               1]
    class2 ground truth   [1,               2,               9]]
    '''
    model.train()
    pre_loss = 1e100
    best_acc = 0
    
    for epoch in range( epochs ):
        '''
        an epoch, that is, train data of all barches(all the data) for one time
        '''

        loss_meter.reset()
        confusion_matrix.reset()

        train_data_manager.reshuffle_dataframe()

        n_batch = train_data_manager.n_batches() # it was ceiled, so it is "instances/batch_size + 1"

        batch_index = 0
        for batch_index in range( 0 , n_batch - 1):
            ( x , y ) = train_data_manager.next_batch() # this operation is time consuming

            x = Variable( torch.from_numpy( x ).long() ) # long seems to trigger cuda error, it cannot handle long # variable by defalut requires_grad
            #print( x.size() )
            y = Variable( torch.LongTensor( y ) , requires_grad = False )
            y = y - 1
            #print(y.size())

            ##########################logger.info('Begin fetching a batch')
            loss , scores , corrects = eval_batch( model , x , y , criterion , config.on_cuda )
            ##########################logger.info('End fetching a batch, begin optimizer')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ##########################logger.info('End optimizer')
            loss_meter.add( loss.data.item() ) # data is the tensor, [0] is a Python number, if a 0-dim tensor, then .item will get the python number, if 1-dim then .items will get list

            confusion_matrix.add( scores.data , y.data ) 
            if ( batch_index + 1 ) % 50 == 0:# if batch_index == 10 then display the accuracy of the batch
                accuracy = corrects.float() / config.batch_size # for 2 LongTensors,  17 / 18 = 0 
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f' % ( epoch , epochs , batch_index , n_batch , loss_meter.value()[0] , accuracy ) ) # .value()[0] is the loss value

        if TRAINING_INSTANCES % batch_size == 0:
            train_data_manager.set_current_cursor_in_dataframe_zero()
        else:
            batch_index += 1 # the value can be inherented
            ( x , y ) = train_data_manager.tail_batch()
            x = Variable( torch.from_numpy( x ).long() ) # long seems to trigger 
            y = Variable( torch.LongTensor( y ) , requires_grad = False )
            y = y - 1
            loss , scores , corrects = eval_batch( model , x , y , criterion , config.on_cuda )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.add( loss.data.item() ) 
            confusion_matrix.add( scores.data , y.data ) 
            if ( batch_index + 1 ) % 50 == 0:# if batch_index == 10 then display the accuracy of the batch
                accuracy = corrects.float() / config.batch_size # for 2 LongTensors,  17 / 18 = 0 
                #print("accuracy = %f, corrects = %d"%(accuracy, corrects))
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f' % ( epoch , epochs , batch_index , n_batch , loss_meter.value()[0] , accuracy ) ) # .value()[0] is the loss value  y = Variable( torch.LongTensor( y ) , requires_grad = False )

        # after an epoch it should be evaluated
        model.eval() # switch to evaluate model
        #if ( batch_epochsindex + 1 ) % 25 == 0:# every 50 batches peek its accuracy and get the best accuracy
        confusion_matrix_value=confusion_matrix.value()
        acc = 0
        for i in range(CLASS_COUNT):
            acc += confusion_matrix_value[i][i] #correct prediction count
        acc = acc / confusion_matrix_value.sum() #the accuracy, overall accuracy in an epoch
        the_overall_averaged_loss_in_epoch = loss_meter.value()[0] # a 1-dim tensor with lenth 1, so you have to access the element by [0]
        logger.info( 'epoch: %d/%d\taverage_loss: %f\taccuracy: %f' % ( epoch, epochs, the_overall_averaged_loss_in_epoch, acc ) )
        model.train() # switch to train model

        #if accuracy increased, then save the model and change the learning rate
        if acc > best_acc:
            #save the model
            model.save(model_path_each_save)
            logger.info('model saved to %s'%model_path_each_save)

            #change the learning rate
            lr = lr * config.lr_decay
            logger.info( 'learning_rate changed to %f'% lr )
            for param_group in optimizer.param_groups:
                param_group['lr']=lr

            best_acc = acc

        pre_loss=loss_meter.value()[0]


def eval_batch(model,x,y,criterion,on_cuda):
    '''
    evaluate the logits of each instance, loss, corrects in a batch
    '''
    if on_cuda:
        x , y = x.cuda() , y.cuda()
    else:
        x , y = x.cpu() , y.cpu()

    logits = model(x) # batch_size * dim

    # since the size_average parameter==False, the loss is the sumed loss of the batch. The loss is a value rather than a vector
    loss = criterion( logits , y ) # CrossEntropyLoss takes in a vector and a class num ( usually a index num of the vector )
    
    model_training_predicts = torch.max( logits , 1)[ 1 ] # [0] : max value of dim 1 [1]: max index of dim 1 LongTensor
    
    assert model_training_predicts.size() == y.size()

    # y.data shouldn't contain -1 or 5, or will trigger cuda error in loss = criterion( logits , y ) but will display later
    corrects = ( model_training_predicts.data == y.data ).sum( ) # corrects is a LongTensor sotred in cuda, y.data means the tensor of the variable
    #print('-----------------%d'%(corrects))
    
    return loss , logits , corrects

def predict( model_dir, mtype='BiLSTMAttention'):
    '''
    load the model and conduct the prediction, the prediction is added with 1 since the
    original prediction is the index
    prediction is saved in '%s/0'%SAVE_DIR
    '''

    model_path = '%s/model'%model_dir
    output_path = '%s/res.txt'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )

    if mtype=='BiLSTMAttention':
        model=biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file,
            param_on_cuda = True)
    print('Loading trained model')
    model.load( model_path )

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.on_cuda:
        model=model.cuda()
    else:
        model=model.cpu()
    
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = datamanager.n_batches()
    res= numpy.array([])

    batch_index = 0

    for batch_index in range(n_batch - 1):
        ( x , y ) = datamanager.next_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        scores = model.forward(x)
        _ , predict = torch.max(scores, 1) # predict is the first dimension , its the same as [ 1 ] 

        res = numpy.append( res, predict.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        ( x , y ) = datamanager.tail_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        scores = model.forward(x)
        _ , predict = torch.max(scores, 1) # predict is the first dimension , its the same as [ 1 ] 
        res = numpy.append( res, predict.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  


    res = res[ :TESTING_INSTANCES ]
    res = res + 1

    numpy.savetxt( output_path, res, fmt='%d')

def save_elmo_rep_testset(model_dir, output_path, mtype='BiLSTMAttention'):
    '''
    Given Tokenized CharEncoded test file in TEST_SET_PATH, 
    save the embedded representation in output_path
    each line is label [sentence len] * [word embedding dim]
    '''
    model_path = '%s/model'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )

    if mtype=='BiLSTMAttention':
        model=biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file,
            param_on_cuda = True)
    print('Loading trained model')
    model.load( model_path )

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.on_cuda:
        model=model.cuda()
    else:
        model=model.cpu()
    
    # print(model)
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = datamanager.n_batches()
    res= numpy.empty( (0, DOCUMENT_SEQ_LEN * 1024), dtype = numpy.float32 ) # res is [], shape = (0, 3) , be sure to append on axis 0

    batch_index = 0

    for batch_index in range(n_batch - 1):
        ( x , y ) = datamanager.next_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        elmo_rep = elmo_dict['elmo_representations']
        var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        ( x , y ) = datamanager.tail_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        elmo_rep = elmo_dict['elmo_representations']
        var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  

    res = res[ :TESTING_INSTANCES ]

    numpy.savetxt( output_path, res, fmt='%f')    

def calculate_accuracy(model_dir):
    '''
    # actually, sklearn, meters.confusionMatrix and calculate_accuracy_and_recall.py can calculate the confusionMatrix
    '''
    fpInTestSet = open(TEST_SET_PATH, 'rt')
    fpInPredicted = open('%s/res.txt'%model_dir,'rt')
    rightPredictionCount = 0
    
    for alineG in fpInTestSet:
        (ground_truth , others) = alineG.strip().split( ' ', 1)
        alineP = fpInPredicted.readline()
        predicted = alineP.strip()
        if predicted == ground_truth:
            rightPredictionCount += 1
    fpInTestSet.close()
    fpInPredicted.close()
    
    acc = rightPredictionCount / TESTING_INSTANCES
    print(acc)

def save_elmo_rep(model_dir, input_path, output_path, mtype='BiLSTMAttention'):
    '''
    Given Tokenized CharEncoded txt file in input_path, 
    save the word embedded file in output_path
    each line is [sentence len] * [word embedding dim]
    '''
    model_path = '%s/model'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )

    if mtype=='BiLSTMAttention':
        model=biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file,
            param_on_cuda = True)
    print('Loading trained model')
    model.load( model_path )

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    if config.on_cuda:
        model=model.cuda()
    else:
        model=model.cpu()
    
    # print(model)
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( input_path )
    n_batch = datamanager.n_batches()
    res= numpy.empty( (0, DOCUMENT_SEQ_LEN * 1024), dtype = numpy.float32 ) # res is [], shape = (0, 3) , be sure to append on axis 0

    batch_index = 0

    for batch_index in range(n_batch - 1):
        x  = datamanager.next_batch_nolabel()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        x = datamanager.tail_batch_nolabel()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  

    res = res[ :TESTING_INSTANCES ]

    numpy.savetxt( output_path, res, fmt='%f')

def compute_elmo_rep(model_dir, input_list, mtype = 'BiLSTMAttention'):
    '''
    Given a list of documents, 
    return a list of embedded documents
    each element in list is [sentence len] * [word embedding dim]
    '''
    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )
    model_path = '%s/model'%model_dir

    text_processor = TextPreProcessor(
    normailze = ['url','email','percent','money','phone','user',
        'time','date','number'],
    annotate = {"hashtag","allcaps","elongated","repeated",
        "emphasis","censored"},
    fix_html = True,
    segmenter = "english",
    corrector = "english",
    unpack_hashtags = True,
    unpack_contractions = True,
    spell_correct_elong = False,

    tokenizer = SocialTokenizer(lowercase = True).tokenize,
    dicts = [emoticons])

    listTokenized = list(text_processor.pre_process_docs( input_list ) )
    print('After tokenization:')
    print(listTokenized)

    tensorTokenizedCharEncoded = batch_to_ids( listTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
    # print( listShuffledReviewsCharacterEmbedded[0].size() )

    arrayTokenizedCharEncoded = tensorTokenizedCharEncoded.numpy().astype(numpy.int32)

    x = Variable( torch.from_numpy(arrayTokenizedCharEncoded).long(), requires_grad=False)

    if config.on_cuda:
        x = x.cuda()
    else:
        x = x.cpu()

    #print(x.size())

    model=biLSTMAttention.BiLSTMAttention(
        param_document_seq_len = tensorTokenizedCharEncoded.size(1),# 300 in our model
        param_character_embedding_len = tensorTokenizedCharEncoded.size(2), #it depends on the setting
        param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
        param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
        param_class_count = 5,
        param_options_file = config.options_file,
        param_weight_file = config.weight_file,
        param_on_cuda = True)
    print('Loading trained model')
    model.load( model_path )
    if config.on_cuda:
        model=model.cuda()
    else:
        model=model.cpu()

    elmo_dict = model.forward_obtainTrainedElmoRep(x)
    
    elmo_rep = elmo_dict['elmo_representations'][0]# since num_output_representations = 1, so len(list_elmo_rep) = 1, 
    # if num_output_representations == 2, then will produce 2 same elmo_representations of [batch_size, seq_len, wordembedding_len]

    #print(elmo_rep.size())
    arr_elmo_rep = elmo_rep.data.cpu().numpy()

    return arr_elmo_rep


if __name__=='__main__':
    
    ##==========Train with the training set
    train(  )

    ##==========Predict with the testing set
    #predict('%s/0'%SAVE_DIR)

    ##==========Calculate with the predicted result
    #calculate_accuracy('%s/0'%SAVE_DIR)

    ##==========Load the trained model, compute and save the representation
    ##          for testing set
    #save_elmo_rep_testset( model_dir = '%s/0'%SAVE_DIR, output_path = '%s/0/elmp_rep'%SAVE_DIR )
    
    ##==========Load the trained model, compute and save the representation
    ##          for TokenizedCharEmbedded documents
    # save_elmo_rep( model_dir = '%s/0'%SAVE_DIR, 
    #     input_path = '%s/datasets/clinical_reviews_tokenized_charencoded.txt'%ROOT_DIR, 
    #     output_path = '%s/0/clinical_reviews_embedded.txt'%SAVE_DIR)

    ##==========Load the trained model, compute the representations 
    ##          for a list of text documents

    # list_docs = [ 'Hello RiverBank!!?? This is document 1.',
    #                 'Hi Bank, This is document 2.']
    # array_embedded_docs = compute_elmo_rep(model_dir = '%s/0'%SAVE_DIR,
    #     input_list = list_docs)
    # print( array_embedded_docs.shape )
    # print( array_embedded_docs )


    # elmo_dict = model.forward_obtainTrainedElmoRep(x)