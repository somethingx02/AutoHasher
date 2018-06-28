# -*- coding: utf8 -*-

import json

import random

import os


from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from allennlp.modules.elmo import batch_to_ids
import numpy as np


class YelpPreprocessor:
    '''
    Convert Yelp Preprocessor to a text file consisting of lines
    ======================================================
    parameters:
    ----------
    None

    return:
    ----------
    None
    '''
    def __init__(self):
        return None

    def jsonPreprocess(self, paramFpathInBussiness, paramFpathInReview, paramFpathOutReview, paramFpathOutStars):
        '''
        retrieve clinical documents from business, reviews
        ==================================================
        parameters:
        -----------
        paramFpathInBussiness: business file
        paramFpathInReview: review file
        paramFpathOutReview: texted review
        paramFpathOutStars: stars file

        return:
        -----------
        None
        '''

        # dictClinicalBussinessCategories = {
        # "Chiropractic and physical therapy":{"includeif":{"chiropractor","physical therapy"},"excludeif":{}},
        # "Dental":{"includeif":{"dentist"},"excludeif":{}},
        # "Dermatology":{"includeif":{"dermatologist"},"excludeif":{"optometrist", "veterinarians", "pets"}},
        # "Family practice":{"includeif":{"family practice"},"excludeif":{"psychiatrist", "chiropractor", "beauty", "physical therapy", "specialty", "dermatologists", "weight loss", "acupuncture", "cannabis clinics", "naturopathic", "optometrists"}},
        # "Hospitals and clinics":{"includeif":{"hospital"},"excludeif":{"physical therapy", "rehab", "retirement homes", "veterinarians", "dentist"}},
        # "Optometry":{"includeif":{"optometrist"},"excludeif":{"dermatologist"}},
        # "Mental health":{"includeif":{"psychiatrist","psychologist"},"excludeif":{}},
        # "Dental":{"includeif":{"speech therapy"},"excludeif":{"speech"}},
        # }
        dictClinicalBussinessCategories = {
        "Chiropractic and physical therapy":{"includeif":{"Chiropractor","Physical Therapy"},"excludeif":set()},
        "Dental":{"includeif":{"Dentist"},"excludeif":set()},
        "Dermatology":{"includeif":{"Dermatologist"},"excludeif":{"Optometrist", "Veterinarians", "Pets"}},
        "Family practice":{"includeif":{"Family Practice"},"excludeif":{"Psychiatrist", "Chiropractor", "Beauty", "Physical Therapy", "Specialty", "Dermatologists", "Weight Loss", "Acupuncture", "Cannabis Clinics", "Naturopathic", "Optometrists"}},
        "Hospitals and clinics":{"includeif":{"Hospital"},"excludeif":{"Physical Therapy", "Rehab", "Retirement Homes", "Veterinarians", "Dentist"}},
        "Optometry":{"includeif":{"Optometrist"},"excludeif":{"Dermatologist"}},
        "Mental health":{"includeif":{"Psychiatrist","Psychologist"},"excludeif":set()},
        "Dental":{"includeif":{"Speech Therapy"},"excludeif":{"Speech"}},
        }
        setClinicalBussinessIds = set()
        fpointerInReview = open( paramFpathInBussiness, 'rt', encoding = 'utf8' )
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)
            bussinessCategoryAttributes = jo['categories']
            setBussinessCategory = set(bussinessCategoryAttributes)
            for akey in dictClinicalBussinessCategories:
                includedifset = dictClinicalBussinessCategories[akey]['includeif']
                excludedifset = dictClinicalBussinessCategories[akey]['excludeif']
                if len(includedifset.intersection(setBussinessCategory) ) != 0:
                    if len(excludedifset.intersection(setBussinessCategory)) != 0 :
                        pass
                    else:
                        bussinessId = jo['business_id']
                        setClinicalBussinessIds.add( bussinessId )
                        break

        fpointerInReview.close()
        #print( setClinicalBussinessIds)

        fpointerInReview = open( paramFpathInReview, 'rt', encoding = 'utf8')
        fpointerOutReview = open( paramFpathOutReview, 'wt', encoding = 'utf8')
        fpointerOutStars = open(paramFpathOutStars,'wt', encoding = 'utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)

            bussinessId = jo['business_id']
            if bussinessId in setClinicalBussinessIds:
                reviewText = jo[ 'text' ]
                reviewText = reviewText.replace('\r\n',' ')
                reviewText = reviewText.replace('\r',' ')
                reviewText = reviewText.replace('\n',' ') # actually in Unbuntu and Windows only this line went into effect
                reviewStar = jo[ 'stars' ]
                fpointerOutReview.write( reviewText + '\n' )
                fpointerOutStars.write( str(reviewStar) + '\n' )

            #print(reviewText)
            #break
        fpointerInReview.close()
        fpointerOutReview.close()
        fpointerOutStars.close()

    def yelpTrainAndTestConstructFromWhole(self, 
        paramFpathInReview, 
        paramFpathInStars, 
        paramFpathOutTrain, 
        paramFpathOutTest,
        paramFpathOutParams,
        paramTrainsetSize = 10000):
        '''
        combine reviews with stars, reshuffle reviews, and split into two sets
        ===================================================
        parameters:
        -----------
        paramFpathInReview: texted review
        paramFpathInStars: stars file
        paramFpathOutTrain: train set
        paramFpathOutTest: test set
        paramFpathOutParams: the parameters needed for training
        paramTrainsetSize: train set size
        
        return:
        -----------
        None
        '''

        fpointerInReview = open(paramFpathInReview, 'rt', encoding = 'utf8')
        fpointerInStars = open(paramFpathInStars, 'rt', encoding = 'utf8')

        def __function4map(elem4map):
            '''
            stripe elem
            ===================================================
            parameters:
            -----------
            elem4map
            
            return:
            -----------
            mapped elem
            '''
            elemstriped = elem4map.strip()
            return elemstriped

        listReviews = list( map( __function4map, fpointerInReview.readlines() ) )
        listStars = list( map( __function4map, fpointerInStars.readlines() ) )
        fpointerInReview.close()
        fpointerInStars.close()

        # zip can only create tuples
        listReviewAndStars = list( zip( listStars,listReviews ) ) # merge two collunms
        
        random.shuffle( listReviewAndStars )

        #-----------------------------------------output in text
        def __textTuple4map( elem ):
            '''
            convert elem tuple to text
            ==============================
            parameters:
            -----------
            elem tuple

            return:
            -----------
            combined text
            '''
            return ' '.join( elem ) + '\n' # join by ' '

        listReviewAndStarsTexted = list( map( __textTuple4map, listReviewAndStars ))
        listTrainset = listReviewAndStarsTexted[ :paramTrainsetSize ]
        listTestset = listReviewAndStarsTexted[ paramTrainsetSize: ]

        fpointerOutTrain = open( os.path.splitext(paramFpathOutTrain)[0] + '.txt', 'wt', encoding = 'utf8' )
        fpointerOutTrain.writelines( listTrainset )
        fpointerOutTrain.close()
        fpointerOutTest = open( os.path.splitext(paramFpathOutTest)[0] + '.txt', 'wt', encoding = 'utf8' )
        fpointerOutTest.writelines( listTestset )
        fpointerOutTest.close()
        
        listReviewAndStarsTexted = None # release memory, Note that in pandas you will have to use fflush
        listTrainset = None
        listTestset = None
        listReviews = None
        listStars = None
        #-----------------------------------------output in text

        #-----------------------------------------output in character encoding
        #----------------------------------------Initialize TextPreProcessor
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
        dicts = [emoticons]
        )
        #----------------------------------------Initialize TextPreProcessor

        (listShuffledStars, listShuffledReviews) = zip( *listReviewAndStars )

        listShuffledReviewsTokenized = list(text_processor.pre_process_docs( listShuffledReviews ) )

        # its elems are int
        tensorShuffledReviewsCharacterEmbedded = batch_to_ids( listShuffledReviewsTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
        # print( listShuffledReviewsCharacterEmbedded[0].size() )
        
        arrayShuffledStars = np.array(listShuffledStars).astype(np.int32)
        arrayShuffledStars = arrayShuffledStars.reshape( (arrayShuffledStars.shape[0], 1) )
        arrayShuffledReviewsCharacterEmbedded = tensorShuffledReviewsCharacterEmbedded.numpy().astype(np.int32)
        arrayShuffledReviewsCharacterEmbedded = arrayShuffledReviewsCharacterEmbedded.reshape( ( arrayShuffledReviewsCharacterEmbedded.shape[0],-1 ) ) # convert to flat except for batch size

        #print(arrayShuffledReviewsCharacterEmbedded.shape)
        arrayConcatenated = np.concatenate( (arrayShuffledStars, arrayShuffledReviewsCharacterEmbedded) , axis = 1) # concatenate must ensure that the two array have the same number of dim

        arrayTrainset = arrayConcatenated[ :paramTrainsetSize ]
        arrayTestset = arrayConcatenated[ paramTrainsetSize: ]
        np.savetxt( paramFpathOutTrain, arrayTrainset, fmt = '%d' )
        np.savetxt( paramFpathOutTest, arrayTestset, fmt = '%d' ) 

        fpointerOutParams = open( paramFpathOutParams, 'wt', encoding = 'utf8' )
        fpointerOutParams.write(
            'TrainingInstances: %d\n'%( arrayTrainset.shape[0] ) +
            'TestingInstances: %d\n'%( arrayTestset.shape[0] ) + 
            'DocumentSeqLen: %d\n'%( tensorShuffledReviewsCharacterEmbedded.size(1) ) + # size(0) is the instanceCount
            'CharacterEmbeddingLen: %d\n'%( tensorShuffledReviewsCharacterEmbedded.size(2) ) 
            )
        fpointerOutParams.close()

        #-----------------------------------------output in character encoding

    def yelpTrainAndTestConstructFromTrainAndTest(self, 
        paramFpathInTrainTxt, 
        paramFpathInTestTxt, 
        paramFpathOutTrain, 
        paramFpathOutTest,
        paramFpathOutParams):
        '''
        convert Train and Test into Character Encoded File
        ===================================================
        parameters:
        -----------
        paramFpathInTrainTxt: train txt file
        paramFpathInTestTxt:  test txt file
        paramFpathOutTrain: train set
        paramFpathOutTest: test set
        paramFpathOutParams: the parameters needed for training

        return:
        -----------
        None
        '''

        fpointerInTrain = open(paramFpathInTrainTxt, 'rt', encoding = 'utf8')
        fpointerInTest = open(paramFpathInTestTxt, 'rt', encoding = 'utf8')

        def __function4map(elem4map):
            '''
            stripe elem
            ===================================================
            parameters:
            -----------
            elem4map
            
            return:
            -----------
            mapped elem
            '''
            (strStar, strDoc) = elem4map.strip().split(' ',1)
            return (strStar, strDoc)

        listTrain = list( map( __function4map, fpointerInTrain.readlines() ) )
        listTest = list( map( __function4map, fpointerInTest.readlines() ) )
        fpointerInTrain.close()
        fpointerInTest.close()

        #-----------------------------------------output in character encoding
        #----------------------------------------Initialize TextPreProcessor
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
        dicts = [emoticons]
        )
        #----------------------------------------Initialize TextPreProcessor

        (listTrainStars, listTrainReviews) = zip( *listTrain )

        listTrainReviewsTokenized = list(text_processor.pre_process_docs( listTrainReviews ) )

        # its elems are int
        tensorTrainReviewsTokenizedCharEncoded = batch_to_ids( listTrainReviewsTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
        # print( listShuffledReviewsCharacterEmbedded[0].size() )
        
        arrayTrainStars = np.array(listTrainStars).astype(np.int32)
        arrayTrainStars = arrayTrainStars.reshape( (arrayTrainStars.shape[0], 1) )
        arrayTrainReviewsTokenizedCharEncoded = tensorTrainReviewsTokenizedCharEncoded.numpy().astype(np.int32)
        arrayTrainReviewsTokenizedCharEncoded = arrayTrainReviewsTokenizedCharEncoded.reshape( ( arrayTrainReviewsTokenizedCharEncoded.shape[0],-1 ) ) # convert to flat except for batch size

        #print(arrayShuffledReviewsCharacterEmbedded.shape)
        arrayTrainConcatenated = np.concatenate( (arrayTrainStars, arrayTrainReviewsTokenizedCharEncoded) , axis = 1) # concatenate must ensure that the two array have the same number of dim
        np.savetxt( paramFpathOutTrain, arrayTrainConcatenated, fmt = '%d' )

        #--------------------------------------------Test
        (listTestStars, listTestReviews) = zip( *listTest )

        listTestReviewsTokenized = list(text_processor.pre_process_docs( listTestReviews ) )

        # its elems are int
        tensorTestReviewsTokenizedCharEncoded = batch_to_ids( listTestReviewsTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
        # print( listShuffledReviewsCharacterEmbedded[0].size() )
        
        arrayTestStars = np.array(listTestStars).astype(np.int32)
        arrayTestStars = arrayTestStars.reshape( (arrayTestStars.shape[0], 1) )
        arrayTestReviewsTokenizedCharEncoded = tensorTestReviewsTokenizedCharEncoded.numpy().astype(np.int32)
        arrayTestReviewsTokenizedCharEncoded = arrayTestReviewsTokenizedCharEncoded.reshape( ( arrayTestReviewsTokenizedCharEncoded.shape[0],-1 ) ) # convert to flat except for batch size

        #print(arrayShuffledReviewsCharacterEmbedded.shape)
        arrayTestConcatenated = np.concatenate( (arrayTestStars, arrayTestReviewsTokenizedCharEncoded) , axis = 1) # concatenate must ensure that the two array have the same number of dim
        np.savetxt( paramFpathOutTest, arrayTestConcatenated, fmt = '%d' )

        fpointerOutParams = open( paramFpathOutParams, 'wt', encoding = 'utf8' )
        fpointerOutParams.write(
            'TrainingInstances: %d\n'%( arrayTrainConcatenated.shape[0] ) +
            'TestingInstances: %d\n'%( arrayTestConcatenated.shape[0] ) + 
            'DocumentSeqLen: %d\n'%( tensorTrainReviewsTokenizedCharEncoded.size(1) ) + # size(0) is the instanceCount
            'CharacterEmbeddingLen: %d\n'%( tensorTrainReviewsTokenizedCharEncoded.size(2) ) 
            )
        fpointerOutParams.close()

        #-----------------------------------------output in character encoding

    def yelpTokenizeAndCharEncode(self, 
        paramFpathInTxt,  
        paramFpathOut, 
        paramFpathOutParams):
        '''
        convert Text into Character Encoded File
        ===================================================
        parameters:
        -----------
        paramFpathInTxt: txt file
        paramFpathOut:  tokenized char encoded txt file
        paramFpathOutParams: the parameters needed for embedding

        return:
        -----------
        None
        '''

        fpointerInTxt = open(paramFpathInTxt, 'rt', encoding = 'utf8')

        def __function4map(elem4map):
            '''
            stripe elem
            ===================================================
            parameters:
            -----------
            elem4map
            
            return:
            -----------
            mapped elem
            '''
            strDoc = elem4map.strip()
            return strDoc

        listTxt = list( map( __function4map, fpointerInTxt.readlines() ) )
        fpointerInTxt.close()

        #-----------------------------------------output in character encoding
        #----------------------------------------Initialize TextPreProcessor
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
        dicts = [emoticons]
        )
        #----------------------------------------Initialize TextPreProcessor

        listTxtTokenized = list(text_processor.pre_process_docs( listTxt ) )

        # its elems are int
        tensorTxtTokenizedCharEncoded = batch_to_ids( listTxtTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
        # print( listShuffledReviewsCharacterEmbedded[0].size() )

        arrayTxtTokenizedCharEncoded = tensorTxtTokenizedCharEncoded.numpy().astype(np.int32)
        arrayTxtTokenizedCharEncoded = arrayTxtTokenizedCharEncoded.reshape( ( arrayTxtTokenizedCharEncoded.shape[0],-1 ) ) # convert to flat except for batch size

        np.savetxt( paramFpathOut, arrayTxtTokenizedCharEncoded, fmt = '%d' )

        fpointerOutParams = open( paramFpathOutParams, 'wt', encoding = 'utf8' )
        fpointerOutParams.write(
            'TestingInstances: %d\n'%( arrayTxtTokenizedCharEncoded.shape[0] ) +
            'DocumentSeqLen: %d\n'%( tensorTxtTokenizedCharEncoded.size(1) ) + # size(0) is the instanceCount
            'CharacterEmbeddingLen: %d\n'%( tensorTxtTokenizedCharEncoded.size(2) ) 
            )
        fpointerOutParams.close()
