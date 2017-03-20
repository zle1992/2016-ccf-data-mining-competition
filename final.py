# coding: utf-8
import sys
import numpy
import sklearn
from time import time
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, bmat
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline   
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV 
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from pprint import pprint
olderr = np.seterr(all='ignore')  # S
def input_data(train_file,divide_number,end_number,tags):
    train_words = []
    train_tags=[]
    test_words = []
    test_tags=[]
    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        train_data=text[:divide_number]   
        for single_query in train_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            if(single_query_list[tags]!='0'):
                train_tags.append(single_query_list[tags])
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
        test_data=text[divide_number:end_number]   
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            if(single_query_list[tags]!='0'):
                test_tags.append(single_query_list[tags])
                single_query_list.pop(0)
                single_query_list.pop(0)
                single_query_list.pop(0)
                test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
       # print(test_words)
        #print(test_tags_age)
    print('input_data done!')
    return train_words, train_tags, test_words, test_tags

 
def input_data_write_tags(train_file, test_file,tags):
    train_words = []
    train_tags=[]
    test_words = []
   # with open(train_file, 'r') as train_data:
    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        train_data=text[0:]   
        for single_query in train_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            #if(single_query_list[tags]!='0'):
            train_tags.append(single_query_list[tags])
            single_query_list.pop(0)
            single_query_list.pop(0)
            single_query_list.pop(0)
            train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
           # print(train_words)
        #print(train_tags)
    with open(test_file, 'r') as f:
        text=f.readlines()
        test_data=text[0:]
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)
            test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
            
   
    print('input_data done!')
    return train_words, train_tags,test_words

 
def write_test_tags(test_file,test_tags_age,test_tags_gender,test_tags_education):
    test_ID=[]
    with open(test_file,'r') as test_data:
        for single_query in test_data:
            single_query_list=single_query.split(' ')
            test_ID.append(single_query_list[0])

    with open('test_tags_file_cv_hv_chi2_59000_15000_13500.csv','w',encoding='gbk') as test_tags_file:
        for x in range(0,len(test_tags_age)):
            test_tags_file.write(test_ID[x]+' '+test_tags_age[x]+' '+test_tags_gender[x]+' '+test_tags_education[x]+'\n')  
  
def vectorize(train_words,test_words,n_feature):
    print ('*************************HashingVectorizer*************************')  
    v = HashingVectorizer(n_features=n_feature,non_negative =True)
    print("n_features:%d"%n_feature)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    print ("the shape of train is "+repr(train_data.shape))
    print ("the shape of test is "+repr(test_data.shape)) 
    
     
    return train_data, test_data
    #print('vectorize done!')
    
    
def tfidf_vectorize_1(train_words,test_words):
    #method 2:TfidfVectorizer  
    print ('*************************\nTfidfVectorizer\n*************************')   
    tv = TfidfVectorizer(sublinear_tf = True)#,min_df=9.0090090090090091e-05) # 
                                          
    tfidf_train_2 = tv.fit_transform(train_words);  #得到矩阵
    tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_);  
    tfidf_test_2 = tv2.fit_transform(test_words);  
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    return  tfidf_train_2 ,tfidf_test_2   

def tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality):
    #method 2:TfidfVectorizer  
    print ('*************************TfidfVectorizer*************************')   
    t0=time()
    tv = TfidfVectorizer(sublinear_tf = True) # 
                                          
    tfidf_train_2 = tv.fit_transform(train_words);  #得到矩阵
    tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_);  
    tfidf_test_2 = tv2.fit_transform(test_words);  
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    train_data,test_data=feature_selection_chi2(tfidf_train_2,train_tags,tfidf_test_2,n_dimensionality) 
    print("done in %0.3fs." % (time() - t0))
    return  train_data,test_data
def feature_union_tv_hv(train_words,train_tags,test_words,test_tags,n_feature,n_dimensionality):
    print('*************************feature_union_tv_hv*************************')
    hv = HashingVectorizer(n_features=n_feature,non_negative =True)
    tv1 = TfidfVectorizer(sublinear_tf = True,  max_df = 0.5) #
    train_combined_features = FeatureUnion([('hv',hv),('tv1',tv1)])
    train_data=train_combined_features.fit_transform(train_words)
    print ("the shape of train is "+repr(train_data.shape))  
    tv2 = TfidfVectorizer(vocabulary = tv1.vocabulary_ ) 
    test_combined_features = FeatureUnion([('hv',hv),('tv2',tv2)])  
    test_data=test_combined_features.fit_transform(test_words)
    print ("the shape of train is "+repr(test_data.shape)) 
    train_data,test_data=feature_selection_chi2(train_data,train_tags,test_data,n_dimensionality) 
    return train_data,test_data
def feature_union_lda_tv(train_words,test_words,train_tags,test_tags,n_dimensionality,n_topics):
 	 #LDA主题提取
    print('*************************feature_union_lda_tv*************************')
    train_data_lda,test_data_lda = LDA(train_words,test_words,n_topics)
     #归一化lda
    train_data_lda_normalize=preprocessing.normalize(train_data_lda, norm='l2')
    test_data_lda_normalize=preprocessing.normalize(test_data_lda, norm='l2')
    # #向量化
    train_data_tv,test_data_tv = tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality)  
    #特征矩阵合并
    train_data=bmat([[train_data_lda_normalize, train_data_tv]])
    test_data=bmat([[test_data_lda_normalize, test_data_tv]])
    return train_data,test_data
    
def sgd_single(train_data,test_data,train_tags):
    print ('*************************\nSVM\n*************************' )
    clf = linear_model.SGDClassifier()
    clf.fit(train_data,train_tags)  
    pred_tags = clf.predict(test_data) 
    print('clf done!')
    return pred_tags    
    
def SVM_single(train_data,test_data,train_tags): 
#SVM Classifier  
    from sklearn.svm import SVC  
    print ('******************************SVM*****************************' )
    t0=time()
    svclf = SVC(kernel = 'linear')#default with 'rbf'  
    svclf.fit(train_data,train_tags)  
    pred_tags = svclf.predict(test_data) 
    print("done in %0.3fs." % (time() - t0))
    print('clf done!')
    return pred_tags
def evaluate_single(test_tags, test_tags_pre):
    actual=test_tags
    pred=test_tags_pre
    print ('accuracy_score:{0:.3f}'.format(accuracy_score(actual, pred)))
    print('confusion_matrix:')
    print(confusion_matrix(actual, pred))

def feature_selection_chi2(train_data,train_tags,test_data,n_dimensionality):

    print('feature_selection_chi2'+'\n'+'n_dimensionality:%d' %n_dimensionality)
    ch2= SelectKBest(score_func=chi2, k=n_dimensionality)
    train_data=ch2.fit_transform(train_data,train_tags)
    test_data=ch2.transform(test_data)
    return train_data  , test_data 

def LDA(train_words,test_words,n_topics):
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    t0 = time()
    train_tf = tf_vectorizer.fit_transform(train_words)
    test_tf = tf_vectorizer.fit_transform(test_words)
    print("done in %0.3fs." % (time() - t0))
    lda = LatentDirichletAllocation(n_topics=n_topics,max_iter=10,learning_method='online')
    t0 = time()
    print('n_topics:%d' %n_topics)
    train_word_lda=lda.fit_transform(train_tf)
    test_word_lda=lda.fit_transform(test_tf)
    print(" done in %0.3fs." % (time() - t0))
    return  train_word_lda,test_word_lda      
def test():
    #            标签（年龄性别学历）  卡方选取后的维数    主题个数
    test_single(0,59500,100)
    test_single(1,12000,5)
    test_single(2,130,10)
def test_single(tags,n_dimensionality,n_topics):
    train_file = 'train_data_fenci.txt'
    devide_number=15500
    end_number=17633#17633
    n_feature=320000
    print('file:'+train_file)
    print('tags:%d   ' %tags )
    tag="age"
    #将数据分为训练与测试，获取训练与测试数据的标签
    train_words, train_tags,test_words, test_tags = input_data(train_file,devide_number,end_number,tags)
    # #向量化
    # #hv
    #train_data,test_data = vectorize(train_words,test_words,n_feature)
    # #tv
    #train_data,test_data= tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality)
    train_data,test_data= tfidf_vectorize_1(train_words,test_words)
 	#lda_tv
    #train_data,test_data=feature_union_lda_tv(train_words,test_words,train_tags,test_tags,n_dimensionality,n_topics)
    #tv_hv 
    #train_data,test_data = feature_union_tv_hv(train_words,train_tags,test_words,test_tags,n_feature,n_dimensionality)
    
    test_tags_pre=SVM_single(train_data,test_data,train_tags)
    #计算正确率
    evaluate_single(numpy.asarray(test_tags), test_tags_pre)
def write_single(train_file,test_file,tags,n_dimensionality,n_topics):
    n_feature=320000
    train_words, train_tags, test_words = input_data_write_tags(train_file, test_file,tags)
    # #向量化
    # #hv
    #train_data,test_data = vectorize(train_words,test_words,n_feature)
    # #tv
    train_data,test_data= tfidf_vectorize(train_words,train_tags,test_words,test_tags,n_dimensionality)
 	#lda_tv
    #train_data,test_data=feature_union_lda_tv(train_words,test_words,train_tags,test_tags,n_dimensionality,n_topics)
    #tv_hv 
    #train_data,test_data = feature_union_tv_hv(train_words,train_tags,test_words,test_tags,n_feature,n_dimensionality)
    
    test_tags_pre=SVM_single(train_data,test_data,train_tags)
    return test_tags_pre
def write():
    train_file='train_data_fenci.txt'
    test_file = 'test_data_fenci.txt'
    test_tags_age_pre=write_single(train_file,test_file,0,59000,50)
    test_tags_gender_pre=write_single_1(train_file,test_file,1,15000,50)
    test_tags_education_pre=write_single(train_file,test_file,2,135000,50)
    write_test_tags(test_file,test_tags_age_pre,test_tags_gender_pre,test_tags_education_pre)
#########################################
def optimize_single(tags):
    train_file = 'train_data_fenci.txt'
    devide_number=15000
    end_number=17633#17633
    n_feature=1000
    print('file:'+train_file)
    print('tags:%d   ' %tags )
    train_words, train_tags,test_words, test_tags = input_data(train_file,devide_number,end_number,tags)
    train_data,test_data = tfidf_vectorize_1(train_words,test_words)

    pipeline = Pipeline([     
    #('TfidfVectorizer',TfidfVectorizer(sublinear_tf = True)),
    ('feature_selection',SelectKBest(chi2)),
    ('clf',SGDClassifier()), 
    ]);

    a=np.linspace(10000,200000,num=1000,dtype=int)
   # min_df=np.linspace(0,0.01,num=1000,dtype=float)

    parameters={
    #'TfidfVectorizer__sublinear_tf':[True,False],
   # 'TfidfVectorizer__min_df':list(min_df),
    #'TfidfVectorizer__sublinear_tf':[True,False],

    #'feature_selection__score_func':[chi2],
   'feature_selection__k':list(a)
    }
    grid_search = GridSearchCV(pipeline,parameters,n_jobs =6,verbose=1);  
    print("Performing grid search...")  
    print("pipeline:", [name for name, _ in pipeline.steps])  
    print("parameters:")  
    #pprint(parameters)  
  
    grid_search.fit(train_words, train_tags)  
    print("Best score: %0.3f" % grid_search.best_score_)  
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
def optimize():
    optimize_single(0)
    optimize_single(1)
    optimize_single(2)
def main():
    if len(sys.argv) < 2:
        print ('[Usage]: python classifier.py train_file test_file')
        sys.exit(0)
    if(sys.argv[1]=="test"):
        test()
    if(sys.argv[1]=="write"):
    	write()
    if(sys.argv[1]=="optimize"):
   		optimize()

if __name__ == '__main__':
    main()
