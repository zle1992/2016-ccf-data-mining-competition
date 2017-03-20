# coding: utf-8
 
import sys
import jieba
import numpy
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics  
from sklearn.metrics import accuracy_score
def input_data(train_file,divide_number,end_number):
    train_words = []
    train_tags_age = []
    train_tags_gender= []
    train_tags_education = []
    test_words = []
    test_tags_age = []
    test_tags_gender= []
    test_tags_education= []
    with open(train_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        train_data=text[:divide_number]   
        for single_query in train_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            train_tags_age.append(single_query_list.pop(0))
            train_tags_gender.append(single_query_list.pop(0))
            train_tags_education.append(single_query_list.pop(0))
            train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
           # print(train_words)
        #print(train_tags_gender)
        test_data=text[divide_number:end_number]   
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            test_tags_age.append(single_query_list.pop(0))
            test_tags_gender.append(single_query_list.pop(0))
            test_tags_education.append(single_query_list.pop(0))
            test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
       # print(test_words)
        #print(test_tags_age)
    print('input_data done!')
    return train_words, train_tags_age,train_tags_gender,train_tags_education, test_words, test_tags_age,test_tags_gender,test_tags_education
 
 
def input_data_write_tags(train_file, test_file):
    train_words = []
    train_tags_age = []
    train_tags_gender= []
    train_tags_education = []
    test_words = []
    test_tags_age = []
    test_tags_gender= []
    test_tags_education= []
   # with open(train_file, 'r') as train_data:
    with open(train_file, 'r') as f:
        text = f.readlines()
        train_data=text[0:100]
        for single_query in train_data:
        
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)#id
            if(single_query_list[0]!='0' and single_query_list[1]!='0' and single_query_list[2]!='0'):
                train_tags_age.append(single_query_list.pop(0))
                train_tags_gender.append(single_query_list.pop(0))
                train_tags_education.append(single_query_list.pop(0))
                train_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
           # print(train_words)
        #print(train_tags_age)
    #with open(test_file, 'r') as test_data:
    with open(test_file, 'r') as f:
        text=f.readlines()
        test_data=text[0:10]
        for single_query in test_data:
            single_query_list = single_query.split(' ')
            single_query_list.pop(0)
            test_words.append((str(single_query_list)).replace(',',' ').replace('\'','').lstrip('[').rstrip(']').replace('\\n',''))
            
            
       # print(test_words)
        #print(test_tags_age)
        
        #处理测试数据
        #print((train_tags))
        #(train_file,'r').close()
    print('input_data done!')
    return train_words, train_tags_age,train_tags_gender,train_tags_education, test_words

 
def write_test_tags(test_file,test_tags_age,test_tags_gender,test_tags_education):
    pass
    test_ID=[]
    with open(test_file,'r') as test_data:
        for single_query in test_data:
            single_query_list=single_query.split(' ')
            test_ID.append(single_query_list[0])

    with open('test_tags_file.csv','w',encoding='gbk') as test_tags_file:
        for x in range(0,len(test_tags_age)):
            test_tags_file.write(test_ID[x]+' '+test_tags_age[x]+' '+test_tags_gender[x]+' '+test_tags_education[x]+'\n')  
  
def vectorize(train_words,test_words,n_feature):
    print ('*************************\nHashingVectorizer\n*************************')  
    v = HashingVectorizer(n_features=n_feature)
    print("n_features:%d"%n_feature)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    print ("the shape of train is "+repr(train_data.shape))
    print ("the shape of test is "+repr(test_data.shape)) 
    
     
    return train_data, test_data
    #print('vectorize done!')
    
    
    

def tfidf_vectorize(train_words,test_words):
    #method 2:TfidfVectorizer  
    print ('*************************\nTfidfVectorizer\n*************************')  
    from sklearn.feature_extraction.text import TfidfVectorizer  
    tv = TfidfVectorizer(sublinear_tf = True) # ,  max_df = 0.5
                                          
    tfidf_train_2 = tv.fit_transform(train_words);  #得到矩阵
    tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_);  
    tfidf_test_2 = tv2.fit_transform(test_words);  
    print ("the shape of train is "+repr(tfidf_train_2.shape))  
    print ("the shape of test is "+repr(tfidf_test_2.shape))
    analyze = tv.build_analyzer()  
    tv.get_feature_names()#statistical features/terms 
    return  tfidf_train_2 ,tfidf_test_2

def evaluate(test_tags_age, test_tags_age_pre,test_tags_gender, test_tags_gender_pre,test_tags_education, test_tags_education_pre):
    print ('age:')
    actual=test_tags_age
    pred=test_tags_age_pre
    print ('precision:{0:.3f}'.format(sklearn.metrics.accuracy_score(actual, pred)))

    print ('gender:')
    actual=test_tags_gender
    pred=test_tags_gender_pre
    print ('precision:{0:.3f}'.format(sklearn.metrics.accuracy_score(actual, pred)))

    print ('education:')
    actual=test_tags_education
    pred=test_tags_education_pre
    print ('precision:{0:.3f}'.format(sklearn.metrics.accuracy_score(actual, pred)))


def SVM(train_data,test_data,train_tags_age,train_tags_gender,train_tags_education): 
#SVM Classifier  
    from sklearn.svm import SVC  
    print ('*************************\nSVM\n*************************' )
    svclf = SVC(kernel = 'linear')#default with 'rbf'  
    svclf.fit(train_data,train_tags_age)  
    pred_tags_age = svclf.predict(test_data) 
    svclf.fit(train_data,train_tags_gender)  
    pred_tags_gender = svclf.predict(test_data)
    svclf.fit(train_data,train_tags_education)  
    pred_tags_education = svclf.predict(test_data)
    #print(pred_tags_gender) 
    #print(train_tags_gender)
    print('clf done!')
    return pred_tags_age,pred_tags_gender,pred_tags_education
def KNN(train_data,test_data,train_tags_age,train_tags_gender,train_tags_education): 
######################################################  
    #KNN Classifier  
    from sklearn.neighbors import KNeighborsClassifier  
    print ('*************************\nKNN\n*************************')  
    knnclf = KNeighborsClassifier()#default with k=5  
    knnclf.fit(train_data,train_tags_age)  
    pred_tags_age = knnclf.predict(test_data) 
    knnclf.fit(train_data,train_tags_gender)  
    pred_tags_gender = knnclf.predict(test_data) 
    knnclf.fit(train_data,train_tags_education)  
    pred_tags_education = knnclf.predict(test_data)
    return pred_tags_age,pred_tags_gender,pred_tags_education  
def test():

    train_file = 'train_data_fenci.txt'
    devide_number=1500
    end_number=2000#17633
    n_feature=100000
    #将数据分为训练与测试，获取训练与测试数据的标签
    train_words, train_tags_age,train_tags_gender,train_tags_education, test_words, test_tags_age,test_tags_gender,test_tags_education = input_data(train_file,devide_number,end_number)
    #向量化
    #train_data,test_data = vectorize(train_words,test_words,n_feature)
    train_data,test_data = tfidf_vectorize(train_words,test_words)
    # 预测
    test_tags_age_pre,test_tags_gender_pre,test_tags_education_pre=SVM(train_data,test_data,train_tags_age,train_tags_gender,train_tags_education)
    #计算正确率
    evaluate(numpy.asarray(test_tags_age), test_tags_age_pre,numpy.asarray(test_tags_gender), test_tags_gender_pre,numpy.asarray(test_tags_education), test_tags_education_pre)
def write():
    train_file = 'train_data_fenci.txt'
    test_file = 'test_data_fenci.txt'
    n_feature=100000
    train_words, train_tags_age,train_tags_gender,train_tags_education, test_words = input_data_write_tags(train_file, test_file)
    #train_data,test_data = vectorize(train_words,test_words,n_feature)
    train_data,test_data = tfidf_vectorize(train_words,test_words)
    test_tags_age_pre,test_tags_gender_pre,test_tags_education_pre=SVM(train_data,test_data,train_tags_age,train_tags_gender,train_tags_education)
    write_test_tags(test_file,test_tags_age_pre,test_tags_gender_pre,test_tags_education_pre)
def main():
    # if len(sys.argv) < 2:
    #     print ('[Usage]: python classifier.py train_file test_file')
    #     sys.exit(0)
    # if(sys.argv[1]=="test"):
    #     test()
    # if(sys.argv[1]=="write"):
    #     write()
    test()
if __name__ == '__main__':
    main()




