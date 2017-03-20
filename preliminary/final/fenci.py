#!/usr/bin/python
#-*- coding: utf-8 -*-
# File Name: main.py
# Created Time: 2016/10/22 16:20:13 



#将最原始的数据进行分词 保存到train_data_fenci.txt

#用词典做的，估计效率会比较低


__author__ = 'zle'

import jieba
import jieba.analyse
import sys
import logging
def cutWords(msg,stopWords):  
    seg_list = jieba.cut(msg,cut_all=False)  
    #key_list = jieba.analyse.extract_tags(msg,20) #get keywords   
    leftWords = []   
    for i in seg_list:  
        if (i not in stopWords):  
            leftWords.append(i)          
    return leftWords  
def cutWord(train_file,test_file):
    train_data = open(train_file,encoding='gb18030')
    test_data = open(test_file,encoding='gb18030')   
    stop=loadStopWords()
    train_data_count=0
    test_data_count=0
    ID_dict={}
    Age_dict={}
    Gender_dict={}
    Education_dict={}
    feature_dict={}
    keywords_dict = {}
    test_keywords_dict = {}
    leftWords = []
    train_key_word_list=[]
    test_key_word_list=[]
    test_ID_list=[]
    # for single_query in train_data:
    #     train_data_count+=1
    #     single_query_list = single_query.split()#将每一行分块
    #     ID  = single_query_list.pop(0)#读取不要
    #     ID_dict[ID]=ID
    #     Age_dict[ID]=single_query_list.pop(0) 
    #     Gender_dict[ID]=single_query_list.pop(0)
    #     Education_dict[ID]=single_query_list.pop(0)
    #     key_word_list=keywords_dict[ID]=keywords_dict.get(ID,[])
    #     for j,sentence in enumerate(single_query_list):
    #         pass
    #         #key_word =  jieba.cut_for_search(sentence)
    #         key_word =  jieba.analyse.extract_tags(sentence)
    #         #key_word_list.extend(key_word)
    #         #print(key_word)
    #         if( Age_dict[ID]!='0'and Gender_dict[ID]!='0' and Education_dict[ID]!=0):
    #             for i  in key_word:
    #                 if(i not in stop):
    #                     key_word_list.append(i)
    #                     #whole_keyword_list.extend(key_word)
    #             #print(stop)
    #             print ('processing %d in %d'%(j,train_data_count))
    for single_query in test_data:
        test_data_count+=1
        single_query_list = single_query.split()#将每一行分块
        ID  = single_query_list.pop(0)#读取不要  
       # logging.info(age_dict)
       #处理分词
        test_key_word_list=test_keywords_dict[ID]=test_keywords_dict.get(ID,[])
        #key_word_list和dic的value指向的是一样的，改变key_word_list就是改变dic的value
        for k,sentence in enumerate(single_query_list):
            test_key_word = jieba.analyse.extract_tags(sentence)#基于 TF-IDF 算法的关键词抽取
            #test_key_word =  jieba.cut_for_search(sentence)# 搜索引擎模式
            #test_key_word = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
            for i  in test_key_word:
                if(i not in stop):
                    test_key_word_list.append(i) 
            print ('test_processing %d in %d'%(k,test_data_count))
    with open('train_data_fenci.txt','w') as fw_dict_keywords:
     #将分词好的数据存起来
          for key,value in keywords_dict.items():
            if(Age_dict[key]!='0'and Gender_dict[key]!='0'and Education_dict[key]!='0'):
                fw_dict_keywords.write('{0}'.format(key))
                fw_dict_keywords.write(' '+(Age_dict[key]))
                fw_dict_keywords.write(' '+Gender_dict[key])
                fw_dict_keywords.write(' '+Education_dict[key]+' ')
                fw_dict_keywords.write(' '.join((value))+'\n')
    print ('cutWord file save in train_data_fenci.txt')

    with open('test_data_fenci.txt','w') as fw_dict_keywords:
     #将分词好的数据存起来
           for key,value in test_keywords_dict.items():
                fw_dict_keywords.write('{0}'.format(key)+' ')
                fw_dict_keywords.write(' '.join((value))+'\n')  
    print ('cutWord file save in test_data_fenci.txt')  
    train_data.close()
    test_data.close()        
                
    
                            
                    
            

       #处理分词
      #  
        #key_word_list和dic的value指向的是一样的，改变key_word_list就是改变dic的value
     #   
      #      #基于 TF-IDF 算法的关键词抽取
            #key_word =  jieba.cut_for_search(sentence)# 搜索引擎模式
            #key_word = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 

   
    pass

#获取停用词表  
def loadStopWords():   
    stop = [line.strip()  for line in open('stopwords.txt',encoding='gb18030').readlines() ]   
    return stop  
def get_train_table(train_data):
    ID_dict={}
    Age_dict={}
    Gender_dict={}
    Education_dict={}
    feature_dict={}
    keywords_dict = {}
    for single_query in train_data:
        single_query_list = single_query.split()
        ID = single_query_list.pop(0)
        ID_dict[ID]=ID
        Age_dict[ID]=single_query_list.pop(0)
        Gender_dict[ID]=single_query_list.pop(0)
        Education_dict[ID]=single_query_list.pop(0)
    return ID_dict,Age_dict,Gender_dict,Education_dict
def main():

    train_file = 'user_tag_query.2W.TRAIN.csv'
    test_file = 'user_tag_query.2W.TEST.csv'
        #print(get_train_table(train_data))
    cutWord(train_file,test_file)

    #print(loadStopWords())




if __name__ == '__main__':
    main()