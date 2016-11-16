#!/usr/bin/python
#-*- coding: utf-8 -*-
# File Name: main.py
# Created Time: 2016/10/22 16:20:13 

#将最原始的数据进行分词 保存到train_data_fenci.txt

__author__ = 'zle'

import jieba
import jieba.analyse
import sys
import logging
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
    train_keywords_dict = {}
    test_keywords_dict = {}
    leftWords = []
    train_key_word_list=[]
    key_word_list=[]
    test_key_word_list=[]
    test_ID_list=[]
    whole_keyword_list=[]
    for single_query in train_data:
        train_data_count+=1
        single_query_list = single_query.split()#将每一行分块
        ID  = single_query_list.pop(0)#读取不要
        ID_dict[ID]=ID
        Age_dict[ID]=single_query_list.pop(0) 
        Gender_dict[ID]=single_query_list.pop(0)
        Education_dict[ID]=single_query_list.pop(0)
        train_key_word_list=train_keywords_dict[ID]=train_keywords_dict.get(ID,[])
        for j,sentence in enumerate(single_query_list):
            pass
            key_word = jieba.cut(sentence)
            #key_word =  jieba.cut_for_search(sentence)
            #key_word =  jieba.analyse.extract_tags(sentence)
            if( Age_dict[ID]!='0'and Gender_dict[ID]!='0' and Education_dict[ID]!=0):
                for i  in key_word:
                    if(i not in stop):
                        train_key_word_list.append(i)
                        whole_keyword_list.extend(key_word)
                print ('processing %d in %d'%(j,train_data_count))
    with open(test_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        test_data=text[:]
        for single_query in test_data:
            test_data_count+=1
            single_query_list = single_query.split()#将每一行分块       
            ID=single_query_list.pop(0)
            test_ID_list.append(ID)
            test_key_word_list=test_keywords_dict[ID]=test_keywords_dict.get(ID,[])
           #处理分词
            #key_word_list和dic的value指向的是一样的，改变key_word_list就是改变dic的value
            for k,sentence in enumerate(single_query_list):
                test_key_word = jieba.analyse.extract_tags(sentence)#基于 TF-IDF 算法的关键词抽取
                #test_key_word =  jieba.cut_for_search(sentence)# 搜索引擎模式
                #test_key_word = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                for i in test_key_word:
                    if((i not in stop) and  (i in whole_keyword_list)):
                        test_key_word_list.append(i)                     
                print ('test_processing %d in %d'%(k,test_data_count))

    with open('train_data_fenci_no_0.txt','w',encoding='gb18030') as fw_dict_keywords:
     #将分词好的数据存起来
          for key,value in train_keywords_dict.items():
            #if(Age_dict[key]!='0'and Gender_dict[key]!='0'and Education_dict[key]!='0'):
            fw_dict_keywords.write('{0}'.format(key))
            fw_dict_keywords.write(' '+(Age_dict[key]))
            fw_dict_keywords.write(' '+Gender_dict[key])
            fw_dict_keywords.write(' '+Education_dict[key]+' ')
            fw_dict_keywords.write(' '.join((value))+'\n')
    print ('cutWord file save ')

    with open('test_data_fenci_liutao.txt','w') as fw_dict_keywords:
     #将分词好的数据存起来
           for ID  in test_ID_list:
           		fw_dict_keywords.write(ID+' ' +' '.join(test_keywords_dict[ID])+'\n')
    print ('cutWord file save ')  
    train_data.close()
    test_data.close()        

    pass

#获取停用词表  
def loadStopWords():   
    stop = [line.strip()  for line in open('stopwords.txt',encoding='gb18030').readlines() ]   
    return stop  
def main():

    train_file = 'user_tag_query.2W.TRAIN.csv'
    test_file = 'user_tag_query.2W.TEST.csv'
    cutWord(train_file,test_file)

if __name__ == '__main__':
    main()
