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
    train_ID_list=[]
    train_Age_list=[]
    train_Gender_list=[]
    train_Education_list=[]
    train_key_word=[]
    train_key_word_list=[]
    key_word_list=[]
    test_key_word=[]
    test_key_word_list=[]
    test_ID_list=[]
    whole_keyword_list=[]
    with open(train_file, 'r',encoding='gb18030') as f:
    	text=f.readlines()
    	train_data=text[:]
    	for single_query in train_data:
	        train_data_count+=1  
	        single_query_list = single_query.split()#将每一行分块
	        train_ID_list.append(single_query_list.pop(0))#读取不要
	        train_Age_list.append(single_query_list.pop(0))
	        train_Gender_list.append(single_query_list.pop(0))
	        train_Education_list.append(single_query_list.pop(0))
	        for j,sentence in enumerate(single_query_list):
	            key_word =  jieba.analyse.extract_tags(sentence)
	            for i in key_word:
	            	if(i not in stop):
	            		train_key_word.append(i)
	            		print('processing %d in %d'%(j,train_data_count))
	        train_key_word.append(',')
	        train_key_word_list=' '.join(train_key_word).split(',') 
    with open(test_file, 'r',encoding='gb18030') as f:
        text=f.readlines()
        test_data=text[:]
        for single_query in test_data:
            test_data_count+=1
            single_query_list = single_query.split()#将每一行分块       
            test_ID_list.append(single_query_list.pop(0))
            for k,sentence in enumerate(single_query_list):
                key_word_seg = jieba.analyse.extract_tags(sentence)#基于 TF-IDF 算法的关键词抽取
                #test_key_word =  jieba.cut_for_search(sentence)# 搜索引擎模式
                #test_key_word = jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                for i in key_word_seg:
                    if(i not in stop):
                        test_key_word.append(i) 
                        print ('test_processing %d in %d'%(k,test_data_count))                    
            test_key_word.append(',')	
            test_key_word_list=' '.join(test_key_word).split(',')	              
    with open('train_data_fenci_50000.txt','w',encoding='gb18030') as fw_dict_keywords:
     #将分词好的数据存起来
        for x in range(0,len(train_ID_list)):
        	if(train_Education_list[x]!='0' and train_Age_list[x]!='0' and train_Gender_list[x]!=0):
       			fw_dict_keywords.write(train_ID_list[x]+'  '+train_Age_list[x]+' '+train_Gender_list[x]+' '+train_Education_list[x]+' '+train_key_word_list[x]+'\n')
    print ('train_cutWord file save ')
    with open('test_data_fenci_liutao.txt','w') as fw_dict_keywords:
     #将分词好的数据存起来
           for x  in range(0,len(test_ID_list)):
           		fw_dict_keywords.write(test_ID_list[x]+' '+test_key_word_list[x]+'\n')
    print ('test_cutWord file save ')         
#获取停用词表  
def loadStopWords():   
    stop = [line.strip()  for line in open('stopwords.txt',encoding='gb18030').readlines() ]   
    return stop  
def main():

    train_file = 'user_tag_query.10W.TRAIN.csv'
    test_file = 'user_tag_query.10W.TEST.csv'
    cutWord(train_file,test_file)

if __name__ == '__main__':
    main()