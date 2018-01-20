import os
import math

#building a vocabulary vector of single occurences of words across all documents
path = './docs'
temp = []
vocab = []
for fname in os.listdir(path):
    if fname != '.DS_Store':
        temp.append(fname)
        with open('./docs/' + fname) as f:
            content = f.readlines()
            line = content
            content = [x.strip() for x in content]
            vocab = vocab + content
vocab = list(set(vocab));

#calculating word frequency vectors for each document
temp_doc=[]
for fname in os.listdir(path):
    if fname != '.DS_Store':
        temp1 = [0] * len(vocab)
        with open('./docs/' + fname) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for x in content:
                for i in range(0, len(vocab)):
                    if (x == vocab[i]) :
                        temp1[i] += 1
        temp_doc.append(temp1)
        
#calculating word frequency vectors for each query           
path= './queries'
temp_query=[]
for fname in os.listdir(path):
    if fname != '.DS_Store':
        temp2 = [0] * len(vocab)
        with open('./queries/' + fname) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for x in content:
                for i in range(0, len(vocab)):
                    if (x == vocab[i]) :
                        temp2[i] += 1
        temp_query.append(temp2)
        
#Computing dot product and cosine similarity
w,h = len(temp_query),len(temp_doc)
dictionary = dict(zip(range(0,500),temp))
for k in range(0,len(temp_query)):
    print("Top ten document matches for query %d using dot product similarity are" % (k+1))
    S_dot = []
    S_cos = []
    for j in range(0,len(temp_doc)):
        dot_sum = 0
        doc_sum=0
        query_sum=0
        for i in range(0,len(vocab)):
            dot_sum=dot_sum+((temp_doc[j][i])*(temp_query[k][i]))
            doc_sum=doc_sum+((temp_doc[j][i])*(temp_doc[j][i]))
            query_sum=query_sum+((temp_query[k][i])*(temp_query[k][i]))
        S_dot.append(dot_sum)
        S_cos.append(dot_sum/(math.sqrt(doc_sum)*math.sqrt(query_sum)))

#Top ten documents
        
    top_ten_dot = sorted(range(len(S_dot)), key=lambda i: S_dot[i])[-10:]
    top_ten_cos = sorted(range(len(S_cos)), key=lambda i: S_cos[i])[-10:]


    for g in top_ten_dot[::-1]:
        print(dictionary[g])


    print("Top ten document matches for query %d using cosine similarity are" % (k+1))
    for g in top_ten_cos[::-1]:
        print(dictionary[g])
        
        
