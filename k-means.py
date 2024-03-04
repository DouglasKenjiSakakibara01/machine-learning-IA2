import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import csv

dataset=[]
real_labels=[]
pred_labels=[]
count=0

with open('sentiment-analysis.csv', newline='') as csvfile:
    
    reader = csv.reader(csvfile)
    
    for row in reader:
        if(count!=0):
          result=(str(row[0]).split(","))
          if(len(result)>1): 
            dataset.append(result[0])
            sentiment=result[1]
            if(sentiment[1] == 'P'):
                real_labels.append(1)
            else:
                real_labels.append(0)
            
            
                    
            
        count+=1  

# vetorizacao dos textos
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(dataset)


k = 2  # número de clusters (positivo, negativo)
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

labels = kmeans.labels_


sentiments = ['Positive', 'Negative']


for i, texto in enumerate(dataset):
    print(f"Texto: '{texto}' -> Sentimento: {sentiments[labels[i]]}")
    sentiment=sentiments[labels[i]]
    
    if(sentiment[0].startswith('P')):
        pred_labels.append(1)
    else:
        pred_labels.append(0)       
        
'''
pred_right=0
for i in range(len(real_labels)):
    if(real_labels[i]==pred_labels[i]):
        pred_right+=1
    print(f"{real_labels[i]} x {pred_labels[i]}\n")
'''

print("Resultado:", adjusted_rand_score(real_labels, pred_labels))


