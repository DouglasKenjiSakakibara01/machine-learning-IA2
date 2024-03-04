import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv
from sklearn.metrics import confusion_matrix

dataset=[]
real_labels=[]

count=0

with open('sentiment-analysis.csv', newline='') as csvfile:
    
    reader = csv.reader(csvfile)
    
    for row in reader:
        if(count!=0):
          result=(str(row[0]).split(","))
          if(len(result)>1): 
            dataset.append(result[0])
            sentiment=result[1]
            #print(result[0])
            #print(sentiment[1])
            if(sentiment[1] == 'P'):
                real_labels.append(1)
            else:
                real_labels.append(0)
            
            
                    
            
        count+=1  
# vetorização dos textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# 30% do conjunto de dados é utilizado para os testes e 70% é utilizado para o treinamento
x_train, x_test, y_train, y_test = train_test_split(X, real_labels, test_size=0.3, random_state=42)

svc = SVC(kernel='linear')
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

# avaliação do modelo
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)