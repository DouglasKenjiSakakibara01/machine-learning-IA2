import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.svm import SVC
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
dataset=[]
dataset_size=0
real_labels=[]

count=0
# Extraindo as informações do dataset
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
        
# Vetorização dos textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

##################### Holdout ##################
# 30% do conjunto de dados é utilizado para os testes e 70% é utilizado para o treinamento
x_train, x_test, y_train, y_test = train_test_split(X, real_labels, test_size=0.3, random_state=42)

svc = SVC(kernel='linear')


svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

# Avaliação do modelo
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

############# Validação Cruzada ############################

num_folds = 5
fold_size = X.shape[0] // num_folds

svc = SVC(kernel='linear')

 # Conversão da matriz esparsa X para uma matriz densa. 
X_dense = X.toarray()

count_fold=1
for i in range(num_folds):
    # Dividindo os conjuntos para cada fold
    inicio = i * fold_size
    fim = (i + 1) * fold_size
    data_train=[]
    data_test=[]
    labels_train=[]
    labels_test=[]


    for j in range(0, inicio):
       data_train.append(X_dense[j])
       labels_train.append(real_labels[j])
    
    for j in range(fim, X.shape[0]):
       data_train.append(X_dense[j])
       labels_train.append(real_labels[j])

    for j in range(inicio,fim):
       data_test.append(X_dense[j])
       labels_test.append(real_labels[j])   
    
    svc.fit(data_train, labels_train)

    # Avaliação modelo
    y_pred = svc.predict(data_test)
    acc_score= accuracy_score(labels_test, y_pred)
    print(f'**********Fold {count_fold}****************')
    print(acc_score)
    print(labels_test)
    print(y_pred )
    count_fold+=1
    
