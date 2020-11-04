import csv
import math
import colorama
from colorama import Fore

train_label_rows=[]
train_data_rows=[]
train_label_columns=[]
train_data_columns=[]
train_strt_of_docs=[]                 #start of doc no. row for each group

train_priors=[]       #each entry indicates the prior probabilities of each class j
train_count=[]        #each entry indicates total no. of documents in each class (omega)j
train_num_words=[]     #each entry indicates total no. of words in all documentes in class (omega)j
path_to_vocab = 'D:/MS/Sem2/COMS 573/Lab1/20newsgroups/vocabulary.txt'
path_to_train_data = 'D:/MS/Sem2/COMS 573/Lab1/20newsgroups/train_data.csv'
path_to_train_label = 'D:/MS/Sem2/COMS 573/Lab1/20newsgroups/train_label.csv'
path_to_test_data = 'D:/MS/Sem2/COMS 573/Lab1/20newsgroups/test_data.csv'
path_to_test_label = 'D:/MS/Sem2/COMS 573/Lab1/20newsgroups/test_label.csv'

with open(path_to_vocab, 'r') as file:
    lines = len(file.readlines()) #total no. of distinct words in the vocabulary

train_nk = [[0] * lines for n in range(20)]  #no. of times word wk occurs in all documents in class (omega)j
train_Pmle = [[] * lines for n in range(20)] #Maximum Likelihood Estimator Pmle(wk|(omega)j)
train_Pbe = [[] * lines for n in range(20)] #Bayesian Estimator  Pbe(wk|(omega)j)

with open(path_to_train_label, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        train_label_rows.append(row)
for row in train_label_rows:
	for col in row:
		train_label_columns.append(col)


train_num_docs = len(train_label_columns)						#total no. of training documents

for i in range(1,21):
	 train_count.append(train_label_columns.count(str(i)))
for i in range(0,20):
	train_priors.append(train_count[i] / train_num_docs)
print(Fore.RED + "1. Class Priors")
for i in range(0,20):
    print(Fore.BLACK + "P(Omega = " + str(i+1) + ") = ",end="")
    print(round(train_priors[i],4))
print('\n')

with open(path_to_train_data, 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		train_data_rows.append(row)
for row in train_data_rows:
	for col in row:
		train_data_columns.append(col)
train_data_columns = list(map(int, train_data_columns))     #converting elements datatype from string to integer    
#print(len(train_data_columns))

sum=0
k=0
j=0
for i in range(0,20):        
    while train_data_columns[j] <= (train_count[i]+k):
        if(train_data_columns[j] < k):
            j = j + 3
            continue
        sum += train_data_columns[j+2]
        j = j + 3
        if(j > len(train_data_columns)-1):
            break
    k += train_count[i]
    train_num_words.insert(i, sum)
    sum=0
#print(num_words)        
thresh=0.000363
sum=0
for i in range(0,20):
    sum+= train_count[i]
    train_strt_of_docs.append(sum)
#print(strt_of_docs)


l=0   
m=0
for i in train_strt_of_docs:
    while(train_data_columns[l] <= i):
        train_nk[train_strt_of_docs.index(i)][train_data_columns[l+1]-1] += train_data_columns[l+2]
        l=l+3
        if(l > len(train_data_columns)-1):
            break
        if(train_data_columns[l] > i):
            break


for i in range(0,20):
    for j in range(0, lines):
        train_Pbe[i].append((train_nk[i][j]+1)/(train_num_words[i]+lines))
        train_Pmle[i].append((train_nk[i][j])/train_num_words[i])

j=1
train_count_index = [0] * train_num_docs
train_wordid=[[] for i in range (train_num_docs)]
for i in range(0, len(train_data_columns), 3):
    if(train_data_columns[i] == j):
        train_count_index[j-1] += 1
        train_wordid[j-1].append(train_data_columns[i+1])
    else:
        j += 1
        train_count_index[j-1] += 1
        train_wordid[j-1].append(train_data_columns[i+1])

'''##########Calculating omega nb############'''
train_global_sum_mle=[0] * 20
train_global_sum_be=[0] * 20
train_maximum=0
train_sum_mle=0
train_sum_be=0
train_nb_mle=[]   #omega NB with Pmle for train data
train_nb_be=[]   #omega NB with Pbe for train data
for i in range(0, train_num_docs):
    for j in range(0, 20):
        for k in range(0,train_count_index[i]):
            if(train_Pmle[j][train_wordid[i][k]-1] ==0):
                train_Pmle[j][train_wordid[i][k]-1] = 0.00000001
            train_sum_mle += math.log(train_Pmle[j][train_wordid[i][k]-1])
            train_sum_be += math.log(train_Pbe[j][train_wordid[i][k]-1])                   
        train_global_sum_mle[j] = math.log(train_priors[j]) + train_sum_mle
        train_global_sum_be[j] = math.log(train_priors[j]) + train_sum_be
        train_sum_mle=0
        train_sum_be=0
    train_nb_mle.append(train_global_sum_mle.index(max(train_global_sum_mle))+1)
    train_nb_be.append(train_global_sum_be.index(max(train_global_sum_be))+1)


train_count_be=0
train_label_columns = list(map(int, train_label_columns))
for i in range(0, train_num_docs):
    if(train_label_columns[i] == train_nb_be[i]):
        train_count_be += 1

j=0
k=0
train_class_count_be=[0]*20
train_wrong_pred_be=[[0]*20 for m in range(0,20)]
for i in range(0,20):        
    while j+1 <= (train_count[i]+k):
        if(j+1 < k):
            j = j + 1
            continue
        if(train_label_columns[j] == train_nb_be[j]):
            train_class_count_be[i] += 1
            j +=1
        else:
            train_wrong_pred_be[i][train_nb_be[j]-1] += 1
            j = j + 1
        if(j > len(train_label_columns)-1):
            break
    k += train_count[i]    


train_count_mle=0
for i in range(0, train_num_docs):
    if(train_label_columns[i] == train_nb_mle[i]):
        train_count_mle += 1

j=0
k=0
train_class_count_mle=[0]*20
train_wrong_pred_mle=[[0]*20 for m in range(0,20)]
for i in range(0,20):        
    while j+1 <= (train_count[i]+k):
        if(j+1 < k):
            j = j + 1
            continue
        if(train_label_columns[j] == train_nb_mle[j]):
            train_class_count_mle[i] += 1
            j +=1
        else:
            train_wrong_pred_mle[i][train_nb_mle[j]-1] += 1
            j = j + 1
        if(j > len(train_label_columns)-1):
            break
    k += train_count[i]    
        

'''########################################## TEST DATA #################################################'''
test_label_rows=[]
test_data_rows=[]
test_label_columns=[]
test_data_columns=[]
test_strt_of_docs=[]                 #start of doc no. row for each group

test_priors=[]       #each entry indicates the prior probabilities of each class j
test_count=[]        #each entry indicates total no. of documents in each class (omega)j
test_num_words=[]     #each entry indicates total no. of words in all documentes in class (omega)


with open(path_to_vocab, 'r') as file:
    lines = len(file.readlines()) #total no. of distinct words in the vocabulary

test_nk = [[0] * lines for n in range(20)]  #no. of times word wk occurs in all documents in class (omega)j
test_Pmle = [[] * lines for n in range(20)] #Maximum Likelihood Estimator Pmle(wk|(omega)j)
test_Pbe = [[] * lines for n in range(20)] #Bayesian Estimator  Pbe(wk|(omega)j)

with open(path_to_test_label, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        test_label_rows.append(row)
for row in test_label_rows:
	for col in row:
		test_label_columns.append(col)


test_num_docs = len(test_label_columns)						#total no. of test documents

for i in range(1,21):
	 test_count.append(test_label_columns.count(str(i)))
for i in range(0,20):
	test_priors.append(test_count[i] / test_num_docs)

with open(path_to_test_data, 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		test_data_rows.append(row)
for row in test_data_rows:
	for col in row:
		test_data_columns.append(col)
test_data_columns = list(map(int, test_data_columns))     #converting elements datatype from string to integer    
#print(len(train_data_columns))

sum=0
k=0
j=0
for i in range(0,20):        
    while test_data_columns[j] <= (test_count[i]+k):
        if(test_data_columns[j] < k):
            j = j + 3
            continue
        sum += test_data_columns[j+2]
        j = j + 3
        if(j > len(test_data_columns)-1):
            break
    k += test_count[i]
    test_num_words.insert(i, sum)
    sum=0
#print(num_words)        

sum=0
for i in range(0,20):
    sum+= test_count[i]
    test_strt_of_docs.append(sum)
#print(strt_of_docs)


l=0   
m=0
for i in test_strt_of_docs:
    while(test_data_columns[l] <= i):
        test_nk[test_strt_of_docs.index(i)][test_data_columns[l+1]-1] += test_data_columns[l+2]
        l=l+3
        if(l > len(test_data_columns)-1):
            break
        if(test_data_columns[l] > i):
            break


for i in range(0,20):
    for j in range(0, lines):
        test_Pbe[i].append((test_nk[i][j]+1)/(test_num_words[i]+lines))
        test_Pmle[i].append((test_nk[i][j])/test_num_words[i])


j=1
test_count_index = [0] * test_num_docs
test_wordid=[[] for i in range (test_num_docs)]
for i in range(0, len(test_data_columns), 3):
    if(test_data_columns[i] == j):
        test_count_index[j-1] += 1
        test_wordid[j-1].append(test_data_columns[i+1])
    else:
        j += 1
        test_count_index[j-1] += 1
        test_wordid[j-1].append(test_data_columns[i+1])

'''##########Calculating omega nb############'''
test_global_sum_mle=[0] * 20
test_global_sum_be=[0] * 20
test_maximum=0
test_sum_mle=0
test_sum_be=0
test_nb_mle=[]   #omega NB with Pmle
test_nb_be=[]   #omega NB with Pbe
for i in range(0, test_num_docs):
    for j in range(0, 20):
        for k in range(0,test_count_index[i]):
            if(test_Pmle[j][test_wordid[i][k]-1] ==0):
                test_Pmle[j][test_wordid[i][k]-1] = thresh
            test_sum_mle += math.log(test_Pmle[j][test_wordid[i][k]-1])
            test_sum_be += math.log(test_Pbe[j][test_wordid[i][k]-1])                   
        test_global_sum_mle[j] = math.log(test_priors[j]) + test_sum_mle
        test_global_sum_be[j] = math.log(test_priors[j]) + test_sum_be
        test_sum_mle=0
        test_sum_be=0
    test_nb_mle.append(test_global_sum_mle.index(max(test_global_sum_mle))+1)
    test_nb_be.append(test_global_sum_be.index(max(test_global_sum_be))+1)



test_count_be=0
test_label_columns = list(map(int, test_label_columns))
for i in range(0, test_num_docs):
    if(test_label_columns[i] == test_nb_be[i]):
        test_count_be += 1

#print(Fore.BLACK + "Overall Accuracy with Pbe =", end = " ")
#print(test_count_be/test_num_docs)

j=0
k=0
test_class_count_be=[0]*20
test_wrong_pred_be=[[0]*20 for m in range(0,20)]
for i in range(0,20):        
    while j+1 <= (test_count[i]+k):
        if(j+1 < k):
            j = j + 1
            continue
        if(test_label_columns[j] == test_nb_be[j]):
            test_class_count_be[i] += 1
            j +=1
        else:
            test_wrong_pred_be[i][test_nb_be[j]-1] += 1
            j = j + 1
        if(j > len(test_label_columns)-1):
            break
    k += test_count[i]    
#print(test_class_count_be)
#print(test_wrong_pred_be)    


test_count_mle=0
for i in range(0, test_num_docs):
    if(test_label_columns[i] == test_nb_mle[i]):
        test_count_mle += 1

#print("Overall Accuracy with Pmle =", end = " ")
#print(test_count_mle/test_num_docs)

j=0
k=0
test_class_count_mle=[0]*20
test_wrong_pred_mle=[[0]*20 for m in range(0,20)]
for i in range(0,20):        
    while j+1 <= (test_count[i]+k):
        if(j+1 < k):
            j = j + 1
            continue
        if(test_label_columns[j] == test_nb_mle[j]):
            test_class_count_mle[i] += 1
            j +=1
        else:
            test_wrong_pred_mle[i][test_nb_mle[j]-1] += 1
            j = j + 1
        if(j > len(test_label_columns)-1):
            break
    k += test_count[i]    
#print(test_class_count_mle)
#print(test_wrong_pred_mle)    


print(Fore.RED + "2. Results based on Bayesian Estimator", end='\n\n')
print(Fore.BLACK + "2.1) Training Data on Bayesian Estimator", end='\n\n')
print("Overall Accuracy =", end = " ")
print(train_count_be/train_num_docs)
print("Class Accuracy: ")
for i in range(0,20):
    print("Group " + str(i+1) + " : ",end="")
    print(train_class_count_be[i]/train_count[i])
print('\n')
print("Confusion Matrix:")
for i in range(0,20):
    for j in range(0,20):
        if(i==j):
            print(train_class_count_be[i], end="   ")
            continue
        if(j==19):
            print(train_wrong_pred_be[i][j])
        else:
            print(train_wrong_pred_be[i][j], end="   ")
print('\n')

print("2.2) Test Data on Bayesian Estimator")
print('\n')
print("Overall Accuracy =", end = " ")
print(test_count_be/test_num_docs)
print("Class Accuracy: ")
for i in range(0,20):
    print("Group " + str(i+1) + " : ",end="")
    print(test_class_count_be[i]/test_count[i])
print('\n')
print("Confusion Matrix:")
for i in range(0,20):
    for j in range(0,20):
        if(i==j):
            print(test_class_count_be[i], end="   ")
            continue
        if(j==19):
            print(test_wrong_pred_be[i][j])
        else:
            print(test_wrong_pred_be[i][j], end="   ")
print('\n')
    
print(Fore.RED + "3. Results based on Maximum Likelihood Estimator", end='\n\n')
print(Fore.BLACK + "3.1) Training Data on Maximum Likelihood Estimator")
print('\n')
print("Overall Accuracy =", end = " ")
print(train_count_mle/train_num_docs)
print("Class Accuracy: ")
for i in range(0,20):
    print("Group " + str(i+1) + " : ",end="")
    print(train_class_count_mle[i]/train_count[i])
print('\n')
print("Confusion Matrix:")
for i in range(0,20):
    for j in range(0,20):
        if(i==j):
            print(train_class_count_mle[i], end="   ")
            continue
        if(j==19):
            print(train_wrong_pred_mle[i][j])
        else:
            print(train_wrong_pred_mle[i][j], end="   ")
print('\n')

print("3.2) Test Data on Maximum Likelihood Estimator")
print('\n')
print("Overall Accuracy =", end = " ")
print(test_count_mle/test_num_docs)
print("Class Accuracy: ")
for i in range(0,20):
    print("Group " + str(i+1) + " : ",end="")
    print(test_class_count_mle[i]/test_count[i])
print('\n')
print("Confusion Matrix:")
for i in range(0,20):
    for j in range(0,20):
        if(i==j):
            print(test_class_count_mle[i], end="   ")
            continue
        if(j==19):
            print(test_wrong_pred_mle[i][j])
        else:
            print(test_wrong_pred_mle[i][j], end="   ")
print('\n')

