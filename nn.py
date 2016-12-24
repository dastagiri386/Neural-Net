import sys
import math
import scipy.io.arff as arff
import numpy
import random
from random import shuffle
import math

lr = float(sys.argv[1])
h = int(sys.argv[2])
e = int(sys.argv[3])
train_file = sys.argv[4]
test_file = sys.argv[5]

train_data = []
train_class = []
train_labels = []
train_features = []
feature_vec_length = 0
#---------------------------------------------------- Read and transform the training data --------------------------------
data, meta = arff.loadarff(train_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]))
	train_data.append(l)
	
mstr = str(meta)
mstr = mstr.split("\n\t")
for im in mstr:
	if not im.startswith("class") and "type is numeric" in im:
		train_features.append([im.split('\'s')[0], None])
		feature_vec_length += 1
	elif not im.startswith("class") and "type is nominal" in im:
		feature = im.split('\'s')[0]
		attr = im.split("(")[1].split(")")[0].replace(" ","").replace("'","").split(",")
		train_features.append([feature, attr])
		feature_vec_length += 1
	elif im.startswith("class"):
		flag = 0
		train_labels = im.split("(")[1].split(")")[0].replace(" ","").replace("'","").split(",")
		train_class = ['class', train_labels]

#Standardize numeric features (if any)
indices = []
for i in range(len(train_features)):
	if train_features[i][1] == None:
		indices.append(i)

if len(indices) > 0:
	for t in indices:
		vals = []
		for item in train_data:
			vals.append(float(item[t]))
		mean = numpy.mean(vals)
		stddev = numpy.std(vals)
		for item in train_data:
			item[t] = (float(item[t]) - mean)/stddev

#Encode discrete features using one-of-k encoding
for i in range(len(train_features)):
	if train_features[i][1] != None: # feature is nominal
		l = train_features[i][1]
		d = {}
		
		for j in range(len(l)):
			code = []
			for k in range(len(l)):
				code.append('0')
			code[j] = '1'
			d[l[j]] = ' '.join(code)
		train_features[i][1] = d
#Encode class labels as 0 for first class and 1 for the second class (assuming binary classfication)
l = train_class[1]
d = {}
d[l[0]] = '0'
d[l[1]] = '1'
train_class[1] = d
train_labels = d

shuffle(train_data)

encoded_train_data = []
for item in train_data:
	for i in range(len(item)-1): #encode all feature values
		if isinstance(item[i], str):
			item[i] = train_features[i][1][item[i]]
	item[len(item)-1] = train_labels[item[len(item)-1]]
	encoded_train_data.append(item)
	
act_train_data = []
for item in encoded_train_data:
	act = [1] #activation
	for i in range(len(item)):
		if isinstance(item[i], str):
			l = item[i].split(' ')
			for j in range(len(l)):
				l[j] = int(l[j])
			act += l
		else:
			act.append(item[i])
	act_train_data.append(act)
#------------------------------------------ Test data -------------------------------	
test_data = []
data, meta = arff.loadarff(test_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]))
 	test_data.append(l)
 	
if len(indices) > 0:
	for t in indices:
		vals = []
		for item in test_data:
			vals.append(float(item[t]))
		mean = numpy.mean(vals)
		stddev = numpy.std(vals)
		for item in test_data:
			item[t] = (float(item[t]) - mean)/stddev
	
encoded_test_data = []
for item in test_data:
	for i in range(len(item)-1): #encode all feature values
		if isinstance(item[i], str):
			item[i] = train_features[i][1][item[i]]
	item[len(item)-1] = train_labels[item[len(item)-1]]
	encoded_test_data.append(item)

act_test_data = []
for item in encoded_test_data:
	act = [1] #activation
	for i in range(len(item)):
		if isinstance(item[i], str):
			l = item[i].split(' ')
			for j in range(len(l)):
				l[j] = int(l[j])
			act += l
		else:
			act.append(item[i])
	act_test_data.append(act)
#-------------------------------------------------------------------------------------

act_input_length = len(act_train_data[0]) - 1
	
input_wts = []
for i in range(h): #for hidden units, list of list of input weights (one list for each hidden unit)
	l = []
	for j in range(act_input_length):
		l.append(random.uniform(-0.01, 0.01))
	input_wts.append(l)
	
if h == 0: #for no hidden units, list of input weights
	for j in range(act_input_length):
		input_wts.append(random.uniform(-0.01, 0.01))
		
if h > 0:
	act_hidden = [1]
	for i in range(h):
		act_hidden.append(0)
	hidden_wts = []
	for i in range(h+1):
		hidden_wts.append(random.uniform(-0.01, 0.01))
		
if h > 0:
	for i in range(e):
		error = 0
		for j in range(len(act_train_data)): #for each training instance
			item = act_train_data[j]
			for k in range(h):
				sum = 0
				for m in range(act_input_length):
					sum += input_wts[k][m] * item[m]
				act_hidden[k+1] = 1/(1+math.exp(-1*sum)) # activation for the kth hidden unit
			
			sum = 0
			for k in range(h+1):
				sum += hidden_wts[k]*act_hidden[k]
			out = 1/(1+math.exp(-1*sum)) # output 
			
			y = item[act_input_length]
			error += -1*y*math.log(out) - (1-y)*math.log(1-out) # cross-entropy error
			
			#Update the weights for the next training instance
			for k in range(h+1):
				#print l, act_hidden[k], (y-out)
				hidden_wts[k] += lr * act_hidden[k] * (y - out) # only one output unit
				
			for k in range(h):
				for l in range(act_input_length):
					delta = act_hidden[k+1] * (1 - act_hidden[k+1]) * (y - out) * hidden_wts[k+1]
					#print lr, act_train_data[l], delta
					input_wts[k][l] += lr * act_train_data[j][l] * delta
		
		#After each epoch, get number of correctly classified and misclassified instances
		count_cor = 0
		count_icor = 0
		for j in range(len(act_train_data)):
			item = act_train_data[j]
			for k in range(h):
				sum = 0
				for m in range(act_input_length):
					sum += input_wts[k][m] * item[m]
				act_hidden[k+1] = 1/(1+math.exp(-1*sum)) # activation for the kth hidden unit
				
			sum = 0
			for k in range(h+1):
				sum += hidden_wts[k]*act_hidden[k]
			out = 1/(1+math.exp(-1*sum))
			#print "out", out
			
			if out > 0.5:
				out_val = 1
			else:
				out_val = 0
			
			if out_val == act_train_data[j][act_input_length]:
				count_cor += 1
			else:
				count_icor += 1
		print i+1,"\t",error, "\t", count_cor, "\t", count_icor
		
	#print "Predictions for the test instances : "
	count_cor = 0
	count_icor = 0
	for j in range(len(act_test_data)):
		item = act_train_data[j]
		for k in range(h):
			sum = 0
			for m in range(act_input_length):
				sum += input_wts[k][m] * item[m]
			act_hidden[k+1] = 1/(1+math.exp(-1*sum)) # activation for the kth hidden unit
				
		sum = 0
		for k in range(h+1):
			sum += hidden_wts[k]*act_hidden[k]
		out = 1/(1+math.exp(-1*sum))
			
		if out > 0.5:
			out_val = 1
		else:
			out_val = 0
		print out, "\t", out_val, "\t", act_train_data[j][act_input_length]
			
		if out_val == act_train_data[j][act_input_length]:
			count_cor += 1
		else:
			count_icor += 1
	print count_cor, "\t", count_icor
			
		
elif h == 0:
	for i in range(e):
		error = 0
		for j in range(len(act_train_data)):
			sum = 0
			for k in range(act_input_length):
				sum += input_wts[k]*act_train_data[j][k]
			out = 1/(1+math.exp(-1*sum))
			y = act_train_data[j][act_input_length]
			
			for k in range(act_input_length):
				input_wts[k] += lr * act_train_data[j][k] *(y - out)
			error += -1*y*math.log(out) - (1-y)*math.log(1-out) # cross-entropy error
				
		count_cor = 0
		count_icor = 0
		for j in range(len(act_train_data)):
			sum = 0
			for k in range(act_input_length):
				sum += input_wts[k]*act_train_data[j][k]
			out = 1/(1+math.exp(-1*sum))
			
			if out > 0.5:
				out_val = 1
			else:
				out_val = 0
			
			if out_val == act_train_data[j][act_input_length]:
				count_cor += 1
			else:
				count_icor += 1
		print i+1,"\t",error, "\t", count_cor, "\t", count_icor
	
	count_cor = 0
	count_icor = 0
	for j in range(len(act_test_data)):
		sum = 0
		for k in range(act_input_length):
			sum += input_wts[k]*act_train_data[j][k]
		out = 1/(1+math.exp(-1*sum))
			
		if out > 0.5:
			out_val = 1
		else:
			out_val = 0
		print out, "\t", out_val, "\t", act_train_data[j][act_input_length]
			
		if out_val == act_train_data[j][act_input_length]:
			count_cor += 1
		else:
			count_icor += 1
	print count_cor, "\t", count_icor
		
			
				
				
		
	

			




	


	
		
			
	

