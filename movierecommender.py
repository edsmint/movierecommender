import csv
import numpy as np
import scipy as sc
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import seaborn as s 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

data = []
with open('u.data') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t')
	for row in spamreader:
		data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data) # num_observations = 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

def a(train):
	movie_score_sum = {} #maps movie id to tuple of sum of scores and number of users that rated it
	movie_avg_ratings = {} #maps movie id to avg score given by all users
	for i in range(len(train)):
		line = train[i]
		movie = line[1]
		score = line[2]
		if movie in movie_score_sum:
			#update the (score_sum, num) tuple
			new_score = movie_score_sum[movie][0] + score
			new_num = movie_score_sum[movie][1] + 1
			movie_score_sum[movie] = (new_score, new_num)
		else:
			#add movie, score, and 1
			movie_score_sum[movie] = (score, 1)

	for movie in movie_score_sum:
		tup = movie_score_sum[movie]
		movie_avg_ratings[movie] = 1.0 * tup[0] / tup[1]

	return movie_avg_ratings

def calc_error(mu, test):
	squared_error_sum = 0
	for i in range(len(test)):
		line = test[i]
		movie = line[1]
		score = line[2]
		diff_squared = 0
		if(movie in mu):
			diff_squared = (mu[movie] - score) ** 2
		squared_error_sum = squared_error_sum + diff_squared

	error = squared_error_sum / len(test)
	return error

#given input matrix, d, and training data, generates a new model that predicts each spot in matrix
def b(train, d, Matrix):
	for i in range(len(train)):
		line = train[i]
		user = line[0]
		movie = line[1]
		score = line[2]
		Matrix[user][movie] = score 
		
	u, s, vt = sc.sparse.linalg.svds(Matrix, d)
	#This model is a n,m matrix with predictions and we will use it to compute train and test error by going spot by spot
	model_matrix = u @ np.diag(s) @ vt 
	return model_matrix

def calc_error_b(data, model):
	squared_error_sum = 0
	for i in range(len(data)):
		line = data[i]
		user = line[0]
		movie = line[1]
		score = line[2]
		diff_squared = (model[user][movie] - score) ** 2
		squared_error_sum = squared_error_sum + diff_squared

	error = squared_error_sum / len(data)
	return error

def fill_matrix(mu):
	m = 1682 #movies 
	n = 943 #users
	Matrix = np.zeros((n,m))
	#Go through Matrix and update spot based on prediction from mu
	for i in range(n):
		for j in range(m):
			if(j in mu):
				Matrix[i][j] = mu[j]
	return Matrix

def alternating_minimization(train, d):
	sigma = np.sqrt(3/d)
	lam = 10
	m = 1682 #movies 
	n = 943 #users
	identity = np.identity(d)*lam
	u = np.zeros((n,d)) #user vector
	v = np.zeros((d,m)) #movie vector
	#populate vectors with random values
	for i in range(n):
		for j in range(d):
			u[i][j] = np.random.rand() * sigma
	for i in range(d):
		for j in range(m):
			v[i][j] = np.random.rand() * sigma

	#alternate minimization until convergence	
	for num in range(120):
		#update user
		for i in range(n):
			left = np.zeros((d,d))
			right = np.zeros((d,))
			relevant = train[train[:,0] == i] #returns all rows for this user in training set
			movie_vec = relevant[:,1] # 1x?	all movies in training set that user rated
			score_vec = relevant[:,2] #the scores that each user gave
			y = v[:, movie_vec]
			right = score_vec.T @ y.T
			left = y @ y.T + identity
			u[i] = np.linalg.solve(left, right)
			#Alternative method that failed and gave worse results
			#for j in range(len(relevant)):
			#	movie_vec = v[:,relevant[j][1]]
			#	right_update = relevant[j][2] * movie_vec #score*dx1 vector for the correct movie
			#	right = right + right_update
			#	meat = np.matmul(movie_vec.reshape(d,1), (movie_vec.T).reshape(1,d))
			#	left_update = meat + identity
			#	left = left + left_update
			#inverse = np.linalg.inv(left)
			#u[i] = np.matmul(inverse, right)
			#try:
			#	u[i] = np.linalg.solve(left, right)
			#except np.linalg.LinAlgError:
			#	u[i] = np.matmul(left, right)

		#update movie
		for i in range(m):
			left = np.zeros((d,d))
			right = np.zeros((d,))
			relevant = train[train[:,1] == i] #returns all rows for this user in training set
			user_vec = relevant[:,0] # 1x?	all movies in training set that user rated
			score_vec = relevant[:,2] #the scores that each user gave
			y = v[:, user_vec]
			right = score_vec.T @ y.T
			left = y @ y.T + identity
			v[:,i] = np.linalg.solve(left, right)
			#Alternative method that failed and gave worse results
			#left = np.zeros((d,d))
			#right = np.zeros((d,))
			#relevant = train[train[:,1] == i] #returns all rows for this movie in training set
			#for j in range(len(relevant)):
				#user_vec = u[relevant[j][0]]
				#right_update = relevant[j][2] * user_vec #score*dx1 vector for the correct user
				#right = right + right_update
				#meat = np.matmul(user_vec.reshape(d,1), (user_vec.T).reshape(1,d))
				#left_update = meat + identity
				#left = left + left_update
			#inverse = np.linalg.inv(left)
			#v[:,i] = np.matmul(inverse, right)	
			#try: 
			#	v[:,i] = np.linalg.solve(left, right) 
			#except np.linalg.LinAlgError:
			#	v[:,i] = np.matmul(left, right)		

		model = np.matmul(u,v)
		return model

def e(d):
	sigma = np.sqrt(3/d)
	lam = 10
	step = 0.01
	batch_size = 50

	model = nn.Linear(1,1)
	optimizer = optim.SGD(model.parameters(), lr=step)
	#for 100 times
		#grab subset of data
		#optimizer.zero_grad()
		#calculate loss from batch size
		loss_fn.backward()
		optimizer.step()

	u = np.zeros((n,d)) #user vector
	v = np.zeros((d,m)) #movie vector
	#populate vectors with random values
	for i in range(n):
		for j in range(d):
			u[i][j] = np.random.rand() * sigma
	for i in range(d):
		for j in range(m):
			v[i][j] = np.random.rand() * sigma



	#run until convergence
	for num in range(100):
		#update user 
		for i in range(n):
			relevant = train[train[:,0] == i] #rows in training data that have this user
			if(len(relevant) > batch_size):
				batch = relevant[batch_size]
			else:
				batch = relevant



		#update movie
		for i in range(m):
			relevant = train[train[:,1] == i] #rows in training data that have this movie
			if(len(relevant) > batch_size):
				batch = relevant[batch_size]
			else:
				batch = relevant





#2a
mu = a(train) #has each movie and average score from train data
test_error = calc_error(mu, test) #test error for part a
print(test_error)

#2b
m = 1682 #movies 
n = 943 #users
Matrix = np.zeros((n,m)) #sparse matrix that will hold all combos of users and movies afer we populate it using train data
d = [1, 2, 5, 10, 20, 50]
train_errors = [0 for x in range(6)]
test_errors = [0 for x in range(6)]
for i in range(len(d)):
	model = b(train, d[i], Matrix)
	train_errors[i] = calc_error_b(train, model)
	test_errors[i] = calc_error_b(test, model)
s.set()
train_plot, = plt.plot(d, train_errors)
test_plot, = plt.plot(d, test_errors)
plt.legend((train_plot, test_plot), ("Train Error", "Test Error"))
plt.xlabel("d")
plt.ylabel("Error")
plt.show()

#2c
#generate the matrix filled with estimates using mu from part a
matrix = fill_matrix(mu)
train_errors_c = [0 for x in range(6)]
test_errors_c = [0 for x in range(6)]
for i in range(len(d)):
	model = b(train, d[i], matrix)
	train_errors_c[i] = calc_error_b(train, model)
	test_errors_c[i] = calc_error_b(test, model)
s.set()
train_plot, = plt.plot(d, train_errors_c)
test_plot, = plt.plot(d, test_errors_c)
plt.legend((train_plot, test_plot), ("Train Error", "Test Error"))
plt.xlabel("d")
plt.ylabel("Error")
plt.show()

#2d
#Populate two 1xd vectors u and v - np.random.rand() * sigma
#Alternating minimization until they converge
#For all training data - calculate error (ui * vj - Rij)^2 and sum it up
train_errors_d = [0 for x in range(6)]
test_errors_d = [0 for x in range(6)]
for i in range(len(d)):
	model = alternating_minimization(train, d[i])
	train_errors_d[i] = calc_error_b(train, model)
	test_errors_d[i] = calc_error_b(test, model)
s.set()
train_plot, = plt.plot(d, train_errors_d)
test_plot, = plt.plot(d, test_errors_d)
plt.legend((train_plot, test_plot), ("Train Error", "Test Error"))
plt.xlabel("d")
plt.ylabel("Error")
plt.title("Mean Squared Error vs. d")
plt.show()

#2e
#Batched stochastic gradient descent
train_errors_e = [0 for x in range(6)]
test_errors_e = [0 for x in range(6)]
for i in range(len(d)):
	model = e(d[i])
	train_errors_e[i] = calc_error_b(train, model)
	test_errors_e[i] = calc_error_b(test, model)
s.set()
train_plot, = plt.plot(d, train_errors_e)
test_plot, = plt.plot(d, test_errors_e)
plt.legend((train_plot, test_plot), ("Train Error", "Test Error"))
plt.xlabel("d")
plt.ylabel("Error")
plt.title("Mean Squared Error vs. d")
plt.show()























