import pylab as pl
from numpy import genfromtxt
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import scipy.io

data = scipy.io.loadmat('dataku.mat')["dataimage_plus"]

print(data)

#data = genfromtxt('dataimage.csv', delimiter=',')
n_samples = len(data)
#split data input and data output
data_i = data[:,2:4] #2:4 => fft (mean dan standard deviasi FFT), 0:2 => (entropy dan energi GLCM)

print data_i
data_o = data[:,4]

#split 50 % data (data training)
data_i_train = data_i[0:][::2]
data_o_train = data_o[0:][::2]

#split 50 % data (data testing)
data_i_test = data_i[1:][::2]
data_o_test = data_o[1:][::2]

#create knn classifier
neigh = KNeighborsClassifier(n_neighbors=3)

#We learn the digits on the first half of the digits
neigh.fit(data_i_train, data_o_train)

# Now predict the value of the digit on the second half:
expected = data_o_test
predicted = neigh.predict(data_i_test)

print predicted
print expected

print("Classification report for classifier %s:\n%s\n"
      % (neigh, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
