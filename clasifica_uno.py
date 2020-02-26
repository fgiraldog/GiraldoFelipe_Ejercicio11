import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import sklearn.metrics as cf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def PCA(x):
	cov = np.cov(x.T)
	valores, vectores = np.linalg.eig(cov)
	valores = np.real(valores)
	vectores = np.real(vectores)
	ii = np.argsort(-valores)
	valores = valores[ii]
	vectores = vectores[:,ii]

	return valores, vectores

valores_PCA, vectores_PCA = PCA(x_train)

def input_scikit(vectores,x,n):
	proyect = np.dot(vectores[:,0],x.T)
	for i in range(1,n):
		proyect = np.vstack([proyect,np.dot(vectores[:,i],x.T)])

	return proyect.T

def F1_score(vectores,x,y,n,one):
	F1_score = []
	components = []
	pepito = LinearDiscriminantAnalysis()
	for i in range(3,n):
		pepito.fit(input_scikit(vectores,x,i),y)
		F1_score.append(cf.f1_score(pepito.predict(input_scikit(vectores,x,i)),y, pos_label=one))
		components.append(i)

	return components, F1_score

y_train[np.where(y_train != 1)] = 0

y_test[np.where(y_test != 1)] = 0



plt.subplot(121)
plt.scatter(F1_score(vectores_PCA,x_train,y_train,31,1)[0],F1_score(vectores_PCA,x_train,y_train,31,1)[1], label = 'Train')
plt.scatter(F1_score(vectores_PCA,x_test,y_test,31,1)[0],F1_score(vectores_PCA,x_test,y_test,31,1)[1], label = 'Test')
plt.xlabel('# Components')
plt.ylabel('F1 Score')
plt.title('F1 Score Ones')
plt.legend()
plt.subplot(122)
plt.scatter(F1_score(vectores_PCA,x_train,y_train,31,0)[0],F1_score(vectores_PCA,x_train,y_train,31,0)[1], label = 'Train')
plt.scatter(F1_score(vectores_PCA,x_test,y_test,31,0)[0],F1_score(vectores_PCA,x_test,y_test,31,0)[1], label = 'Test')
plt.legend()
plt.xlabel('# Components')
plt.ylabel('F1 Score')
plt.title('F1 Score Others')
plt.subplots_adjust(wspace=0.5)
plt.savefig('F1_score_LinearDiscriminantAnalysis.png')

