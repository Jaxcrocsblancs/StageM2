import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from SOM import *
from DSOM import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from goppert import *
import pylab
from matplotlib import pyplot as plt
####Création donnée 
xD, yD = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
xy = np.column_stack([xD.flat, yD.flat])
mu = np.array([0.0, 0.0])
sigma = np.array([.3, .3])
covariance = np.diag(sigma**2)
zD = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
zD = zD.reshape(xD.shape)
larD = len(xD)
lonD = len(xD[0])
sizeD = larD * lonD
xD = np.reshape(xD,sizeD)
yD = np.reshape(yD,sizeD)
zD = np.reshape(zD,sizeD)
DataD = np.array([(xD[i],yD[i],zD[i]) for i in range(sizeD)])

def voisinageGrille2D(lar,lon):
	size = lar*lon
	return np.array([np.array([[k-1 if (k-1>=0 and (k)%lar!=0) else -1 , k+1 if (k+1<size and (k+1)%lar!=0) else -1],[k-lar if k-lar>=0 else -1, k+lar if(k+lar<size) else -1]]) for k in range (size)])
					
NN= 15
dataDSOM = [ [] for i in range(2,NN)]
#####Apprentissage 
for l in range (2,NN):
	print("Longueur ",l)
	lon=l
	lar=lon
	V= voisinageGrille2D(lar,lon)
	size = lar * lon 
	dim = 3
	T = 10000
	nb = 10
	TabmDSOM = [DSOM(-1, dim, lon, lar, eta=l, epsilon=0.1, tFinal =T) for i in range(nb)]			
	
	for i in range(nb):
		print("Calcul modéle ",i," long ",l)
		donne = Donnee()

		donne.Data		= DataD
		donne.type	 	= "Test"
		donne.N			= sizeD
		donne.limHaute 	= 1
		donne.limBasse 	= -1
		donne.dim 		= 3

		TabmDSOM[i].Donnee = donne

		#Carte initialisé au hazard
		TabmDSOM[i].weights = TabmDSOM[i].Donnee.initNeurones(lon*lar,dim)


		for j in range(0,(T+1)):
			distBMU=0.0
			distortion=1.0
			while (distBMU<distortion):
				X = TabmDSOM[i].Donnee.getVecteurEntre()
				distBMU = TabmDSOM[i].distBMU(X)
				distortion = TabmDSOM[i].calculeDistortion()
			TabmDSOM[i].epochs(j, X)

		#######Application goppert############################################################ 
		gop = goppert(TabmDSOM[i].weights,V)
		X=[]
		Y=[]
		Z=[]
		tgrille = 40
		for xi in range (-20,21):
			for yi in range (-20,21):
				C = np.array([xi*0.05,yi*0.05])
				X.append(C[0])
				Y.append(C[1])
				Z.append(gop.YCFSOM(C))
		shape = (int(math.sqrt(len(X))),int(math.sqrt(len(X))))	
		X = np.reshape(X,shape)		
		Y = np.reshape(Y,shape)	
		Z = np.reshape(Z,shape)	
		where_are_NaNs = isnan(Z)
		Z[where_are_NaNs] = 0
		########Comparaison fonction  et goppert####################################################################################
		xy = np.column_stack([X.flat, Y.flat])
		ZT = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
		ZT = ZT.reshape(shape)
		ZREP = abs(ZT-Z)	
		where_are_NaNs = isnan(ZREP)
		ZREP[where_are_NaNs] = 0			
		dataDSOM[l-2].append(np.mean(ZREP))

BoxName = [""]
dataDSOM = np.array(dataDSOM).T
print(dataDSOM)
for i in range (2,NN):
		BoxName.append(i)
fig = plt.figure()
bp1 = plt.boxplot(dataDSOM)
for box in bp1['boxes']:
	box.set(color='blue', linewidth=2)
plt.ylim(0,1)
tick = [ i for i in range(0,NN)]
pylab.xticks(tick, BoxName)
plt.title("Evolution goppert DSOM T= 10000 \nChoix du X selon distortion")
plt.savefig('GopDSOMTEST.png' )

















