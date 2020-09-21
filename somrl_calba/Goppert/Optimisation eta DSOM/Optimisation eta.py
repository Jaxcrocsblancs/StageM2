import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from SOM import *
from DSOM import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from goppert import *
import pylab

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
from matplotlib import pyplot as plt


#####Apprentissage 
for l in range (25,30):
	lon=l
	lar=lon
	size = lar * lon 
	dim = 3
	T = 10000
	nb = 10
	for n in range (1,50):
		TabmDSOM = [DSOM(-1, dim, lon, lar, eta=n, epsilon=0.1, tFinal =T) for i in range(nb)]			
		dataDSOM = [ [] for i in range(nb)]
		for i in range(nb):
			print("Calcul modéle ",i)
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
				X = TabmDSOM[i].Donnee.getVecteurEntre()
				TabmDSOM[i].epochs(j, X)
				if(j%(T/10)==0 and j != 0):
					print(j)
					dataDSOM[i].append(TabmDSOM[i].calculeDistortion())
					
		BoxName = [""]
		#data = np.array(data)
		dataDSOM = np.array(dataDSOM)
		for i in range (10):
				BoxName.append("%d" % (T/10*(i+1)))
		fig = plt.figure()
		bp1 = plt.boxplot(dataDSOM)
		for box in bp1['boxes']:
			box.set(color='blue', linewidth=2)
		plt.ylim(0,1)
		tick = [ i for i in range(10)]
		pylab.xticks(tick, BoxName)
		plt.title("Evolution distortion DSOM\n grille = %d selon eta = %d"  % (l,n))
		plt.savefig('G%d_%d.png' % (l,n))
		

	#plt.show()

















