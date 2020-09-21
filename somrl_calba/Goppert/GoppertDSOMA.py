import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from SOM import *
from DSOMA import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from goppert import *
from math import exp
from scipy.stats import invgauss
import matplotlib.pyplot as plt
import random


####Création donnée############################################################  
Modele = "DSOMA"
Init = "Alea"
TypeDonnee = "Grille"
mu = np.array([0.0, 0.0])
sigma = np.array([.3, .3])
titre = "DSOMA_Goppert"
if (TypeDonnee == "Grille"):
	xD, yD = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
	larD = len(xD)
	lonD = len(xD[0])
	sizeD = larD * lonD
	xy = np.column_stack([xD.flat, yD.flat])
elif(TypeDonnee == "AleaGaussien"):
	xD = np.random.normal(mu[0], sigma[0], 30**2)
	yD = np.random.normal(mu[0], sigma[0], 30**2)
	sizeD = 30**2
	xy = np.column_stack([xD.flat, yD.flat])


covariance = np.diag(sigma**2)
zD = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
zD = zD.reshape(xD.shape)

xD = np.reshape(xD,sizeD)
yD = np.reshape(yD,sizeD)
zD = np.reshape(zD,sizeD)
DataD = np.array([(xD[i],yD[i],zD[i]) for i in range(sizeD)])
fig = plt.figure()
fig.suptitle(titre)


ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(xD, yD, zD)
ax.set_xlim3d(-1.1, 1.1)
ax.set_ylim3d(-1.1, 1.1)
ax.set_zlim3d(0,1.5)
ax.set_title("Donnée d'apprentissage \n"+TypeDonnee)

#####Apprentissage############################################################ 
lon=5
lar=lon
size = lar * lon 
dim = 3
T = 10000
print("Apprentissage")
nb = 1
Tabm = [DSOMA(-1, dim, lon, lar, eta=lon, epsilon=0.1, tFinal =T) for i in range(nb)]
for i in range(nb):
	donne = Donnee()
	donne.Data		= DataD
	donne.type	 	= "Test"
	donne.N			= sizeD
	donne.limHaute 	= 0.01
	donne.limBasse 	= -0.01
	donne.dim 		= 3

	Tabm[i].Donnee = donne

	Tabm[i].weights = Tabm[i].Donnee.initNeurones(lon*lar,dim)

	for j in range(0,T):
		X = Tabm[i].Donnee.getVecteurEntre()
		Tabm[i].epochs(j, X)

	X	=	Tabm[i].weights.T[0]
	Y	=	Tabm[i].weights.T[1]
	Z   =	Tabm[i].weights.T[2]
	shape = ((lon,lar))	
	X = np.reshape(X,shape)		
	Y = np.reshape(Y,shape)	
	Z = np.reshape(Z,shape)	
	ax = fig.add_subplot(2, 2, 2, projection='3d')
	ax.plot_wireframe(X=X, Y=Y, Z=Z)
	ax.set_xlim3d(-1.1, 1.1)
	ax.set_ylim3d(-1.1, 1.1)
	ax.set_zlim3d(0,1.5)
	ax.set_title("Resultat Apprentissage \ndu premier "+Modele)		
	texte = "Distortion\n%.5f" % Tabm[i].calculeDistortion()		
	ax.text(-1, -1, 1, s=texte, fontsize=10)
	
	def voisinageGrille2D(lar,lon):
		size = lar*lon
		return np.array([np.array([[k-1 if (k-1>=0 and (k)%lar!=0) else -1 , k+1 if (k+1<size and (k+1)%lar!=0) else -1],[k-lar if k-lar>=0 else -1, k+lar if(k+lar<size) else -1]]) for k in range (size)])
					
	#######Application goppert############################################################ 
	print("Application goppert")
	V= voisinageGrille2D(lar,lon)
	tabgop = [goppert(Tabm[i].weights,V) for i in range(nb)]
	X=[]
	Y=[]
	Z=[]

	for xi in range (-20,21):
		print(xi+20)
		for yi in range (-20,21):
			C = np.array([xi*0.05,yi*0.05])
			X.append(C[0])
			Y.append(C[1])
			sum = 0
			for i in range(nb):
				sum += tabgop[i].YCFSOM(C)
			Z.append(sum/nb)
							
	shape = (int(math.sqrt(len(X))),int(math.sqrt(len(X))))	
	X = np.reshape(X,shape)		
	Y = np.reshape(Y,shape)	
	Z = np.reshape(Z,shape)	

	# Normalize to [0,1]
	norm = plt.Normalize(Z.min(), Z.max())
	colors = cm.viridis(norm(Z))
	rcount, ccount, _ = colors.shape

	ax = fig.add_subplot(2, 2, 3, projection='3d')
	where_are_NaNs = isnan(Z)
	Z[where_are_NaNs] = 0
	surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
				facecolors=colors, shade=False)
	surf.set_facecolor((0,0,0,0))
	ax.set_xlim3d(-1.1, 1.1)
	ax.set_ylim3d(-1.1, 1.1)
	ax.set_zlim3d(0,1.5)
	ax.set_title("ResultatGoppert")

	########Comparaison fonction  et goppert####################################################################################
	xy = np.column_stack([X.flat, Y.flat])
	ZT = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	ZT = ZT.reshape(shape)
	ZREP = abs(ZT-Z)	
	where_are_NaNs = isnan(ZREP)
	ZREP[where_are_NaNs] = 0
	norm = plt.Normalize(ZREP.min(), ZREP.max())
	colors = cm.viridis(norm(ZREP))
	rcount, ccount, _ = colors.shape
	ax = fig.add_subplot(2, 2, 4, projection='3d')
	surf = ax.plot_surface(X, Y, ZREP, rcount=rcount, ccount=ccount,
		facecolors=colors, shade=False)
	#ax.scatter(X, Y, ZREP)
	ax.set_xlim3d(-1.1, 1.1)
	ax.set_ylim3d(-1.1, 1.1)
	ax.set_zlim3d(0,1.5)
	ax.set_title("Différence")
	#Box avec le nombre de frame
	textstr = '\nMoy %f' % (np.mean(ZREP)) 
	#print(textstr)
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(-1, -1, 1, s=textstr, fontsize=10)

	plt.savefig("TEST"+titre)
	#plt.show()






























