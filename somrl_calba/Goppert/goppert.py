import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib import cm
import math
from scipy.spatial import distance

class goppert():

	def __init__(self, supp, V):
		#Ensemble des points support, sous la forme [input,output]
		self.supp = supp
		#Nombre de points support
		self.N = len(supp)
		#Lien de voisinage entre les points support
		self.V = V
		#Calcul du R moyenne distance points support *2
		self.calcR()
	
	#Calcul du R moyenne distance points support *2
	def calcR(self):
		dist = 0
		cpt=0
		for i in range(self.N):
				for k in range(len(self.V[0])):
					for l in range(2):
						if(self.V[i][k][l]!=-1):
							dist += np.linalg.norm(self.supp[i,0:-1] - self.supp[self.V[i][k][l],0:-1])
				cpt+=1
		self.R = dist/cpt
	
	#Calcul sur input selon equation 7	
	def linid(self,X,i,d):
		dim = self.V[i][d][0]
		dip = self.V[i][d][1]
		di 	= i 
		Dj  = np.array([dim,dip])
		if(dim==-1):
			Dj = np.array([dip])
		if(dip==-1):
			Dj = np.array([dim])
		Phim = self.PHIiSF(X,dim,Dj)
		Phip = self.PHIiSF(X,dip,Dj)
		return Phip*(self.supp[dip,d]-self.supp[di,d])-Phim*(self.supp[dim,d]-self.supp[di,d])
		
	#Calcul sur output selon equation 7	
	def louid(self,X,i,d):
		dim = self.V[i][d][0]
		dip = self.V[i][d][1]
		di 	= i 
		Dj  = np.array([dim,dip])
		if(dim==-1):
			Dj = np.array([dip])
		if(dip==-1):
			Dj = np.array([dim])
		Phim = self.PHIiSF(X,dim,Dj)
		Phip = self.PHIiSF(X,dip,Dj)

		return Phip*(self.supp[dip,-1]-self.supp[di,-1])-Phim*(self.supp[dim,-1]-self.supp[di,-1])

	#Retourne la valeur YiI-SOM pour un neurones i donnee
	def YiISOMXD(self,X,i):
		#Si X est un point support
		if (X in self.supp[:,0:-1]):
			#Si i est le point support alors retour Sortie du support sinon  peut importe
			#le PHIiSFXD sera egal a 0 pour tout les autres i autre que le support
			return self.supp[i,-1]
		size = X.size
		#Calcul du tableau Lin
		Lin = np.array([self.linid(X,i,d) for d in range(size)])
		alpha = np.linalg.inv(np.dot(Lin,Lin.T)+0.1*np.eye(size))*Lin.T*(X-self.supp[i,0:-1])
		
		Lou = np.array([self.louid(X,i,d) for d in range(size)])
		#Calcul du Yisom a retourner
		Yisom = self.supp[i,-1]+Lou*alpha[0]
		return Yisom[0]
		
	#Retourne la valeur de PHIiSf selon l'ensemble d'indice Dj
	def PHIiSF(self,X,i,Dj):
		if (i==-1):
			return 0

		tabDj = np.linalg.norm(X-self.supp[:,0:-1],axis = 1)
		if(not(0 in tabDj)): 
			tabDj = tabDj**-2
			sousTabDj = tabDj[Dj.tolist()]
			sumSousTabDj = np.sum(sousTabDj)
			return tabDj[i]/sumSousTabDj
		else:
			#i est le point support
			if (tabDj[i]==0):
				return 1
			else:
				return 0
				
	#Retourne la valeur de PHIiLRSf selon la distance R 	
	def PHIiLRSF(self,X,i):
		tabDj = np.linalg.norm(X-self.supp[:,0:-1],axis = 1)
		
		if(not(0 in tabDj)): 
			tabRmDJp = self.R-tabDj
			tabRmDJp[tabRmDJp<0] = 0 
			tabRxDJ = self.R * tabDj
			
			tabRmDJpdRxDJ = (tabRmDJp/tabRxDJ)**2
			Sum = np.sum(tabRmDJpdRxDJ)
			if(Sum  == 0):
				Dj = np.array([i for i in range(self.N)])
				return self.PHIiSF(X,i,Dj)
			return tabRmDJpdRxDJ[i]/Sum
		else:
			#i est le point support
			if (tabDj[i]==0):
				return 1
			else:
				return 0	
		
	#Retourne la valeur final 	
	def YCFSOM(self,X):  	
		tabYiISOM = np.array([self.YiISOMXD(X,i) for i in range(self.N)])
		tabDj = np.array([i for i in range(self.N)])
		#tabPHIiSF = np.array([self.PHIiSF(X,i,tabDj) for i in range(self.N)])
		tabPHIiSF = np.array([self.PHIiLRSF(X,i) for i in range(self.N)])
		return np.sum(np.multiply(tabPHIiSF,tabYiISOM))


########FONCTION TESTE Pas toutes adapte pour marcher avec la classe######################################################

def test1D():
	fig = plt.figure(facecolor='white')

	supp  = np.array([[0,0],[1,0],[4.5,1],[5.5,-1],[9,0],[10,-3]])
	V = np.array([np.array([[k-1 if k-1>=0 else -1 , k+1 if k+1<len(supp)  else -1]]) for k in range (len(supp))])
	plt.xlim(-3.0, 12.0)
	plt.ylim(-6.0, 6.0)

	ax = fig.add_subplot(111)
	
	gop = goppert(supp,V)
	
	
	x=[]
	y=[]
	for i in range (-300,1200):
		X = np.array(i*0.01)
		x.append(X)
		y.append([gop.YCFSOM(X)])
	plt.plot(x,y)


	ax.scatter(supp[:, 0], supp[:, 1], s=50,marker='X',c="red" )
	plt.show()
	
#Repere droite haut 
def voisinageGrille2D(lar,lon):
	size = lar*lon
	return np.array([np.array([[k-1 if (k-1>=0 and (k)%lar!=0) else -1 , k+1 if (k+1<size and (k+1)%lar!=0) else -1],[k-lar if k-lar>=0 else -1, k+lar if(k+lar<size) else -1]]) for k in range (size)])
	
def testGausienne():
	x_min = -20.0
	x_max = 20.0

	mean = 0.0 
	std = 2.0

	x = np.linspace(x_min, x_max, 10)
	y = scipy.stats.norm.pdf(x,mean,std)
	
	
	
	
	V = np.array([np.array([[k-1 if k-1>=0 else -1 , k+1 if k+1<len(x)  else -1]]) for k in range (len(x))])
	X = np.linspace(x_min, x_max, 1000)
	
	y1=[]
	phi = []
	supp = np.array((x,y)).T
	for i in X:
		y1.append([YCFSOM(supp,np.array([i]),V)])
	fig = plt.figure(facecolor='white')
	plt.plot(x,y)
	plt.savefig("Objectif")

	plt.plot(X,y1)
	plt.savefig("ObtenuLR")
	R =  5
	tabDj = np.array([i for i in range(len(supp))])
	for i in X:
		tabPHIiSF = np.array([PHIiLRSF(supp,i,k,tabDj,V,R) for k in range(len(supp))])
		phi.append(tabPHIiSF)
	fig = plt.figure(facecolor='white')
	print()
	plt.plot(X,phi)
	y = np.array([-0.005 for k in range (len(x))])
	plt.scatter(x, y, s=50,marker='X',c="red" )
	plt.savefig("1DGlobauxPhiLR")
	
	
	
	
	
	
	plt.show()
	
def test2D():
	
	x, y = np.mgrid[-1.0:1.0:6j, -1.0:1.0:6j]
	# Need an (N, 2) array of (x, y) pairs.
	xy = np.column_stack([x.flat, y.flat])

	mu = np.array([0.0, 0.0])

	sigma = np.array([.3, .3])
	covariance = np.diag(sigma**2)
	
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	
	# Reshape back to a (30, 30) grid.
	z = z.reshape(x.shape)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X=x, Y=y, Z=z)
	plt.savefig("GaussienObj6")
	lar = len(x)
	lon = len(x[0])
	size = lar * lon 
	x1 = np.reshape(x,size)
	y1 = np.reshape(y,size)
	z1 = np.reshape(z,size)
	supp = np.array([(x1[i],y1[i],z1[i]) for i in range(size)])
	V = voisinageGrille2D(lar,lon)
	gop = goppert(supp,V)
	X=[]
	Y=[]
	Z=[]

	for xi in range (-20,21):
		print(xi+20)
		for yi in range (-20,21):
			X.append(xi*0.05+0.001)
			Y.append(yi*0.05+0.001)
			C =  np.array([xi*0.05+0.001,yi*0.05+0.001])
			azer= gop.YCFSOM(C)
			Z.append(azer)
			
	shape = (int(math.sqrt(len(X))),int(math.sqrt(len(X))))	
	X = np.reshape(X,shape)		
	Y = np.reshape(Y,shape)	
	Z = np.reshape(Z,shape)	


	# Normalize to [0,1]
	norm = plt.Normalize(Z.min(), Z.max())
	colors = cm.viridis(norm(Z))
	rcount, ccount, _ = colors.shape

	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
						   facecolors=colors, shade=False)
	surf.set_facecolor((0,0,0,0))
	plt.savefig("GaussienLRCalc6")
	plt.show()
				
def AffichagePhi8():
	t = 5
	x, y = np.mgrid[-1.0:1.0:5j, -1.0:1.0:5j]
	# Need an (N, 2) array of (x, y) pairs.
	xy = np.column_stack([x.flat, y.flat])

	mu = np.array([0.0, 0.0])

	sigma = np.array([.3, .3])
	covariance = np.diag(sigma**2)

	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

	# Reshape back to a (30, 30) grid.
	z = z.reshape(x.shape)
	
	
	
	fig = plt.figure()
	lar = len(x)
	lon = len(x[0])
	size = lar * lon 
	x1 = np.reshape(x,size)
	y1 = np.reshape(y,size)
	z1 = np.reshape(z,size)
	supp = np.array([(x1[i],y1[i],z1[i]) for i in range(size)])
	V = voisinageGrille2D(lar,lon)
	tabDj = np.array([i for i in range(len(supp))])
	R = calcR(supp,V)
	co = 10
	x, y = np.mgrid[-1.0:1.0:10j, -1.0:1.0:10j]
	titre = "PhiGlobauxSupp5v5Calc10v10"
	plt.title(titre)
	for i in range(len(supp)):
		z = []
		#print(i)
		for xi in range (-int(co/2),int(co/2)):
			for yi in range (-int(co/2),int(co/2)):
				C =  np.array([xi*0.5,yi*0.5])
				#print(C)
				z.append(PHIiSF(supp,C,i,tabDj))
				#z.append(PHIiLRSF(supp,C,i,tabDj,V,R))
		z = np.array(z)
		z = z.reshape(x.shape)
		print(z)
		ax = fig.add_subplot(t, t, i+1)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		plt.imshow(z)
	#plt.show()
	plt.savefig(titre)
	
def AffichagePhi7():
	x, y = np.mgrid[-1.0:1.0:10j, -1.0:1.0:10j]
	# Need an (N, 2) array of (x, y) pairs.
	xy = np.column_stack([x.flat, y.flat])

	mu = np.array([0.0, 0.0])

	sigma = np.array([.3, .3])
	covariance = np.diag(sigma**2)

	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

	# Reshape back to a (30, 30) grid.
	z = z.reshape(x.shape)

	fig = plt.figure()
	lar = len(x)
	lon = len(x[0])
	size = lar * lon 
	x1 = np.reshape(x,size)
	y1 = np.reshape(y,size)
	z1 = np.reshape(z,size)
	supp = np.array([(x1[i],y1[i],z1[i]) for i in range(size)])
	V = voisinageGrille2D(lar,lon)

	columns = 20
	rows = 20
	ctp = 0

	plt.title("Phi7Dim20")
	plt.xlim(-1.0, 1.0)
	plt.ylim(-1.0, 1.0)
	x, y = np.mgrid[-1.0:1.0:20j, -1.0:1.0:20j]
	for i in range(len(supp)):
		ctp+=1
		Z=[]
		d=0
		dim = V[i][d][0]
		dip = V[i][d][1]
		Dj  = np.array([dim,dip])
		if(dim==-1):
			Dj = np.array([dip])
			dim = 0
		if(dip==-1):
			Dj = np.array([dim])
		for xi in range (-int(columns/2),int(columns/2)):
			for yi in range (-int(rows/2),int(rows/2)):
				
				C =  np.array([xi*0.1,yi*0.1])
				Z.append(PHIiSF(supp,C,dim,Dj))
		Z = np.array(Z)
		z = Z.reshape(x.shape)
		ax = fig.add_subplot(10,10,ctp)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		plt.imshow(z)
	#plt.show()
	
	plt.savefig("Phi7Dim20")

def test2DDD():
	
	x, y = np.mgrid[-1.0:1.0:6j, -1.0:1.0:6j]
	# Need an (N, 2) array of (x, y) pairs.
	xy = np.column_stack([x.flat, y.flat])

	mu = np.array([0.0, 0.0])
	
	sigma = np.array([.3, .3])
	covariance = np.diag(sigma**2)
	
	z = np.array([0  for i in range(len(x)**2)])
	z[20] = 1
	z[14] = 1
	# Reshape back to a (30, 30) grid.
	z = z.reshape(x.shape)
	

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X=x, Y=y, Z=z)
	plt.savefig("Test2Objectif")
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	lar = len(x)
	lon = len(x[0])
	size = lar * lon 
	x1 = np.reshape(x,size)
	y1 = np.reshape(y,size)
	z1 = np.reshape(z,size)
	supp = np.array([(x1[i],y1[i],z1[i]) for i in range(size)])
	tabVoisin = voisinageGrille2D(lar,lon)

	X=[]
	Y=[]
	Z=[]
	
	print(calcR(supp,tabVoisin))
	for xi in range (-20,21):
		print(xi+20)
		for yi in range (-20,21):
			X.append(xi*0.05)
			Y.append(yi*0.05)
			C =  np.array([xi*0.05,yi*0.05])
			azer= YCFSOM(supp, C, tabVoisin)
			#print(xi*0.1,yi*0.1,azer)
			Z.append(azer)
			
	shape = (int(math.sqrt(len(X))),int(math.sqrt(len(X))))	
	X = np.reshape(X,shape)		
	Y = np.reshape(Y,shape)	
	Z = np.reshape(Z,shape)	


	# Normalize to [0,1]
	norm = plt.Normalize(Z.min(), Z.max())
	colors = cm.viridis(norm(Z))
	rcount, ccount, _ = colors.shape

	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
						   facecolors=colors, shade=False)
	surf.set_facecolor((0,0,0,0))
	plt.savefig("Test2Resultat")
	plt.show()


#AffichagePhi8()	
#AffichagePhi7()

print(voisinageGrille2D(2,2))
#testGausienne()
#test1D()
#test2D()
#test2()
#test2DDD()

#Tester apprentissage SOM (input+outpout) +  interpolation (input)


