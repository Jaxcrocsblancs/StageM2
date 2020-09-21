from Model import *
from Neurones import *
import math

class DSOM(Model):
	def __init__(self, nbD, dim=2, lon=10, lar=10, eta=20, epsilon=0.1, tFinal = 10000):
		super().__init__(nbD)
		self.lon = lon
		self.lar = lar
		self.N 			  = lon*lar
		self.dim		  = dim
		self.eta 		= eta
		self.epsilon 	= epsilon	
		self.tFinal		= tFinal
		self.titre 		= "DSOM"
		#self.Donnee.maxDistDonnee()

		self.canonical_dist = np.zeros((2*self.lar + 1, 2*self.lon + 1))
		# Manhattan distance
		for i in range(self.canonical_dist.shape[0]):
			self.canonical_dist[i, self.lon] = np.abs(self.lar - i)
		for j in range(self.canonical_dist.shape[1]):
			self.canonical_dist[self.lar, j] = np.abs(self.lon - j)
		for i in range(self.lar):
			for j in range(self.lon):
				val = np.sqrt((i+1)**2+(j+1)**2)
				self.canonical_dist[self.lar+i+1, self.lon+j+1] = val
				self.canonical_dist[self.lar-i-1, self.lon-j-1] = val 
				self.canonical_dist[self.lar-i-1, self.lon+j+1] = val
				self.canonical_dist[self.lar+i+1, self.lon-j-1] = val
		self.canonical_dist = np.roll(np.roll(self.canonical_dist, -self.lar, axis=0), -self.lon, axis=1)
		self.initialisation(lon, lar)

	def initialisation(self, lon, lar):
		self.weights = self.Donnee.initNeurones(lon*lar,self.dim)
		#Liens classique par une grille  	
		self.voisinageGrille(lon, lar)

	def voisinageGrille(self, lon, lar):
		for i in range (lon):
			for j in range (lar):
				n1 = i*lar+j
				if(j<lar-1):
					n2 = i*lar+j+1
					self.Voisinage.addVoisin(n1,n2)
				if(i<lon-1):
					n2 = i*lar+j+lar
					self.Voisinage.addVoisin(n1,n2)

	def distVoisinage(self, bmu_idx):
		bmu_i = bmu_idx // self.lon
		bmu_j = bmu_idx % self.lon
		return np.roll(np.roll(self.canonical_dist, bmu_j, axis=1), bmu_i, axis=0)[:self.lar, :self.lon].ravel()

	def epochs(self,t,x):
		if (self.Donnee.distNorm ==-1):
			self.Donnee.maxDistDonnee()
		if(t<self.tFinal):
			#Recherche du BMU (best maching unit)
			dist = np.linalg.norm(self.weights - x, axis=1)
			bmu_idx = np.argmin(dist)
            #Recuperation distance entre les neurones et le BMU
			distGrille = self.distVoisinage(bmu_idx)
			voisinage = np.multiply(self.distEuclideanNorm(x), self.hvoisin(distGrille,x,bmu_idx)).reshape((self.N, 1))
			self.weights += self.epsilon * np.multiply(voisinage, (x - self.weights))
			
	def hvoisin(self,distGrille,x,bmu_idx):
		return np.exp(-(1/(self.eta**2)) * distGrille**2/(np.linalg.norm(self.weights[bmu_idx] - x)/ self.Donnee.distNorm)**2)

	def distEuclideanNorm(self,x):
		return np.linalg.norm(self.weights - x, axis=1) / self.Donnee.distNorm
	
	def distBMU(self,x):
		dist = np.linalg.norm(self.weights - x, axis=1)
		return dist[np.argmin(dist)]
