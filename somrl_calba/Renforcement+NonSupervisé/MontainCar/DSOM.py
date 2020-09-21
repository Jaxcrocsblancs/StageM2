import math
import numpy as np
from Voisinage import *

class DSOM():
	def __init__(self, dim=2, lon=10, lar=10, eta=10, epsilon=0.1, tFinal = 10000, lim=[[-1.2,0.6],[-0.07,0.07]]):
		self.lon 			= lon
		self.lar 			= lar
		self.N 				= lon*lar
		self.dim			= dim
		self.eta 			= eta
		self.epsilon 		= epsilon	
		self.tFinal			= tFinal
		self.titre 			= "DSOM"
		self.lim 		  	= lim
		self.Donnee 	  	= []
		self.Voisinage = Voisinage()
		self.distNorm		= 1

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
		self.distGrille = np.array([self.distVoisinage(i) for i in range(self.N)])


	def initialisation(self, lon, lar):
		self.weights = np.array([self.initWeight() for i in range(self.N)])
		self.Q 		 = np.array([[0.0,0.0,0.0] for i in range(self.N)])
		#Liens classique par une grille  	
		self.voisinageGrille(lon, lar)
	
	def initWeight(self):
		rep = []
		for i in range(self.dim):
			rep.append(np.random.random())
		return rep
		
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
	
	def maxDistDonnee(self):
		max = []
		for i in range(self.N):
			d = np.linalg.norm(self.Donnee - self.Donnee[i], axis=1)
			max.append(d[np.argmax(d)])
		self.distNorm = max[np.argmax(max)]
		
	def epochs(self,t,x):
		xNorm = np.array([0.0 for i in range (self.dim)])
		for i in range (self.dim):
			xNorm[i] = (x[i]-self.lim[i][1])/(self.lim[i][1]-self.lim[i][0])+1
	
		self.Donnee.append(xNorm)
		if (len(self.Donnee)%5000 == 0 and len(self.Donnee)< 500000):
			self.maxDistDonnee()
		if(t<self.tFinal):
			
				
			#Recherche du BMU (best maching unit)
			dist = np.linalg.norm(self.weights - xNorm, axis=1)
			bmu_idx = np.argmin(dist)
            #Recuperation distance entre les neurones et le BMU
			voisinage = np.multiply(self.distEuclideanNorm(xNorm), self.hvoisin(self.distGrille[bmu_idx],xNorm,bmu_idx)).reshape((self.N, 1))
			self.weights += self.epsilon * np.multiply(voisinage, (xNorm - self.weights))
			
	def hvoisin(self,distGrille,x,bmu_idx):
		return np.exp(-(1/(self.eta**2)) * distGrille**2/(np.linalg.norm(self.weights[bmu_idx] - x)/ self.distNorm)**2)

	def distEuclideanNorm(self,x):
		return np.linalg.norm(self.weights - x, axis=1) / self.distNorm
	
	def distBMU(self,x):
		dist = np.linalg.norm(self.weights - x, axis=1)
		return dist[np.argmin(dist)]
		
	def getBMU(self,x):
		xNorm = np.array([0.0 for i in range (self.dim)])
		for i in range (self.dim):
			xNorm[i] = (x[i]-self.lim[i][1])/(self.lim[i][1]-self.lim[i][0])+1
		dist = np.linalg.norm(self.weights - xNorm, axis=1)
		bmu_idx = np.argmin(dist)
		return bmu_idx

	def getWeight(self,j):
		rep = np.array([0.0 for i in range (self.dim)])
		for i in range (self.dim):
			rep[i] = (self.weights[j][i]-1.0)*(self.lim[i][1]-self.lim[i][0])+self.lim[i][1]
		return rep
		
	def getWeights(self):
		rep = []
		for j in range(self.N):
			current = np.array([0.0 for i in range (self.dim)])	
			for i in range (self.dim):
				current[i] = (self.weights[j][i]-1.0)*(self.lim[i][1]-self.lim[i][0])+self.lim[i][1]
			rep.append(current)
		rep = np.array(rep)
		return rep	
		
	def normalize(self,x):
		xNorm = np.array([0.0 for i in range (self.dim)])
		for i in range (self.dim):
			xNorm[i] = (x[i]-self.lim[i][1])/(self.lim[i][1]-self.lim[i][0])+1	
		return xNorm
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
