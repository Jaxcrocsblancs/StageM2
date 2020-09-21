from Model import *
from Neurones import *
import math

class SOM(Model):
	def __init__(self, nbD, lon=10, lar=10, dim = 2, sigmaInit=1.0, sigmaFinal=0.01, epsilonInit=0.5, epsilonFinal=0.01, tFinal=1000):
		super().__init__(nbD)
		self.N 			  = lon*lar
		self.dim		  = dim
		self.sigmaInit    = sigmaInit
		self.sigmaFinal   = sigmaFinal
		self.epsilonInit  = epsilonInit
		self.epsilonFinal = epsilonFinal
		self.tFinal 	  = tFinal
		self.titre 		  = "SOM"	
		self.initialisation(lon, lar)
		
	def initialisation(self, lon, lar):
		#Génération des neurones
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
					
	def epochs(self,t,x):
		#print(t)
		if(t<self.tFinal):
			#Recherche du BMU (best maching unit)
			dist = np.linalg.norm(self.weights - x, axis=1)
			bmu_idx = np.argmin(dist)
			#Recupération distance entre les neurones et le BMU
			distGrille = self.Voisinage.distVoisinage(bmu_idx,self.N)
			
			#Actualisation des vecteurs de tout les neurones
			self.weights += self.epsilon(t) * self.hvoisin(t,distGrille) *(x - self.weights)
			
			
	def sigma(self, t):
		return self.sigmaInit * (self.sigmaFinal/self.sigmaInit)**(t/self.tFinal)
		
	def hvoisin(self,t,distGrille):
		rep = np.exp( - np.multiply(distGrille,distGrille)/(2*self.sigma(t))**2)
		return np.c_[[rep for i in range(self.dim)]].T
		
		
	def epsilon(self, t):
		return self.epsilonInit * (self.epsilonFinal/self.epsilonInit)**(t/self.tFinal)	

