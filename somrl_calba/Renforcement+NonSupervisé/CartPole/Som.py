import gym
import numpy as np
import matplotlib.pyplot as plt

from Voisinage import *

class SOM():
	def __init__(self, lon=10, lar=10, dim = 2, sigmaInit=1, sigmaFinal=0.1, epsilonInit=0.5, epsilonFinal=0.01, tFinal=100, lim=[[-1.2,0.6],[-0.07,0.07]], nbAction=3):
		self.N 			  = lon*lar
		self.lar		  = lar
		self.lon		  = lon
		self.dim		  = dim
		self.sigmaInit    = sigmaInit
		self.sigmaFinal   = sigmaFinal
		self.epsilonInit  = epsilonInit
		self.epsilonFinal = epsilonFinal
		self.tFinal 	  = tFinal
		self.titre 		  = "SOM"	
		self.lim 		  = lim
		self.Donnee 	  = []
		self.nbAction     = nbAction
		self.Voisinage = Voisinage()
		self.initialisation(lon,lar)
		self.distGrille = np.array([self.Voisinage.distVoisinage(i,self.N) for i in range(self.N)])
		
	def initialisation(self,lon,lar):
		#Génération des neurones
		self.weights = np.array([self.initWeight() for i in range(self.N)])
		self.Q 		 = np.array([[0.0 for j in range(self.nbAction)] for i in range(self.N)])
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
					
	def epochs(self,t,x):
		#Test erreur
		if(len(x) != self.dim):
			return 
			
		self.Donnee.append(x)
		if(t<self.tFinal):
			
			xNorm = np.array([0.0 for i in range (self.dim)])
			for i in range (self.dim):
				xNorm[i] = (x[i]-self.lim[i][1])/(self.lim[i][1]-self.lim[i][0])+1
			#Recherche du BMU (best maching unit)
			dist = np.linalg.norm(self.weights - xNorm, axis=1)
			bmu_idx = np.argmin(dist)
			#Recupération distance entre les neurones et le BMU
			
			#Actualisation des vecteurs de tout les neurones
			self.weights += self.epsilon(t) * self.hvoisin(t,self.distGrille[bmu_idx]) *(xNorm - self.weights)
			
	def sigma(self, t):
		return self.sigmaInit * (self.sigmaFinal/self.sigmaInit)**(t/self.tFinal)
		
	def hvoisin(self,t,distGrille):
		rep = np.exp( - np.multiply(distGrille,distGrille)/(2*self.sigma(t))**2)
		return np.c_[[rep for i in range(self.dim)]].T
		
	def epsilon(self, t):
		return self.epsilonInit * (self.epsilonFinal/self.epsilonInit)**(t/self.tFinal)	
	
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





