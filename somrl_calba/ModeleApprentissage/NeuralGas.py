from Model import *
from Neurones import *
import math

class NG(Model):
	def __init__(self, nbD,N = 100, dim = 2 ,sigmaInit=1.0, sigmaFinal=0.01, epsilonInit=0.5, epsilonFinal=0.01, tFinal=10000):
		super().__init__(nbD)
		self.N 				= N
		self.dim 			= dim 
		self.sigmaInit    	= sigmaInit
		self.sigmaFinal   	= sigmaFinal
		self.epsilonInit  	= epsilonInit
		self.epsilonFinal 	= epsilonFinal
		self.tFinal 		= tFinal
		self.titre 		    = "NG"	
		self.init_weights()
		
		
	def init_weights(self):
		#Génération des neurones
		self.weights = self.Donnee.initNeurones(self.N,self.dim)
					
	def epochs(self,t,x):
		#print(t)
		if(t<self.tFinal):
			
			#Récupération distance
			dist = np.linalg.norm(self.weights - x, axis=1)
			#L'ordre des id pour trier
			trie = np.argsort(dist)
			#Recuperation de l'ordre de chaque id
			trietrie = np.argsort(trie)
			#Actualisation des vecteurs de tout les neurones
			self.weights += self.epsilon(t) * self.hvoisin(t,trietrie) *(x - self.weights)
			
			
	def sigma(self, t):
		return self.sigmaInit * (self.sigmaFinal/self.sigmaInit)**(t/self.tFinal)
		
	def hvoisin(self,t,dist):
		return np.c_[np.exp( - dist/self.sigma(t)),np.exp( - dist/self.sigma(t))]
		
	def epsilon(self, t):
		return self.epsilonInit * (self.epsilonFinal/self.epsilonInit)**(t/self.tFinal)	

