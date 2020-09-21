from Model import *
from Neurones import *
import math

class GNG(Model):
	def __init__(self, nbD, dim = 2,epsilonB=0.2, epsilonN=0.0006, ageM=10, lambdaT = 50, alpha = 0.5, d=0.995, tFinal=100):
		super().__init__(nbD)
		# Etape 0, initialisation de 2 neurones 
		self.dim 		= dim 
		self.epsilonB 	= epsilonB
		self.epsilonN	= epsilonN
		self.ageM  		= ageM
		self.lambdaT	= lambdaT
		self.alpha		= alpha
		self.d 			= d
		self.tFinal		= tFinal
		self.titre 		= "GNG"
		self.initialisation()
		
	def initialisation(self):
		#Génération des neurones
		self.weights = self.Donnee.initNeurones(2,self.dim)
		self.error 	 = np.array([0.0, 0.0])		
		
	def epochs(self,t,x):
		#print(t)
		#Etape10 critère d'arret 
		if(t<self.tFinal and t>0 ):
			#Etape 2, recuperation des 2 meilleurs neurones
			#Tri ordre selon distance euclidien
			dist = np.linalg.norm(self.weights - x, axis=1)
			trie = np.argsort(dist)
			
			s1 = trie[0]
			s2 = trie[1]
			
			#Etape3 incrementation de l'âge des liens voisins de s1
			self.Voisinage.upLife(s1)
			
			#Etape4 Ajout carré distance euclidien comme erreur dans s1
			#SWAP ALGO DIAPO
			self.error[s1] += dist[s1]**2
			
			#Etape5 Déplacement des vecteurs Donnees
			#Neurones s1			
			self.weights[s1] += self.epsilonB * (x - self.weights[s1])
			
			#Voisin s1
			lVoisin = self.Voisinage.getLvoisin(s1)
			for n in lVoisin:
				self.weights[n] += self.epsilonN * (x - self.weights[n])
				
			#Etape6 Reset life ou Ajout connection s1 s2
			#Suppression si existant
			self.Voisinage.delVoisin(s1,s2)
			#Création lien
			self.Voisinage.addVoisin(s1,s2)
			#Etape7 suppression des liens trop vieux et 
			#Suppression des liens trop vieux
			self.Voisinage.removeToOld(s1, self.ageM)
			
			#suppression des neurones isolé
			for n in lVoisin:
				l = self.Voisinage.getLvoisin(n)
				if(len(l)==0):
					print(n)
					np.delete(self.weights, 0, n)
					
			#Etape8 ajout nouveau neurones 
			if (t%self.lambdaT == 0):
				#Recherche du neurones avec la plus grosse erreur
				trie = np.argsort(self.error)
				idMaxE = trie[-1]
				
				#Recherche du voisin de nMax avec la plus grosse erreur
				lV = self.Voisinage.getLvoisin(idMaxE)
				trie = np.argsort(self.error[lV])
				idMaxE2 = lV[trie[-1]]
				
				#Calcul vecteur du nouveau neurones
				self.weights = np.append(self.weights,0.5*(self.weights[idMaxE]+self.weights[idMaxE2]))
				self.weights = self.weights.reshape((int(len(self.weights)/self.dim),self.dim))
				#Calcul error du nouveau neurones
				self.error[idMaxE] 	-= self.error[idMaxE]*self.alpha
				self.error[idMaxE2] -= self.error[idMaxE2]*self.alpha
	
				self.error	= np.append(self.error,0.5*(self.error[idMaxE]+self.error[idMaxE2]))
				#self.error	= self.error.reshape(len(self.error),1)
				
				#Suppression lien nMax nMax2
				self.Voisinage.delVoisin(idMaxE,idMaxE2)
				#Ajout nouveau lien entre le nouveau neurones et nMax/nMax2
				self.Voisinage.addVoisin(idMaxE,len(self.weights)-1)
				self.Voisinage.addVoisin(idMaxE2,len(self.weights)-1)
				
				#Etape9 Decroitre erreur sur tout les neurones
				self.error = self.error * self.d
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				