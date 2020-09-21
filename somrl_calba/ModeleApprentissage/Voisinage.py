from Voisin import Voisin
import numpy as np
import sys 

class Voisinage:
	
	def __init__(self):
		self.listVoisin = []
		
	def addVoisin(self,n1,n2,init=0):
		nV = Voisin(n1,n2,init)
		self.listVoisin.append(nV)
	
	def delVoisin(self,n1,n2):
		for i in range (len(self.listVoisin)):
			if( i<(len(self.listVoisin))):
				currentV = self.listVoisin[i]
				if((currentV.n1 == n1 and currentV.n2 == n2) or (currentV.n2 == n1 and currentV.n1 == n2)):
					self.listVoisin.remove(self.listVoisin[i])

	def upLife(self,n1):
		for v in self.listVoisin:
			if(v.n1 == n1):
				v.upLife()
			if(v.n2 == n1):
				v.upLife()
	
	#Retourne la liste des voisins d'un neurone
	def getLvoisin(self,n1):
		lVoisin = []
		for v in self.listVoisin:
			if(v.n1 == n1):
				lVoisin.append(v.n2)
			if(v.n2 == n1):
				lVoisin.append(v.n1)
		return lVoisin
		
	#Recherche par largeur d'abord
	def distMin(self,n1,n2):
		#File current
		f = [n1]
		#Sommet deja testé
		fN = []
		#Distance avec sommet testé
		fDist = [0]
		#Position dans la liste FN current
		nb = 0
		
		while len(f)>0:
			current = f.pop(0)
			fN.append(current)
			#Recherche des voisins
			for i in range (len(self.listVoisin)):
				currentV = self.listVoisin[i]
				
				#Current trouvé 
				if(currentV.n1 == current):
					#Si sommet nouveau
					if(not(exist(fN,currentV.n2)) and not(exist(f,currentV.n2))):
						f.append(currentV.n2)
						fDist.append(fDist[nb]+1)
						
				#Current trouvé 
				if(currentV.n2 == current):
					#Si sommet nouveau
					if(not(exist(fN,currentV.n1)) and not(exist(f,currentV.n1))):
						f.append(currentV.n1)
						fDist.append(fDist[nb]+1)
			nb += 1

		for i in range(len(fN)):
			if (fN[i] == n2):
				return fDist[i]
				
		#Pas de connection entre n1 et n2
		return -1
		
	#Recherche par largeur d'abord
	def distVoisinage(self,s1,N):
		#File current
		f = [s1]
		#Sommet deja testé
		fN = []
		
		fDist = [sys.maxsize for i in range (N)]
		fDist[s1] = 0
		while len(f)>0:
			current = f.pop(0)
			fN.append(current)
			#Recherche des voisins
			for i in range (len(self.listVoisin)):
				currentV = self.listVoisin[i]
				
				#Current trouvé 
				if(currentV.n1 == current):
					#Si sommet nouveau
					if(not(exist(fN,currentV.n2)) and not(exist(f,currentV.n2))):
						f.append(currentV.n2)
						fDist[currentV.n2] = fDist[current]+1
						
				#Current trouvé 
				if(currentV.n2 == current):
					#Si sommet nouveau
					if(not(exist(fN,currentV.n1)) and not(exist(f,currentV.n1))):
						f.append(currentV.n1)
						fDist[currentV.n1] = fDist[current]+1

		return np.array(fDist)
		
		
	#Suppresion des liens voisinage si lien trop vieux
	def removeToOld(self, n1, ageM):
		for v in self.listVoisin:
			if(v.n1 == n1):
				if(v.Life>=ageM):
					self.listVoisin.remove(v)
			if(v.n2 == n1):
				if(v.Life>=ageM):
					self.listVoisin.remove(v)

	
def exist(L,e):
	for l in L:
		if l == e:
			return True
	return False