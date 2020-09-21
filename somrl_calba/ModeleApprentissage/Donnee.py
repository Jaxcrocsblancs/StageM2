import matplotlib.pyplot as plt
from pylab import *
from scipy import signal
import math 
import numpy as np
class Donnee:
	
	def __init__(self):
		self.Data = []
		self.type = ""
		self.distNorm = -1
		
	def setDistributionUniform(self, N=1000, a=-0.5, b=0.5,tailleVecteur = 2):
		self.Data= []
		self.type	 	= "Uniform"
		self.N			= N
		self.limHaute 	= b
		self.limBasse 	= a
		self.dim = tailleVecteur
		self.initialisation()	
		
	def setDistributionUniformMouvent(self, N=1000, a=[-0.5,0,-0.5,0], b=[0,0.5,0,0.5],a2=[0,0,-0.5,-0.5], b2=[0.5,0.5,0,0],tailleVecteur = 2):
		self.Data= []
		self.type	 	= "UniformMouvent"
		self.N			= N
		self.limHaute 	= b
		self.limBasse 	= a
		self.limHaute2 	= b2
		self.limBasse2 	= a2
		self.cpt 		= 0
		self.dim = tailleVecteur
		self.initialisation()	
		
	def setDistributionAnneauUniform(self, N=1000, a=-0.5, b=0.5, centre = [0.5,0.5], distMin = 0.1, distMax = 0.4, tailleVecteur = 2):
		self.Data = []
		self.type		= "AnneauUniform"
		self.N			= N
		self.limHaute 	= b
		self.limBasse	= a
		self.centre 	= centre
		self.distMin 	= distMin 
		self.distMax 	= distMax
		self.dim = tailleVecteur
		self.initialisation()	

	def setDistributionDisqueDensiteIn(self, N=3000, a=-0.5, b=0.5, centre = [0.5,0.5], distMax = 0.5, tailleVecteur = 2):
		self.Data = []
		self.type		= "DisqueDensiteIn"
		self.N			= N
		self.limHaute 	= b
		self.limBasse	= a
		self.centre 	= centre 
		self.distMax 	= distMax
		self.dim = tailleVecteur
		self.initialisation()	
		
	def setDistributionDisqueDensiteOu(self, N=3000, a=-0.5, b=0.5, centre = [0.5,0.5], distMax = 0.5, tailleVecteur = 2):
		self.Data = []
		self.type		= "DisqueDensiteOu"
		self.N			= N
		self.limHaute 	= b
		self.limBasse	= a
		self.centre 	= centre 
		self.distMax 	= distMax
		self.dim = tailleVecteur
		self.initialisation()	
	
	def setDistributionDeuxGroupeRec(self, N=2000, a=-0.2, b=0.2, c = [-0.4,0.2], tailleVecteur = 2):
		self.Data= []
		self.type	 	= "DeuxGroupeRec"
		self.N			= N
		self.limHaute 	= b
		self.limBasse 	= a
		self.c = c
		self.dim = tailleVecteur
		self.initialisation()	


	def initialisation(self):
		if(self.type == "Uniform"):
			self.Data =  self.uniform(self.N,self.dim,self.limHaute,self.limBasse )
		if(self.type == "AnneauUniform"):
			self.Data = self.anneauUniform(self.N)
		if(self.type == "UniformMouvent"):
			self.Data = self.uniform(self.N,self.dim,self.limHaute[self.cpt],self.limBasse[self.cpt] )
		if(self.type == "DisqueDensiteIn"):
			tabIn = self.anneauDense(int(3*self.N/4+1), 0 , self.distMax/2.0)
			tabOu = self.anneauDense(int(self.N/4), self.distMax/2.0, self.distMax )
			self.Data = np.concatenate((tabIn,tabOu))
		if(self.type == "DisqueDensiteOu"):
			tabOu = self.anneauDense(int(9*self.N/10+1), self.distMax/2.0 , self.distMax)
			tabIn = self.anneauDense(int(self.N/10), 0, self.distMax/2.0)
			self.Data = np.concatenate((tabIn,tabOu))
		if(self.type == "DeuxGroupeRec"):
			g1 = self.uniform(int(self.N/2),self.dim, self.limHaute,self.limBasse)+self.c[0]
			g2 = self.uniform(int(self.N/2),self.dim, self.limHaute,self.limBasse)+self.c[1]
			self.Data = np.concatenate((g1,g2))
			
			
	def deplacement(self):
		if(self.type == "UniformMouvent"):
			if(self.cpt<4):
				self.Data = self.uniformBis(self.N,self.dim,[self.limHaute[self.cpt],self.limHaute2[self.cpt]],[self.limBasse[self.cpt],self.limBasse2[self.cpt]])
				self.cpt += 1 
			
	def getVecteurEntre(self):
		id = int(len(self.Data) * np.random.random())
		return self.Data[id]
	
	def uniform(self, N,dim,b,a):
		return  (b-a)*(np.random.random((N, dim)))+a
		
	def uniformBis(self, N,dim,b,a):
		tab = np.random.random((N, dim))
		for i in range (dim):
			tab[:,i] = (b[i]-a[i])*tab[:,i]+a[i]
		return tab
		
	def anneauUniform(self, N):
		return np.array([self.sample_anneau(self.distMin, self.distMax) for i in range(N)])
		
	def anneauDense(self, N, ins, o):
		return np.array([self.sample_anneau(ins, o) for i in range(N)])
		
	def initNeurones(self,N,dim):
		if(self.type == "UniformMouvent" or self.type == "DeuxGroupeRec"):
			return  self.uniform(N,dim,0.1,-0.1)
		elif(self.type != ""):
			return  self.uniform(N,dim,self.limHaute,self.limBasse)
			
	def sample_anneau(self, radius_in, radius_out):
		found = False
		while not found:
			x = radius_out * (2 * np.random.random((2, )) - 1.0)
			found = np.linalg.norm(x) >= radius_in and np.linalg.norm(x) <= radius_out
		return x
	
	def maxDistDonnee(self):
		max = []
		for i in range(self.N):
			d = np.linalg.norm(self.Data - self.Data[i], axis=1)
			max.append(d[np.argmax(d)])
		self.distNorm = max[np.argmax(max)]