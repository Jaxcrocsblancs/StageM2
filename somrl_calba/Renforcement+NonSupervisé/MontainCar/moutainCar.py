import gym
import numpy as np
import matplotlib.pyplot as plt
from Som import *
from DSOM import *
from goppert import *

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import argparse
import os
import pylab
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def all_video( index ):
    return True
	
def voisinageGrille2D(lar,lon):
					size = lar*lon
					return np.array([np.array([[k-1 if (k-1>=0 and (k)%lar!=0) else -1 , k+1 if (k+1<size and (k+1)%lar!=0) else -1],[k-lar if k-lar>=0 else -1, k+lar if(k+lar<size) else -1]]) for k in range (size)])
							
class MountainCar():
	def __init__(self):
		self.env 				= gym.make('MountainCar-v0')
		self.save 				= True
		self.load 				= False
		self.nameFile 			= "MountainCar"
		self.repertoire  		= "Image"
		self.nbCartPosition 	= 40
		self.minCartPosition 	= -1.2
		self.maxCartPosition 	= 0.6
		
		self.nbCartVelocity 	= 40
		self.minCartVelocity	= -0.07
		self.maxCartVelocity	= 0.07
		
		self.nbAction       	= 3
		
		self.alphaInit 			= 0.1
		self.alphaFinal			= 0.001	
		self.gamma 				= 0.9
		self.som 				= SOM()	
		self.dSom				= DSOM()
		self.Q 					= np.zeros((self.nbCartPosition, self.nbCartVelocity,self.nbAction))

		self.main()		
		
	def maxAction(self,Q):
			idMax  = 0;
			valMax = -100000;
			for i in range(len(Q)):
				valCur = Q[i]
				if(valMax<valCur):
					idMax  = i;
					valMax = valCur;
			return idMax
			
	def getPosP(self,ob,max,min,nb):
		d = max-min
		for i in range(nb):
			if ob<(d/nb*i+min):
				return i-1
		return nb-1
		
	def getPos(self,observation):
		return (self.getPosP(observation[0],self.maxCartPosition,self.minCartPosition,self.nbCartPosition),self.getPosP(observation[1],self.maxCartVelocity ,self.minCartVelocity ,self.nbCartVelocity))
	
	#==========APPRENTISSAGE=========#
	def Apprentissage(self,nbEpi,nbIte):
		eps = 0.02
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			pos = self.getPos(observation)
			ansPos = pos
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.Q[pos]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				pos = self.getPos(observation)
				posSom = self.som.getBMU(observation)
				self.Q[ansPos][action] = self.Q[ansPos][action] + alphaCurrent * (reward + self.gamma*self.Q[pos][self.maxAction(self.Q[pos])] - self.Q[ansPos][action])

				ansPos = pos
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)
	
	def ApprentissageSom(self,nbEpi,nbIte):
		eps = 0.02
		self.som = SOM()	
		self.som.tFinal = nbEpi/3*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.som.getBMU(observation)
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps or (i_episode<nbEpi/3):
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.som.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.som.getBMU(observation)
				if (i_episode>nbEpi/3):
					self.som.Q[ansPosSom][action] = self.som.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.som.Q[posSom][self.maxAction(self.som.Q[posSom])]-self.som.Q[ansPosSom][action])
				else:
					self.som.epochs(i_episode*nbIte+t,observation)
				
				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)		
			
	def ApprentissageSom2(self,nbEpi,nbIte):
		eps = 0.02
		self.som.tFinal = nbEpi*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.som.getBMU(observation)
			ansPosSom = posSom
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.som.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.som.getBMU(observation)
				self.som.Q[ansPosSom][action] = self.som.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.som.Q[posSom][self.maxAction(self.som.Q[posSom])]-self.som.Q[ansPosSom][action])
				self.som.epochs(i_episode*nbIte+t,observation)
				
				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)
			
	def ApprentissageSom3(self,nbEpi,nbIte):
		eps = 0.02
		pourcentageSOM = 4
		self.som.tFinal = nbEpi*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.som.getBMU(observation)
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps or (i_episode%10<pourcentageSOM):
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.som.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.som.getBMU(observation)
				if ((i_episode%10>pourcentageSOM)):
					self.som.Q[ansPosSom][action] = self.som.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.som.Q[posSom][self.maxAction(self.som.Q[posSom])]-self.som.Q[ansPosSom][action])
				else:
					self.som.epochs(i_episode*nbIte+t,observation)
				
				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)
	
	def ApprentissageDSom(self,nbEpi,nbIte):
		eps = 0.02
		self.dSom.tFinal = nbEpi/3*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.dSom.getBMU(observation)
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps or (i_episode<nbEpi/3):
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.dSom.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.dSom.getBMU(observation)
				if (i_episode>nbEpi/3):
					self.dSom.Q[ansPosSom][action] = self.dSom.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.dSom.Q[posSom][self.maxAction(self.dSom.Q[posSom])]-self.dSom.Q[ansPosSom][action])
				else:
					self.dSom.epochs(i_episode*nbIte+t,observation)
				

				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)
			#self.affichageDSom(i_episode)
			
	def ApprentissageDSom2(self,nbEpi,nbIte):
		eps = 0.02
		self.dSom.tFinal = nbEpi/10*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.dSom.getBMU(observation)
			t = 0
			#Alpha décroissant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps or (i_episode<nbEpi/10):
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.dSom.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.dSom.getBMU(observation)
				if (i_episode>nbEpi/10):
					self.dSom.Q[ansPosSom][action] = self.dSom.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.dSom.Q[posSom][self.maxAction(self.dSom.Q[posSom])]-self.dSom.Q[ansPosSom][action])
				else:
					self.dSom.epochs(i_episode*nbIte+t,observation)
				

				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			print(i_episode,t, reward,observation)
			#self.affichageDSom(i_episode)
			
	def ApprentissageDSomSimul(self,nbEpi,nbIte):
		eps = 0.02
		self.dSom.tFinal = nbEpi*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.dSom.getBMU(observation)
			ansPosSom = posSom
			t = 0
			#Alpha constant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.dSom.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.dSom.getBMU(observation)
				self.dSom.epochs(i_episode*nbIte+t,observation)
				self.dSom.Q[ansPosSom][action] = self.dSom.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.dSom.Q[posSom][self.maxAction(self.dSom.Q[posSom])]-self.dSom.Q[ansPosSom][action])
				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			now = datetime.now()
			current_tim = now.strftime("%H:%M:%S:%f")
			print(i_episode,t,current_tim)	
			
	def ApprentissageSomSimul(self,nbEpi,nbIte):
		eps = 0.02
		self.som.tFinal = nbEpi*nbIte
		
		for i_episode in range(nbEpi):
			observation = self.env.reset()
			posSom = self.som.getBMU(observation)
			ansPosSom = posSom
			t = 0
			#Alpha constant
			alphaCurrent = self.alphaInit * (self.alphaFinal/self.alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = self.som.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = self.som.getBMU(observation)
				self.som.epochs(i_episode*nbIte+t,observation)
				self.som.Q[ansPosSom][action] = self.som.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*self.som.Q[posSom][self.maxAction(self.som.Q[posSom])]-self.som.Q[ansPosSom][action])
				ansPosSom = posSom
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
			now = datetime.now()
			current_tim = now.strftime("%H:%M:%S:%f")
			print(i_episode,t,current_tim)	

	def ApprentissageModeleSimulIteConstant(self,model,nbIteMax,nbIte,nbFrequenceSave,repertoire):
		eps = 0.02
		model.tFinal = nbIteMax
		cpt = 0
		nbepi = 0
		alphaCurrent = self.alphaInit 
		while (cpt<nbIteMax):
			observation = self.env.reset()
			posSom = model.getBMU(observation)
			ansPosSom = posSom
			#Alpha constant
			for t in range(nbIte):
				alphaCurrent = alphaCurrent * (self.alphaFinal/self.alphaInit)**(1/nbIteMax)
				 
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = self.env.action_space.sample()
				else:
					#Pro
					logits = model.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(self.env.action_space.n, p=probs)
					
				observation, reward, done, info = self.env.step(action)
				
				posSom = model.getBMU(observation)
				model.epochs(cpt,observation)
				model.Q[ansPosSom][action] = model.Q[ansPosSom][action] + alphaCurrent * (reward + self.gamma*model.Q[posSom][self.maxAction(model.Q[posSom])]-model.Q[ansPosSom][action])
				ansPosSom = posSom
				if(cpt%nbFrequenceSave == 0):
					chemin = repertoire+"/"+str(cpt)
					os.makedirs(chemin)
					data = np.asarray(model.Q)
					np.save(chemin+"/Q.npy", data)
					data = np.asarray(model.weights)
					np.save(chemin+"/W.npy", data)
				#Arret si résolue
				if (observation[0] >= 0.5 and observation[1] >= 0):
					break;		
				if (cpt>=nbIteMax):
					break;
				cpt += 1
			now = datetime.now()
			current_tim = now.strftime("%H:%M:%S:%f")
			print(nbepi, cpt,current_tim)	
			nbepi += 1 
	
	#==========EXPLOITATION==========#	
	def ExploitationDiscret(self,Q,nbIte):
		print("===========Exploitation=========")	
		self.env = gym.wrappers.Monitor(self.env, './videos/' + 'dqnBN_MsPacman_video',
										   video_callable=all_video, ## video from all run
										   force=True, ## erase previous video
										   mode='evaluation', ## not very useful but...
										  )
		observation = self.env.reset()
		
		X = []
		Y = []
		for t in range(nbIte):
			self.env.render()
			pos = self.getPos(observation)
			action = self.maxAction(self.Q[pos])

			observation, reward, done, info = self.env.step(action)
			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;	
		self.env.reset()		
		return (X,Y)	

	def ExploitationSom(self,nbIte):
		print("===========Exploitation=========")	
		observation = self.env.reset()
		X = []
		Y = []
		for t in range(nbIte):
			self.env.render()
			
			posSom = self.som.getBMU(observation)
			action = self.maxAction(self.som.Q[posSom])
			
			observation, reward, done, info = self.env.step(action)
			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;		
		return (X,Y)	
		
	def ExploitationDSom(self,nbIte):
		print("===========Exploitation=========")	
		observation = self.env.reset()
		X = []
		Y = []
		for t in range(nbIte):
			self.env.render()
			
			posSom = self.dSom.getBMU(observation)
			action = self.maxAction(self.dSom.Q[posSom])
			
			observation, reward, done, info = self.env.step(action)
			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;		
		return (X,Y)
		
	def ExploitationModel(self,nbIte,model):
		observation = self.env.reset()
		X = []
		Y = []
		nbExec = nbIte
		for t in range(nbIte):
			#self.env.render()
			
			posSom = model.getBMU(observation)
			action = self.maxAction(model.Q[posSom])
			
			observation, reward, done, info = self.env.step(action)
			X.append(observation[0])
			Y.append(observation[1])
			
			if (observation[0] >= 0.5 and observation[1] >= 0):
				nbExec = t 
				break;			
		return (X,Y,nbExec)	
	
	def ExploitationGoppertModel(self,nbIte,model):
		observation = self.env.reset()
		X = []
		Y = []
		V= voisinageGrille2D(model.lar,model.lon)
		tabgop = [goppert(np.concatenate((model.weights,model.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		nbExec = nbIte
		for t in range(nbIte):
			rep = []
			for i in range(self.nbAction):
				rep.append(tabgop[i].YCFSOM(model.normalize(observation)))
			rep = np.array(rep)
			action = self.maxAction(rep)
		
			observation, reward, done, info = self.env.step(action)

			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				nbExec = t 
				break;	
		now = datetime.now()
		current_t = now.strftime("%H:%M:%S:%f")
		print(nbExec,current_t)
		return (X,Y,nbExec)	
	
	def ExploitationGoppertSom(self,nbIte):
		print("===========Exploitation=========")	
		observation = self.env.reset()
		X = []
		Y = []
		
		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
		V= voisinageGrille2D(self.som.lar,self.som.lon)
		tabgop = [goppert(np.concatenate((self.som.weights,self.som.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		for t in range(nbIte):
			print(t)
			self.env.render()
			rep = []
			for i in range(self.nbAction):
				rep.append(tabgop[i].YCFSOM(self.som.normalize(observation)))
			rep = np.array(rep)
			action = self.maxAction(rep)
			print(observation,action)
			
			observation, reward, done, info = self.env.step(action)

			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;	
		return (X,Y)	
		
	def ExploitationGoppertSomNoNormalized(self,nbIte):
		print("===========Exploitation=========")	
		observation = self.env.reset()
		X = []
		Y = []
		
		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
		V= voisinageGrille2D(self.som.lar,self.som.lon)
		tabgop = [goppert(np.concatenate((self.som.getWeights(),self.som.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		for t in range(nbIte):
			print(t)
			self.env.render()
			rep = []
			for i in range(self.nbAction):
				rep.append(tabgop[i].YCFSOM(observation))
			rep = np.array(rep)
			action = self.maxAction(rep)
			print(observation,action)
			
			observation, reward, done, info = self.env.step(action)

			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;	
		return (X,Y)	
		
	def ExploitationGoppertDSom(self,nbIte):
		print("===========Exploitation=========")	
		observation = self.env.reset()
		X = []
		Y = []
		
		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
		V= voisinageGrille2D(self.dSom.lar,self.dSom.lon)
		tabgop = [goppert(np.concatenate((self.dSom.weights,self.dSom.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		for t in range(nbIte):
			print(t)
			self.env.render()
			rep = []
			for i in range(self.nbAction):
				rep.append(tabgop[i].YCFSOM(self.dSom.normalize(observation)))
			rep = np.array(rep)
			action = self.maxAction(rep)
			print(observation,action)
			
			observation, reward, done, info = self.env.step(action)

			X.append(observation[0])
			Y.append(observation[1])
			if (observation[0] >= 0.5 and observation[1] >= 0):
				print(t)
				break;	
		return (X,Y)	
		
	#===========AFFICHAGE===========#	
	def affichageDiscret(self,Q,X,Y):
		fig = plt.figure(facecolor='white')
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		ax = fig.add_subplot(111)

		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity

		for i in range(len(self.Q)):
			for j in range(len(self.Q[i])):
				act = self.maxAction(self.Q[i][j])
				#Accelerate to the Left
				if(act == 0):
					color="red" 
					mark = "<"
				#Don't accelerate
				elif(act == 1):
					color="magenta"
					mark = "D"
				#Accelerate to the Right
				else:
					color="blue" 
					mark = ">"
				#Pas appris
				if (self.Q[i][j][act]==0):
					color = "grey"
				ax.scatter(i*inteCartPosition+self.minCartPosition+inteCartPosition/2, j*inteCartVelocity+self.minCartVelocity+inteCartVelocity/2, s=100,marker=mark,c=color)	
		plt.plot(X,Y,c="black")
		plt.xlabel("Position")
		plt.ylabel("Vitesse")
		
		
		gauche = mlines.Line2D([], [], color='red', marker='<', linestyle='None',
                          markersize=10, label='Accélération à gauche')
		rien   = mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
                          markersize=10, label='Ne rien faire')
		droite = mlines.Line2D([], [], color='blue', marker='>', linestyle='None',
                          markersize=10, label='Accélération à droite')
		noir   = mlines.Line2D([], [], color='black', marker='None',
                          markersize=10, label='Exploitation')
		gris   = mlines.Line2D([], [], color='gray', marker='<', linestyle='None',
                          markersize=10, label='Zone pas assez exploré')				  
		plt.legend(handles=[gauche, rien, droite,noir,gris],loc = 'lower right',)
		plt.savefig("Image/Discret.png")
	
	def affichageSom(self,i_episode=0,X=[],Y=[]):
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		for i in range(self.som.N):
			act = self.maxAction(self.som.Q[i])
			#Accelerate to the Left
			if(act == 0):
				color="red" 
				mark = "<"
			#Don't accelerate
			elif(act == 1):
				color="magenta"
				mark = "D"
			#Accelerate to the Right
			else:
				color="blue" 
				mark = ">"
			#Pas appris
			if (self.som.Q[i][act]==0):
				color = "grey"
			ax.scatter(self.som.getWeight(i)[0],self.som.getWeight(i)[1], s=100,marker=mark,c=color)	
		for i in range(len(self.som.Voisinage.listVoisin)):
			n1 = self.som.Voisinage.listVoisin[i].n1
			n2 = self.som.Voisinage.listVoisin[i].n2
			ax.plot([self.som.getWeight(n1)[0], self.som.getWeight(n2)[0]], [self.som.getWeight(n1)[1], self.som.getWeight(n2)[1]], 'g-')
		#for i in range(len(self.som.Donnee)):
		#	ax.scatter(self.som.Donnee[i][0],self.som.Donnee[i][1], s=10)		
		if (len(X)>0 and len(Y)>0 and len(X)== len(Y)):
			plt.plot(X,Y,c="black")
			
			plt.xlabel("Position")
			plt.ylabel("Vitesse")
			
			
			gauche = mlines.Line2D([], [], color='red', marker='<', linestyle='None',
							  markersize=10, label='Accélération à gauche')
			rien   = mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
							  markersize=10, label='Ne rien faire')
			droite = mlines.Line2D([], [], color='blue', marker='>', linestyle='None',
							  markersize=10, label='Accélération à droite')
			noir   = mlines.Line2D([], [], color='black', marker='None',
							  markersize=10, label='Exploitation')
			gris   = mlines.Line2D([], [], color='gray', marker='<', linestyle='None',
							  markersize=10, label='Zone pas assez exploré')				  
			plt.legend(handles=[gauche, rien, droite,noir,gris],loc = 'lower right',)
			print(self.repertoire+"/SOM.png")
			plt.savefig(self.repertoire+"/SOM.png")
		else:
			plt.savefig("Image/%d.png" % i_episode)
		plt.close(fig)
	
	def affichageDSom(self,i_episode=0,X=[],Y=[]):
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		for i in range(self.dSom.N):
			act = self.maxAction(self.dSom.Q[i])
			#Accelerate to the Left
			if(act == 0):
				color="red" 
				mark = "<"
			#Don't accelerate
			elif(act == 1):
				color="magenta"
				mark = "D"
			#Accelerate to the Right
			else:
				color="blue" 
				mark = ">"
			#Pas appris
			if (self.dSom.Q[i][act]==0):
				color = "grey"
			ax.scatter(self.dSom.getWeight(i)[0],self.dSom.getWeight(i)[1], s=100,marker=mark,c=color)	
		for i in range(len(self.dSom.Voisinage.listVoisin)):
			n1 = self.dSom.Voisinage.listVoisin[i].n1
			n2 = self.dSom.Voisinage.listVoisin[i].n2
			ax.plot([self.dSom.getWeight(n1)[0], self.dSom.getWeight(n2)[0]], [self.dSom.getWeight(n1)[1], self.dSom.getWeight(n2)[1]], 'g-')
		#for i in range(len(self.dSom.Donnee)):
		#	ax.scatter(self.dSom.Donnee[i][0],self.dSom.Donnee[i][1], s=10)		
		if (len(X)>0 and len(Y)>0 and len(X)== len(Y)):
			plt.plot(X,Y,c="black")
			plt.savefig(self.repertoire+"/DSOM.png")
		else:
			plt.savefig("Image/DSOM_epi%d.png" % i_episode)
		plt.close(fig)
	
	def affichageModel(self,model,rep,X=[],Y=[]):
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		for i in range(model.N):
			act = self.maxAction(model.Q[i])
			#Accelerate to the Left
			if(act == 0):
				color="red" 
				mark = "<"
			#Don't accelerate
			elif(act == 1):
				color="magenta"
				mark = "D"
			#Accelerate to the Right
			else:
				color="blue" 
				mark = ">"
			#Pas appris
			if (model.Q[i][act]==0):
				color = "grey"
			ax.scatter(model.getWeight(i)[0],model.getWeight(i)[1], s=100,marker=mark,c=color)	
		for i in range(len(model.Voisinage.listVoisin)):
			n1 = model.Voisinage.listVoisin[i].n1
			n2 = model.Voisinage.listVoisin[i].n2
			ax.plot([model.getWeight(n1)[0], model.getWeight(n2)[0]], [model.getWeight(n1)[1], model.getWeight(n2)[1]], 'g-')
		#for i in range(len(model.Donnee)):
		#	ax.scatter(model.Donnee[i][0],model.Donnee[i][1], s=10)		
		if (len(X)>0 and len(Y)>0 and len(X)== len(Y)):
			plt.plot(X,Y,c="black")
		plt.xlabel("Position")
		plt.ylabel("Vitesse")
		plt.savefig(rep+"Image.png")
		print(rep+"Image.png")
		plt.close(fig)

	def affichageV(self,Q):
		
		V= np.zeros((self.nbCartPosition, self.nbCartVelocity))
		for i in range(len(Q)):
			for j in range(len(Q[i])):
				V[j][i] = Q[i][j][self.maxAction(Q[i][j])]
		levels = MaxNLocator(nbins=15).tick_values(V.min(), 0)
		cmap = plt.get_cmap()
		norm = BoundaryNorm(V, ncolors=cmap.N, clip=True)
		fig, ax0= plt.subplots()
		dx, dy = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition, (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
		y, x = np.mgrid[slice(-0.07, 0.07 + dy, dy),
						slice(-1.2, 0.6  + dx, dx)]
		im = ax0.pcolormesh(x, y, V, cmap=cmap)
		fig.colorbar(im, ax=ax0)
		ax0.set_title('V map')
		plt.savefig("Image/V.png")
		
	def affichageGoppertSom(self,i_episode=0,X1=[],Y1=[]):
		fig = plt.figure(facecolor='white')
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		ax = fig.add_subplot(111)

		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
	
	
		V= voisinageGrille2D(self.som.lar,self.som.lon)
		tabgop = [goppert(np.concatenate((self.som.weights,self.som.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		X=[]
		Y=[]
		Z=[]
		for xi in range (self.nbCartPosition):
			print(xi)
			for yi in range (self.nbCartVelocity):
				
				C = np.array([xi/self.nbCartPosition
							 ,yi/self.nbCartVelocity])
				X.append(C[0])
				Y.append(C[1])
				rep = []
				for i in range(self.nbAction):
					rep.append(tabgop[i].YCFSOM(C))
				rep = np.array(rep)
				act = self.maxAction(rep)
				#print(	xi,
				#		yi,
				#		rep,
				#		act)
			
				
				#Accelerate to the Left
				if(act == 0):
					color="red" 
					mark = "<"
				#Don't accelerate
				elif(act == 1):
					color="magenta"
					mark = "D"
				#Accelerate to the Right
				else:
					color="blue" 
					mark = ">"
				ax.scatter(xi*(self.maxCartPosition-self.minCartPosition)/self.nbCartPosition+self.minCartPosition+inteCartPosition/2, 
						   yi*(self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity+self.minCartVelocity+inteCartVelocity/2, 
						   s=100,marker=mark,c=color)			
		if (len(X1)>0 and len(Y1)>0 and len(X1)== len(Y1)):
			plt.plot(X1,Y1,c="black")
			plt.xlabel("Position")
			plt.ylabel("Vitesse")
			gauche = mlines.Line2D([], [], color='red', marker='<', linestyle='None',
							  markersize=10, label='Accélération à gauche')
			rien   = mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
							  markersize=10, label='Ne rien faire')
			droite = mlines.Line2D([], [], color='blue', marker='>', linestyle='None',
							  markersize=10, label='Accélération à droite')
			noir   = mlines.Line2D([], [], color='black', marker='None',
							  markersize=10, label='Exploitation')
			gris   = mlines.Line2D([], [], color='gray', marker='<', linestyle='None',
							  markersize=10, label='Zone pas assez exploré')				  
			plt.legend(handles=[gauche, rien, droite,noir,gris],loc = 'lower right',)
			
		plt.savefig("Image/GoppertSOM.png")
		
	def affichageGoppertSomNoNormalized(self,i_episode=0,X1=[],Y1=[]):
		fig = plt.figure(facecolor='white')
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		ax = fig.add_subplot(111)

		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
	
	
		V= voisinageGrille2D(self.som.lar,self.som.lon)
		tabgop = [goppert(np.concatenate((self.som.getWeights(),self.som.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		X=[]
		Y=[]
		Z=[]
		for xi in range (self.nbCartPosition):
			print(xi)
			for yi in range (self.nbCartVelocity):
				
				C = np.array([xi*(self.maxCartPosition-self.minCartPosition)/self.nbCartPosition+self.minCartPosition+inteCartPosition/2
							 ,yi*(self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity+self.minCartVelocity+inteCartVelocity/2])
				X.append(C[0])
				Y.append(C[1])
				rep = []
				for i in range(self.nbAction):
					rep.append(tabgop[i].YCFSOM(C))
				rep = np.array(rep)
				act = self.maxAction(rep)
				#print(	xi,
				#		yi,
				#		rep,
				#		act)
			
				
				#Accelerate to the Left
				if(act == 0):
					color="red" 
					mark = "<"
				#Don't accelerate
				elif(act == 1):
					color="magenta"
					mark = "D"
				#Accelerate to the Right
				else:
					color="blue" 
					mark = ">"
				ax.scatter(xi*(self.maxCartPosition-self.minCartPosition)/self.nbCartPosition+self.minCartPosition+inteCartPosition/2, 
						   yi*(self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity+self.minCartVelocity+inteCartVelocity/2, 
						   s=100,marker=mark,c=color)			
		if (len(X1)>0 and len(Y1)>0 and len(X1)== len(Y1)):
			plt.plot(X1,Y1,c="black")
			plt.xlabel("Position")
			plt.ylabel("Vitesse")
			gauche = mlines.Line2D([], [], color='red', marker='<', linestyle='None',
							  markersize=10, label='Accélération à gauche')
			rien   = mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
							  markersize=10, label='Ne rien faire')
			droite = mlines.Line2D([], [], color='blue', marker='>', linestyle='None',
							  markersize=10, label='Accélération à droite')
			noir   = mlines.Line2D([], [], color='black', marker='None',
							  markersize=10, label='Exploitation')
			gris   = mlines.Line2D([], [], color='gray', marker='<', linestyle='None',
							  markersize=10, label='Zone pas assez exploré')				  
			plt.legend(handles=[gauche, rien, droite,noir,gris],loc = 'lower right',)
			
		plt.savefig("Image/GoppertSOMNoNormalized.png")	
		
	def affichageGoppertDSom(self,i_episode=0,X1=[],Y1=[]):
		fig = plt.figure(facecolor='white')
		plt.xlim(self.minCartPosition, self.maxCartPosition)
		plt.ylim(self.minCartVelocity, self.maxCartVelocity)
		ax = fig.add_subplot(111)

		inteCartPosition = (self.maxCartPosition-self.minCartPosition)/self.nbCartPosition
		inteCartVelocity = (self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity
	
	
		V= voisinageGrille2D(self.dSom.lar,self.dSom.lon)
		tabgop = [goppert(np.concatenate((self.dSom.weights,self.dSom.Q[:, [i]]), axis=1),V) for i in range(self.nbAction)]
		
		X=[]
		Y=[]
		Z=[]
		for xi in range (self.nbCartPosition):
			for yi in range (self.nbCartVelocity):
				
				C = np.array([xi/self.nbCartPosition
							 ,yi/self.nbCartVelocity])
				X.append(C[0])
				Y.append(C[1])
				rep = []
				for i in range(self.nbAction):
					rep.append(tabgop[i].YCFSOM(C))
				rep = np.array(rep)
				act = self.maxAction(rep)
				#print(	xi,
				#		yi,
				#		rep,
				#		act)
			
				
				#Accelerate to the Left
				if(act == 0):
					color="red" 
					mark = "<"
				#Don't accelerate
				elif(act == 1):
					color="magenta"
					mark = "D"
				#Accelerate to the Right
				else:
					color="blue" 
					mark = ">"
				ax.scatter(xi*(self.maxCartPosition-self.minCartPosition)/self.nbCartPosition+self.minCartPosition+inteCartPosition/2, 
						   yi*(self.maxCartVelocity-self.minCartVelocity)/self.nbCartVelocity+self.minCartVelocity+inteCartVelocity/2, 
						   s=100,marker=mark,c=color)			
		if (len(X1)>0 and len(Y1)>0 and len(X1)== len(Y1)):
			plt.plot(X1,Y1,c="black")
		plt.savefig("Image/GoppertDSOM.png")
	
	def GrosApprentissage(self,nbEpi,nbIte,nbApprentissage,TypeModele,nbIteSave):
		nb =0
		for i in range(nbApprentissage):
			repertoire = TypeModele+"/Apprentissage"+str(nb)
			while (os.path.exists(repertoire)):
				nb +=1
				repertoire = TypeModele+"/Apprentissage"+str(nb)
			os.makedirs(repertoire)
			if(TypeModele=="Som"):
				model = SOM()
			if(TypeModele=="DSom"):
				model = DSOM()
			self.ApprentissageModeleSimulIteConstant(model,nbEpi*nbIte,nbIte,nbIteSave,repertoire)
	
	def GrosExploitation(self,nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave):
		if(TypeModele=="Som"):
			model = SOM()
		if(TypeModele=="DSom"):
			model = DSOM()
			
		for i in range (0,100):
			for j in range (100,101):
				rep = TypeModele+"/Apprentissage"+str(i)+"/"+str(j*nbIteSave)+"/"
				model.Q  	   = np.load(rep+"Q.npy")
				model.weights  = np.load(rep+"W.npy")
				List = np.array([])
				for l in range (100):
					X,Y,nbExec = self.ExploitationGoppertModel(nbEpiExec,model)
					List= np.append(List,np.array([nbExec]))
				print(i,j,List[0:10])
				data = np.asarray(np.array(List))
				np.save(rep+"resultatExecGop.npy", data)
				#self.affichageModel(model,rep,X=X,Y=Y)

	def GrapheResultat(self,nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave):
		data  = np.array([[[l for l in range(10)] for j in range(101)] for i in range(10)])
		#recuperation des data
		for i in range (nbApprentissage):
			for j in range (101):
				rep = TypeModele+"/Apprentissage"+str(i)+"/"+str(j*nbIteSave)+"/"
				data[i][j]= np.load(rep+"resultatExec.npy")[0:10]
				
		#concatenation + séparation point 500
		boxData = [[]for i in range(101)]
		histo   = []
		for i in range (nbApprentissage):
			for j in range(101):
				boxData[j] = np.concatenate((boxData[j], data[i][j]), axis=None)
				histo   = np.concatenate((histo, np.array([j for i in range(len(np.where(boxData[j] == 500)[0]))])), axis=None)
				boxData[j] = np.delete(boxData[j], np.where(boxData[j] == 500))
				
		fig = plt.figure()
		ax = fig.add_subplot(2, 1, 1)
		
		BoxName = [""]
		for i in range (nbApprentissage):
				BoxName.append(i)
		bp1 = ax.boxplot(boxData,sym="|")
		for box in bp1['boxes']:
			box.set(color='blue', linewidth=1)
			
		ax.set_xlim([-1,102])
		ax.set_ylim([0,500])
		ticks = [i*20 for i in range(0,6)]
		ticks[5]+=1
		ax.set_xticks(ticks)
		
		ax.set_xticklabels([i*nbIteSave*20 for i in range(0,6)]) 
		
		ax.set_ylabel("Nombre de pas")
		
		
		ax = fig.add_subplot(2, 1, 2)
		
		ax.set_xlim([-1,102])
		ax.set_xticks(ticks)
		ax.set_xticklabels([i*nbIteSave*20 for i in range(0,6)]) 
		
		ax.hist(histo, range = (0, 102), bins = 102, color = 'yellow',
            edgecolor = 'red')
		ax.set_xlabel("Nombre d'entrée présentés")
		ax.set_ylabel("Nombre d'évaluation échoué")
		plt.savefig(TypeModele)
		
	def GrapheResultatLast(self,nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave):
		data1  = np.array([[l for l in range(100)] for i in range(100)])
		#recuperation des data
		for i in range (nbApprentissage):
			rep = "Som/Apprentissage"+str(i)+"/"+str(100*nbIteSave)+"/"
			data1[i]= np.load(rep+"resultatExec.npy")
		#concatenation + séparation point 500
		boxData1 = []
		histo1   = []
		for i in range (nbApprentissage):
			boxData1 = np.concatenate((boxData1, data1[i]), axis=None)
			histo1   = np.concatenate((histo1, np.array([1 for i in range(len(np.where(boxData1 == 500)[0]))])), axis=None)
			boxData1 = np.delete(boxData1, np.where(boxData1 == 500))
		
		
		
		data2  = np.array([[l for l in range(100)] for i in range(100)])
		#recuperation des data
		for i in range (nbApprentissage):
			rep = "Som/Apprentissage"+str(i)+"/"+str(100*nbIteSave)+"/"
			data2[i]= np.load(rep+"resultatExecGop.npy")
		boxData2 = []
		histo2   = []		
		for i in range (nbApprentissage):
			boxData2 = np.concatenate((boxData2, data2[i]), axis=None)
			histo2   = np.concatenate((histo2, np.array([2 for i in range(len(np.where(boxData2 == 500)[0]))])), axis=None)
			boxData2 = np.delete(boxData2, np.where(boxData2 == 500))


		
				
		data3  = np.array([[l for l in range(100)] for i in range(100)])
		#recuperation des data
		for i in range (nbApprentissage):
			rep = "DSom/Apprentissage"+str(i)+"/"+str(100*nbIteSave)+"/"
			data3[i]= np.load(rep+"resultatExec.npy")
		boxData3 = []
		histo3   = []		
		for i in range (nbApprentissage):
			boxData3 = np.concatenate((boxData3, data3[i]), axis=None)
			histo3   = np.concatenate((histo3, np.array([3 for i in range(len(np.where(boxData3 == 500)[0]))])), axis=None)
			boxData3 = np.delete(boxData3, np.where(boxData3 == 500))		
		
		
		data4  = np.array([[l for l in range(100)] for i in range(100)])
		#recuperation des data
		for i in range (nbApprentissage):
			rep = "DSom/Apprentissage"+str(i)+"/"+str(100*nbIteSave)+"/"
			data4[i]= np.load(rep+"resultatExecGop.npy")		
		boxData4 = []
		histo4   = []		
		for i in range (nbApprentissage):
			boxData4 = np.concatenate((boxData4, data4[i]), axis=None)
			histo4   = np.concatenate((histo4, np.array([4 for i in range(len(np.where(boxData4 == 500)[0]))])), axis=None)
			boxData4 = np.delete(boxData4, np.where(boxData4 == 500))
			
		
		print(np.median(data1),np.median(data2),np.median(data3),np.median(data4))	
		print(len(histo1),len(histo2),len(histo3),len(histo4))
		fig = plt.figure()
		ax = fig.add_subplot(2, 1, 1)
		
		BoxName = [""]

		for i in range (2):
				BoxName.append(i)
				
		boxData = [boxData1,boxData2,boxData3,boxData4]
		bp1 = ax.boxplot(boxData,sym="|")
		
		for box in bp1['boxes']:
			box.set(color='blue', linewidth=1)
			
		ax.set_xlim([0,5])
		ax.set_ylim([0,500])
		ticks = [i for i in range(0,5)]
		ax.set_xticks(ticks)
		
		ax.set_xticklabels(["","SOMSans","SOMAvec","DSOMSans","DSOMAvec"]) 
		
		ax.set_ylabel("Nombre de pas")
		
		
		ax = fig.add_subplot(2, 1, 2)
		
		ax.set_xlim([0,5])
		ax.set_xticks(ticks)
		ax.set_xticklabels(["","SOMSans","SOMAvec","DSOMSans","DSOMAvec"]) 
		histo = np.concatenate((histo1, histo2,histo3, histo4), axis=None)
		ax.hist(histo, range = (0, 5), bins = 10, color = 'yellow',
            edgecolor = 'red')
		ax.set_ylim([0,10000])	
		ax.set_ylabel("Nombre d'évaluation échoué")
		plt.savefig("SAVE")	
		
	def histogramme(self,nbIte,nbApprentissage,TypeModele,nbIteSave):	
		if(TypeModele=="Som"):
			model = SOM()
		if(TypeModele=="DSom"):
			model = DSOM()
		histo  = []
		for i in range (nbApprentissage):
			rep = TypeModele+"/Apprentissage"+str(i)+"/"+str(100*nbIteSave)+"/"
			model.Q  	   = np.load(rep+"Q.npy")
			model.weights  = np.load(rep+"W.npy")
			for l in range(len(model.Voisinage.listVoisin)):
				n1 = model.Voisinage.listVoisin[l].n1
				n2 = model.Voisinage.listVoisin[l].n2
				histo.append(np.linalg.norm(model.weights[n1] - model.weights[n2]))
		fig = plt.figure()
		print("FIGURE",len(histo))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_ylim([0,6000])
		ax.hist(np.array(histo), range = (0, 0.4), bins = 40)
		plt.savefig("Histo"+TypeModele)	
	
	def histogramme3D(self,nbIte,nbApprentissage,TypeModele,nbIteSave,
		nbSave=101,interMin = 0, interMax = 0.4,nbInter  = 100):	
		if(TypeModele=="Som"):
			model = SOM()
		if(TypeModele=="DSom"):
			model = DSOM()
		histo  = [[] for i in range(nbSave)]
		for j in range (nbSave):
			print(j)
			for i in range (nbApprentissage):
				rep = TypeModele+"/Apprentissage"+str(i)+"/"+str(j*nbIteSave)+"/"
				model.Q  	   = np.load(rep+"Q.npy")
				model.weights  = np.load(rep+"W.npy")
				for l in range(len(model.Voisinage.listVoisin)):
					n1 = model.Voisinage.listVoisin[l].n1
					n2 = model.Voisinage.listVoisin[l].n2
					histo[j].append(np.linalg.norm(model.weights[n1] - model.weights[n2]))
		histoF  = [[0 for i in range(nbInter)] for i in range(nbSave)]
		for i in range(len(histo)):
			for j in range(len(histo[i])):
				for l in range (nbInter):
					if ((histo[i][j]<((interMax-interMin)/nbInter*l)) and (histo[i][j]>((interMax-interMin)/nbInter*(l-1)))):
						histoF[i][l] += 1/len(histo[i])
		
		
		fig = plt.figure()
		x, y = np.linspace(interMin, interMax, nbInter), np.linspace(0, 5000000, nbSave)
		X, Y = np.meshgrid(x, y)
		ax = Axes3D(fig)

		ax.set_ylim([0,5000000])
		ax.set_zlim([0,0.17])
		ax.plot_surface(X, Y, np.array(histoF))
		ax.set_xlabel("distance inter-neurones")
		ax.set_ylabel("Progression apprentissage")
		ax.set_zlabel("Pourcentage Repartition")
		surf = ax.plot_surface(X, Y, np.array(histoF), cmap=plt.cm.plasma)
		plt.savefig("Histo3D"+TypeModele)	
					
	def main(self):
		nbEpi 			= 1000
		nbIte 			= 5000
		nbEpiExec 		= 500
		nbIteSave       = 50000
		nbApprentissage = 100
		
		unit 			= True
		apprentissage 	= False
		exploitation  	= False
		graph			= False
		graph2			= False
		histo 			= False
		histo3D 		= False
		TypeModele		= "DSom"
		if(unit):

			if (self.load): #Chargement des apprentissages sauvegarder
				self.som.Q  = np.load(self.repertoire+"/"+self.nameFile+"SomQ.npy")
				self.som.weights  = np.load(self.repertoire+"/"+self.nameFile+"SomW.npy")		
				#self.dSom.Q  = np.load(self.repertoire+"/"+self.nameFile+"DSomQ.npy")
				#self.dSom.weights  = np.load(self.repertoire+"/"+self.nameFile+"DSomW.npy")		
			else: #Apprentissage des modèles
				self.nb = 0
				self.repertoire = "Apprentissage"+str(self.nb)
				while (os.path.exists(self.repertoire)):
					self.nb +=1
					self.repertoire = "Apprentissage"+str(self.nb)
				os.makedirs(self.repertoire)
				print("Debut apprentissage")
				#DSOM
				self.ApprentissageDSom2(nbEpi,nbIte)
				#Som
				#self.ApprentissageSom(nbEpi,nbIte)
				
			#ExploitationModèle
			self.XDSOM,self.YDSOM = self.ExploitationDSom(nbEpiExec)	
			#self.XSOM,self.YSOM = self.ExploitationSom(nbEpiExec)	
			self.affichageDSom(nbEpi,X=self.XDSOM,Y=self.YDSOM)
			#self.affichageSom(nbEpi,X=self.XSOM,Y=self.YSOM)
			
			if self.save:
				#data = np.asarray(self.som.Q)
				#np.save(self.repertoire+"/"+self.nameFile+"SOMQ.npy", data)
				#data = np.asarray(self.som.weights)
				#np.save(self.repertoire+"/"+self.nameFile+"SOMW.npy", data)
				data = np.asarray(self.dSom.Q)
				np.save(self.repertoire+"/"+self.nameFile+"DSOMQ.npy", data)
				data = np.asarray(self.dSom.weights)
				np.save(self.repertoire+"/"+self.nameFile+"DSOMW.npy", data)
				self.affichageDSom(nbEpi,X=self.XDSOM,Y=self.YDSOM)
				#self.affichageSom(nbEpi,X=self.XSOM,Y=self.YSOM)
	
		if(apprentissage):
			self.GrosApprentissage(nbEpi,nbIte,nbApprentissage,TypeModele,nbIteSave)
		if(exploitation):	
			self.GrosExploitation(nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave)
		if(graph):
			self.GrapheResultat(nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave)
		if(graph2):
			self.GrapheResultatLast(nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave)
		if(histo):
			self.histogramme(nbIte,nbApprentissage,TypeModele,nbIteSave)	
		if(histo3D):
			self.histogramme3D(nbIte,nbApprentissage,TypeModele,nbIteSave)
		#DISCRETE	
		if(False):	
			#self.Apprentissage(nbEpi,nbIte)
			XD,YD = self.ExploitationDiscret(self.Q,nbIte)
			self.affichageDiscret(self.Q,X=XD,Y=YD)
			
		"""
		self.dSom.Q  = np.load("Apprentissage"+"1/"+self.nameFile+"DSomQ.npy")
		self.dSom.weights  = np.load("Apprentissage"+"1/"+self.nameFile+"DSomW.npy")	
		self.XSOM,self.YSOM = self.ExploitationDSom(nbEpiExec)
		self.affichageDSom(5000,X=self.XSOM,Y=self.YSOM)
		"""
		
		"""
		#Spec Apprentissage
		som = SOM()
		self.som.Q = np.load("ApprentissageSOM2phase/MountainCarSOMQ.npy")
		self.som.weights = np.load("ApprentissageSOM2phase/MountainCarSOMW.npy")
		x,y = self.ExploitationGoppertSom(500)
		self.affichageGoppertSom(X1=x,Y1=y)
		"""
		self.env.close()
		



now = datetime.now()

current_time1 = now.strftime("%H:%M:%S:%f")
print("Current Time =", current_time1)

m = MountainCar()

now = datetime.now()

current_time = now.strftime("%H:%M:%S:%f")
print("Start Time =", current_time1)
print("Current Time =", current_time)







#apprentissage placement en même temps q table
#courbe performance
#histogramme distance neurones
#méthode de Gibbs

#mercredi 29 14h
#mercredi  5 14h
