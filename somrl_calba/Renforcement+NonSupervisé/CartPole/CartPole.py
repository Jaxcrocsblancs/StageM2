import gym
import numpy as np
import matplotlib.pyplot as plt
from Som import *
from DSOM import *
from datetime import datetime
from numpy import sin, cos, arccos
import os
env = gym.make("CartPole-v1")
TypeModele = "SOM"
defaultRepertoire = "SaveApprentissage/"
listeMode = ["Simultaner","Alterner","UnApresLAutre"]
listePolitique = ["Gop","Aleatoire","Greedy"]

def CartePoleSom():
	return SOM(dim = 4,lim=[[-4.8,4.8],[-0.1,0.1],[-0.418,0.418],[-0.1,0.1]], nbAction=2, lon=20, lar=20)
	
def CartePoleDSom():
	return DSOM(dim = 4,lim=[[-4.8,4.8],[-0.1,0.1],[-0.418,0.418],[-0.1,0.1]], nbAction=2)

def CartePoleTest(observation):
	return bool()

def maxAction(Q):
			idMax  = 0
			valMax = -100000
			for i in range(len(Q)):
				valCur = Q[i]
				if(valMax<valCur):
					idMax  = i
					valMax = valCur
			return idMax
			
def ApprentissageSimul(nbEpi,nbIte,m):
		
		eps = 0.02
		m.tFinal = nbEpi*nbIte
		
		alphaInit 			= 0.1
		alphaFinal			= 0.001	
		gamma 				= 0.9
		for i_episode in range(nbEpi):
			observation = env.reset()
			posSom = m.getBMU(observation)
			ansPosSom = posSom
			t = 0
			#Alpha constant
			alphaCurrent = alphaInit * (alphaFinal/alphaInit)**(i_episode/nbEpi)
			for t in range(nbIte):
				#Possibilité d'aleatoire
				if np.random.uniform(0, 1) < eps:
					action = env.action_space.sample()
				else:
					#Pro
					logits = m.Q[posSom]
					logits_exp = np.exp(logits)
					probs = logits_exp / np.sum(logits_exp)
					action = np.random.choice(env.action_space.n, p=probs)
					
				observation, reward, done, info = env.step(action)
				
				posSom = m.getBMU(observation)
				m.epochs(i_episode*nbIte+t,observation)
				m.Q[ansPosSom][action] = m.Q[ansPosSom][action] + alphaCurrent * (reward + gamma*m.Q[posSom][maxAction(m.Q[posSom])]-m.Q[ansPosSom][action])
				ansPosSom = posSom
				#Arret si résolue
				if done:
					break
			now = datetime.now()
			current_tim = now.strftime("%H:%M:%S:%f")
			print(i_episode,t,current_tim)	
		return m

def MajModel(modele,posSom,observation ,ansPosSom,gamma,temps,action,tCarte,alphaCurrent,reward,mode="Simultaner",freqAlter = [40,60]):

	if (mode=="Simultaner"):
		modele.epochs(temps, observation)
		modele.Q[ansPosSom][action] = modele.Q[ansPosSom][action] + alphaCurrent * (reward + gamma * modele.Q[posSom][maxAction(modele.Q[posSom])] - modele.Q[ansPosSom][action])
		politique = "Gop"
	elif (mode=="Alterner"):
		if(temps%(sum(freqAlter))<freqAlter[0]):
			modele.epochs(temps, observation)
			politique = "Aleatoire"
		else:
			modele.Q[ansPosSom][action] = modele.Q[ansPosSom][action] + alphaCurrent * (reward + gamma * modele.Q[posSom][maxAction(modele.Q[posSom])] - modele.Q[ansPosSom][action])
			politique = "Gop"
	elif (mode=="UnApresLAutre"):
		if (temps<tCarte):
			modele.epochs(temps, observation)
			politique = "Aleatoire"
		else:
			modele.Q[ansPosSom][action] = modele.Q[ansPosSom][action] + alphaCurrent * (reward + gamma * modele.Q[posSom][maxAction(modele.Q[posSom])] - modele.Q[ansPosSom][action])
			politique = "Gop"
	ansPosSom = posSom
	return modele, ansPosSom, politique

def Politique(model,posSom,politique="Gop",eps = 0.02):
	if (politique=="Gop"):
		# Possibilité d'aleatoire
		if np.random.uniform(0, 1) < eps:
			action = env.action_space.sample()
		else:
			# Pro
			logits = model.Q[posSom]
			logits_exp = np.exp(logits)
			probs = logits_exp / np.sum(logits_exp)
			action = np.random.choice(env.action_space.n, p=probs)
	elif (politique=="Aleatoire"):
		action = env.action_space.sample()
	elif (politique=="Greedy"):
		action = maxAction(model.Q[posSom])
	return action

def ApprentissageModeleSimulIteConstant(model,nbIteMax,nbIte,nbFrequenceSave,repertoire,mode,alphaInit = 0.1,alphaFinal = 0.001,gamma = 0.9):
		model.tFinal = nbIteMax
		cpt = 0
		nbepi = 0
		alphaCurrent = alphaInit 
		while (cpt<nbIteMax):
			observation = env.reset()
			posSom = model.getBMU(observation)
			ansPosSom = posSom
			politique = "Aleatoire"
			#Alpha constant
			for t in range(nbIte):
				alphaCurrent = alphaCurrent * (alphaFinal/alphaInit)**(1/nbIteMax)
				posSom = model.getBMU(observation)

				action = Politique(posSom=posSom,politique=politique,model = model)

				observation, reward, done, info = env.step(action)

				model, ansPosSom,politique = MajModel(mode=mode,posSom=posSom,modele=model,alphaCurrent=alphaCurrent,reward=reward,observation=observation,ansPosSom=ansPosSom,gamma=gamma,action=action,temps=cpt,tCarte=(nbIteMax/3))

				if(cpt%nbFrequenceSave == 0):
					chemin = repertoire+"/"+str(cpt)
					if(not(os.path.exists(chemin))):
						os.makedirs(chemin)
					data = np.asarray(model.Q)
					np.save(chemin+"/Q.npy", data)
					data = np.asarray(model.weights)
					np.save(chemin+"/W.npy", data)

				#Arret si résolue
				if done:
					break;

				if (cpt>=nbIteMax):
					break;
				cpt += 1
				now = datetime.now()
				current_tim = now.strftime("%H:%M:%S:%f")
				if(cpt %50000==0):
					print(nbepi, cpt,current_tim)	
			nbepi += 1 

def Exploitation(nbIte,m):
		#print("===========Exploitation=========")	
		observation = env.reset()
		X = []
		Y = []
		rep = nbIte
		for t in range(0,nbIte):
			#env.render()
			pos = m.getBMU(observation)
			action = Politique(politique="Greedy",model=m,posSom=pos)
			
			observation, reward, done, info = env.step(action)
			if done:
				#print(t)
				rep = t
				break;		
		return rep
		
#Apprentissage unitaire
def ApprentissageUnitaire():
	nbEpi			= 100000
	nbIte 			= 5000
	nbExec 			= 500
	nbEva			= 100
	#Init
	if(TypeModele == "DSOM"):
		m = CartePoleDSom()
	if(TypeModele == "SOM"):
		m = CartePoleSom()
	
	#Apprentissage 
	m = ApprentissageSimul(nbEpi,nbIte,m)
	r = []
	#Exploitation
	for i in range (nbEva):
		r.append(Exploitation(nbExec,m))
	print(r)


	#Save
	"""
	repertoire = defaultRepertoire+"/ApprentissageDSOM0"
	nb = 0
	while (os.path.exists(repertoire)):
		nb +=1
		repertoire = defaultRepertoire+"/ApprentissageDSOM"+str(nb)
	os.makedirs(repertoire)
	data = np.asarray(m.Q)
	np.save(repertoire+"/Q.npy", data)
	data = np.asarray(m.weights)
	np.save(repertoire+"/W.npy", data)
	data = np.asarray(r)
	np.save(repertoire+"/r.npy", data)
	"""

def BigApprentissage(nbIteMax = 5000000, nbIte = 5000,nbIteSave = 50000, nbApprentissage = 100,mode="UnApresLAutre"):
	nb =0
	for i in range(nbApprentissage):
		repertoire = defaultRepertoire+TypeModele+mode+str(nbIteMax)+"ite"+"/Apprentissage"+str(nb)
		while (os.path.exists(repertoire)):
			nb +=1
			repertoire = defaultRepertoire+TypeModele+mode+str(nbIteMax)+"ite"+"/Apprentissage"+str(nb)
		os.makedirs(repertoire)
		if(TypeModele=="SOM"):
			model = CartePoleSom()
		if(TypeModele=="DSOM"):
			model = CartePoleDSom()
		ApprentissageModeleSimulIteConstant(model,nbIteMax,nbIte,nbIteSave,repertoire,mode=mode)

def GrosExploitation(nbApprentissage,nbEpiExec,TypeModele,nbIteSave,mode="Simultaner",nbIteMax=5000000):
		if(TypeModele=="SOM"):
			model = CartePoleSom()
		if(TypeModele=="DSOM"):
			model = CartePoleDSom()
			
		for i in range (0,100):
			for j in range (0,int(((nbIteMax/nbIteSave)+1))):
				rep = defaultRepertoire+TypeModele+mode+str(nbIteMax)+"ite"+"/Apprentissage"+str(i)+"/"+str(j*nbIteSave)+"/"
				model.Q  	   = np.load(rep+"Q.npy")
				model.weights  = np.load(rep+"W.npy")
				List = np.array([])
				for l in range (0,100):
					nbExec = Exploitation(nbEpiExec,model)
					List= np.append(List,np.array([nbExec]))
				print(i,j,List[0:10])
				data = np.asarray(np.array(List))
				np.save(rep+"resultatExec.npy", data)

def GrapheResultat(nbIte,nbApprentissage,nbEpiExec,TypeModele,nbIteSave,mode="Simultaner",nbIteMax=5000000):
		nbsave = int(((nbIteMax/nbIteSave)+1))
		if nbsave==51:
			nbtick = 10
		elif nbsave==101:
			nbtick = 20

		data  = np.array([[[l for l in range(100)] for j in range(nbsave)] for i in range(100)])
		#recuperation des data
		for i in range (nbApprentissage):
			for j in range (nbsave):
				rep =   defaultRepertoire+TypeModele+mode+str(nbIteMax)+"ite"+"/Apprentissage"+str(i)+"/"+str(j*nbIteSave)+"/"
				data[i][j]= np.load(rep+"resultatExec.npy")
				
		#concatenation + séparation point 500
		boxData = [[]for i in range(nbsave)]
		histo   = []
		for i in range (nbApprentissage):
			for j in range(nbsave):
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
			
		ax.set_xlim([-1,nbsave+1])
		ax.set_ylim([0,500])
		ticks = [i*nbtick for i in range(0,6)]
		ticks[5]+=1
		ax.set_xticks(ticks)
		
		ax.set_xticklabels([i*nbIteSave*nbtick for i in range(0,6)])
		
		ax.set_ylabel("Nombre de pas")
		
		
		ax = fig.add_subplot(2, 1, 2)
		
		ax.set_xlim([-1,nbsave+1])
		ax.set_xticks(ticks)
		ax.set_xticklabels([i*nbIteSave*nbtick for i in range(0,6)])
		
		ax.hist(histo, range = (0, nbsave+1), bins = nbsave+1, color = 'yellow',
            edgecolor = 'red')
		ax.set_xlabel("Nombre d'entrée présentés")
		ax.set_ylabel("Nombre d'évaluation échoué")
		plt.savefig(TypeModele+mode+str(nbIteMax))
		
"""		
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
"""	

ApprentissageUnitaire()
#BigApprentissage()
#GrosExploitation(100,500,TypeModele,50000)
#GrapheResultat(5000,100,500,TypeModele,50000)
env.close()	
