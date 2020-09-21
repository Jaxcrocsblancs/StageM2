import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.lines as mlines
#Seulement pour une dimension
#TODO généralisé pour toutes dimension
def YISOM1D(tab,X):
	#Recuperation du neurones le plus proches
	dist = np.absolute(tab - X)
	#Recuperation de l'ordre de chaque id
	#bmu_id = np.argmin(dist[:,0])
	trie = np.argsort(dist[:,0])
	#Recuperation de l'ordre de chaque id
	W0 = tab[trie[0]]
	W1 = tab[trie[1]]
	lin = W1[0]-W0[0]

	alpha = (X-W0[0])/lin
	lou = W1[1]-W0[1]
	Y = W0[1]+lou*alpha
	return [Y]
	
def PlusprocheVoisin(tab,X):
	dist = np.absolute(tab - X)
	bmu_id = np.argmin(dist[:,0])
	return tab[bmu_id][1]
	
def SecondPlusprocheVoisin(tab,X):
	dist = np.absolute(tab - X)
	bmu_id = np.argmin(dist[:,0])
	#L'ordre des id pour trier
	trie = np.argsort(dist[:,0])
	return tab[trie[1]][0]
	
	
def shepard1D(tab,X):
	#Calcule des Dj (distance entre le point support et X)
	tabDj = (tab[:,0]-X)
	dist = np.absolute(tab - X)
	#Recuperation de l'ordre de chaque id
	#bmu_id = np.argmin(dist[:,0])
	trie = np.argsort(dist[:,0])
	#Test si on est pas sur un point support
	if(not(0 in tabDj)):   
		tempo = tabDj**-2
		SumDj = tempo[trie[0]]
		if(trie[0]-1>=0):
			SumDj += tempo[trie[0]-1]
		if(trie[0]+1<len(tab)):
			SumDj += tempo[trie[0]+1]
		tabW = tempo/SumDj
		F = YISOM1D(tab,X)
		F = F*tabW
		SumF = F[trie[0]]
		
		return SumF
	else:
		return YISOM1D(tab,X)[0]
	
def test1gooppert1D(tab,X):
	if(X in tab[:,0]):
		#print("test", tab[np.where(tab[:,0] == X)[0]][0][1])
		return  tab[np.where(tab[:,0] == X)[0]][0][1]
	#Recuperation du neurones le plus proches
	dist = np.absolute(tab - X)
	#Recuperation de l'ordre de chaque id
	#bmu_id = np.argmin(dist[:,0])
	trie = np.argsort(dist[:,0])
	
	#Recuperation du neurone le plus proche et de ses voisins
	W0 = tab[trie[0]]
	
	if(trie[0]-1>=0):
		Wm = tab[trie[0]-1]
	else:
		Wm = W0
	if(trie[0]+1<len(tab)):
		Wp = tab[trie[0]+1]
	else:
		Wp = W0	
	tabDj = (tab[:,0]-X)
	#print(tabDj)
	tempo = tabDj**-2
	#print(tempo)
	SumDj = tempo[trie[0]]
	if(trie[0]-1>=0):
		SumDj += tempo[trie[0]-1]
	if(trie[0]+1<len(tab)):
		SumDj += tempo[trie[0]+1]
	SF = tempo[trie[0]]/SumDj
	lin = SF*(Wp[0] - W0[0])- SF*(Wm[0] - W0[0])

	alpha = (X-W0[0])/lin
	lou = SF*(Wp[1] - W0[1])- SF*(Wm[1] - W0[1])
	Y = W0[1]+lou*alpha
	Y = Y *SF
	return Y


#Retourne la valeur YiI-SOM pour un neurones i donnée
def YiISOM1D(tab,X,i):
	if(X in tab[:,0]):
		if(tab[i][0] == X):
			return tab[i][1]
		else:
			return 0

	#Recuperation di-, di et di+
	di = tab[i]
	Dj = []
	if(i-1>=0):
		dim = tab[i-1]
		Dj.append(i-1)
		im = i-1
	else:
		dim = di
		im = i
	if(i+1<len(tab)):
		dip = tab[i+1]
		Dj.append(i+1)
		ip = 1+i
	else:
		dip = di
		ip = i
	
	SFm = PHIiSF(tab,X,im,Dj)
	SFp = PHIiSF(tab,X,ip,Dj)
	lin = SFp*(dip[0] - di[0])- SFm*(dim[0] - di[0])
	alpha = (X-di[0])/lin
	#print(alpha,X,di[0],lin)
	lou = SFp*(dip[1] - di[1])- SFm*(dim[1] - di[1])
	Y = di[1]+lou*alpha
	return Y
	
#Retourne la valeur de PHIiSf selon l'ensemble d'indice Dj
def PHIiSF(tab,X,i,Dj):
	#print("X ",X," i ",i)
	#print("PHIiSF")
	tabDj = tab[:,0]
	#print("tabDj",tabDj)
	tabDj = np.absolute((tabDj - X))
	#On est pas sur un point support
	if(not(0 in tabDj)): 
		#print("tabDj2",tabDj)
		tabDj = tabDj**-2
		sousTabDj = tabDj[Dj]
		sumSousTabDj = np.sum(sousTabDj)
		return tabDj[i]/sumSousTabDj
	else:
		#i est le point support
		if (tabDj[i]==0):
			return 1
		else:
			return 0

def PHIiLRSF(tab,X,i,Dj):
	#print()
	#print(X)
	tempo = tab[1:len(tab),0]-tab[0:-1,0]
	R = np.sum(tempo)/len(tempo) *2
	#print(tempo, R)
	tabDj = tab[:,0]
	tabDj = np.absolute((tabDj - X))
	if(not(0 in tabDj)): 
	#print(tabDj)
		tabRmDJp = R-tabDj
		#print(tabRmDJp)
		tabRmDJp[tabRmDJp<0] = 0 
		#print(tabRmDJp)
		tabRxDJ = R * tabDj
		tabRmDJpdRxDJ = (tabRmDJp/tabRxDJ)**2
		Sum = np.sum(tabRmDJpdRxDJ)
		return tabRmDJpdRxDJ[i]/Sum
	else:
		#i est le point support
		if (tabDj[i]==0):
			return 1
		else:
			return 0
			
#Retourne la valeur final 	
def YCFSOM(tab,X,LR):
	#print()
	tabYiISOM = np.array([YiISOM1D(tab,X,i) for i in range(len(tab))])
	#print(tabYiISOM)
	tabDj = np.array([i for i in range(len(tab))])
	if(LR):
		tabPHIiLRSF = np.array([PHIiLRSF(tab,X,i,tabDj) for i in range(len(tab))])
		return np.sum(np.multiply(tabPHIiLRSF,tabYiISOM))
	else:
		tabPHIiSF = np.array([PHIiSF(tab,X,i,tabDj) for i in range(len(tab))])
		return np.sum(np.multiply(tabPHIiSF,tabYiISOM))
	#print(tabPHIiSF)
	
def testGausienne():
	x_min = -3.0
	x_max = 12.0

	mean = 2.0 
	std = 2.0

	x = np.linspace(x_min, x_max, 10)
	y = scipy.stats.norm.pdf(x,mean,std)


	plt.xlim(x_min,x_max)
	plt.ylim(0,0.25)
	y1=[]
	x1=[]
	tab = np.array((x,y)).T
	for i in range (-300,1200):
		X = i*0.01
		x1.append(X)
		y1.append([YCFSOM(tab,X,False)])
	plt.plot(x1,y1,c="red")
	plt.scatter(tab[:, 0], tab[:, 1], s=50,marker='X',c="red" )
	
	x = np.linspace(x_min, x_max, 20)
	y = scipy.stats.norm.pdf(x,mean,std)
	y1=[]
	x1=[]
	tab = np.array((x,y)).T
	for i in range (-300,1200):
		X = i*0.01
		x1.append(X)
		y1.append([YCFSOM(tab,X,True)])
	plt.plot(x1,y1,c="green" )
	plt.scatter(tab[:, 0], tab[:, 1], s=50,marker='X',c="green" )
	x = np.linspace(x_min, x_max, 1000)
	y = scipy.stats.norm.pdf(x,mean,std)
	plt.plot(x,y)
	
	
	
	plt.savefig("normal_distribution2.png")
	plt.show()

#testGausienne()


def linar(tab,X):
	Y=[]
	for i in range(len(tab)-1):
		W0 = tab[i]
		W1 = tab[i+1]
		lin = W1[0]-W0[0]

		alpha = (X-W0[0])/lin
		lou = W1[1]-W0[1]
		Y.append(W0[1]+lou*alpha)
	return Y
		
		
		
def testmain():		
	fig = plt.figure(facecolor='white')

	tab = np.array([[0,0],[1,0],[4.5,1],[5.5,-1],[9,0],[10,-3]])
	ax = fig.add_subplot(1,1,1)
	plt.xlim(-3.0, 12.0)
	plt.ylim(-6.0, 6.0)



	x=[]
	y=[]
	Z=[]
	for i in range (-300,1200):
		X = i*0.01
		x.append(X)
		Z.append(YISOM1D(tab,X))
		#y.append(SecondPlusprocheVoisin(tab,X))
		y.append(linar(tab,X))
		#y.append([YISOM1D(tab,X)[0],YCFSOM(tab,X,False),YCFSOM(tab,X,True),test1gooppert1D(tab,X),shepard1D(tab,X)])
		
		
		
	lc = ['b','g','r','c','m','y']

	print(lc)
	ax.plot(x,y)
	#ax.plot(x,Z,'k')
	plt.title("I-SOM")


	textstr = '\n'.join((
		r'1*: Plus proche voisin' ,
		r'2*: 2eme Plus proche voisin' ,
		r'3*: Couple 1ere et 2eme plus proche voisin' ))
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
			verticalalignment='top', bbox=props)
			
			
			

	for i in range (len(tab)):
		ax.scatter(tab[i][0], tab[i][1], s=100,marker='X',c=lc[i] )
	#Plus proche voisin
	ax = fig.add_subplot(20,1,18)
	ax.barh(range(1), [15], color = 'y')
	ax.barh(range(1), [12.5], color = 'm')
	ax.barh(range(1), [10.25], color = 'c')
	ax.barh(range(1), [8], color = 'r')
	ax.barh(range(1), [5.75], color = 'g')
	ax.barh(range(1), [3.5], color = 'b')


	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.set_xlim(0.0,15.0)
	ax.text(15.5, 0,"1*", horizontalalignment='center', verticalalignment='center')
	#Second plus proche voisin
	ax = fig.add_subplot(20,1,19)
	ax.barh(range(1), [15], color = 'm')
	ax.barh(range(1), [12.5], color = 'y')
	ax.barh(range(1), [10.75], color = 'c')
	ax.barh(range(1), [10.25], color = 'm')
	ax.barh(range(1), [10], color = 'r')
	ax.barh(range(1), [8], color = 'c')
	ax.barh(range(1), [6.25], color = 'g')
	ax.barh(range(1), [5.75], color = 'r')
	ax.barh(range(1), [5.25], color = 'b')
	ax.barh(range(1), [3.5], color = 'g')


	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.set_xlim(0.0,15.0)
	prop_cycle = plt.rcParams['axes.prop_cycle']
	lc = prop_cycle.by_key()['color']
	ax.text(15.5, 0,"2*", horizontalalignment='center', verticalalignment='center')
	#Choix
	ax = fig.add_subplot(20,1,20)
	ax.barh(range(1), [15], color = lc[4])
	ax.barh(range(1), [10.75], color = lc[3])
	ax.barh(range(1), [10], color = lc[2])
	ax.barh(range(1), [6.25], color = lc[1])
	ax.barh(range(1), [5.25], color = lc[0])



	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.set_xlim(0.0,15.0)
	ax.text(15.5, 0, "3*", horizontalalignment='center', verticalalignment='center')
	plt.show()
	fig.savefig("I-SOM")	


def plot():
	fig = plt.figure(facecolor='white')

	tab = np.array([[0,0],[1,0],[4.5,1],[5.5,-1],[9,0],[10,-3]])
	ax = fig.add_subplot(1,1,1)
	plt.xlim(-30.0, 120.0)
	plt.ylim(-60.0, 60.0)



	x=[]
	y=[]
	z=[]
	for i in range (-3000,12000):
		X = i*0.01
		x.append(X)
		y.append([YCFSOM(tab,X,False)])
	ax.plot(x,y)	
	for i in range (len(tab)):
		ax.scatter(tab[i][0], tab[i][1], s=100,marker='X', c = "red")
	
	Global = mlines.Line2D([], [], color='tab:blue',  markersize=10, label='Global')
	Local  = mlines.Line2D([], [], color='tab:orange',markersize=10, label='Local')
			  
	plt.legend(handles=[Global,Local],loc = 'upper right')
			
	#plt.show()
	fig.savefig("YCFSOM1")	

def plotphi():
	fig = plt.figure(facecolor='white')

	supp  = np.array([[0,0],[1,0],[4.5,1],[5.5,-1],[9,0],[10,-3]])
	tabV = np.array([np.array([[k-1 if k-1>=0 else -1 , k+1 if k+1<len(supp)  else -1]]) for k in range (len(supp))])
	plt.xlim(-3.0, 12.0)

	ax = fig.add_subplot(111)




	x=[]
	y=[]
	for i in range (-300,1200):
		X = np.array(i*0.01)
		x.append(X)
		tabDj = np.array([i for i in range(len(supp))])
		tabPHIiSF = np.array([PHIiSF(supp,X,i,tabDj) for i in range(len(supp))])
		y.append(tabPHIiSF)
	plt.plot(x,y)

	tabDj = np.array([0 for i in range(len(supp))])
	lc = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467db','#8c564b']
	for i in range (len(supp)):
		ax.scatter(supp[i][0], 1, s=100,marker='X',c=lc[i] )
	#ax.scatter(supp[:, 0], tabDj, s=50,marker='X',c="red" )
	plt.savefig("Phi")
	#plt.show()		
	
	
def plotphi2():
	fig = plt.figure(facecolor='white')

	supp  = np.array([[0,0],[1,0],[4.5,1],[5.5,-1],[9,0],[10,-3]])
	tabV = np.array([np.array([[k-1 if k-1>=0 else -1 , k+1 if k+1<len(supp)  else -1]]) for k in range (len(supp))])
	plt.xlim(-3.0, 12.0)

	ax = fig.add_subplot(111)

	x=[]
	y=[]
	for i in range (-300,1200):
		X = np.array(i*0.01)
		x.append(X)
		tabDj = np.array([i for i in range(len(supp))])
		tabPHIiSF = np.array([PHIiLRSF(supp,X,i,tabDj) for i in range(len(supp))])
		y.append(tabPHIiSF)
	plt.plot(x,y)

	tabDj = np.array([0 for i in range(len(supp))])
	lc = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467db','#8c564b']
	for i in range (len(supp)):
		ax.scatter(supp[i, 0], 1, s=100,marker='X',c=lc[i] )
	plt.savefig("PhiLR")
	#plt.show()		
#plotphi2()
#plotphi()
plot()