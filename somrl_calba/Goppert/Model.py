from Donnee import *
from Voisinage import *
from Neurones import *
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter
import argparse


#Classe modèle
class Model:

	def __init__(self,nbD):
		self.Donnee = Donnee()
		self.weights = []
		if(nbD == 0):
			self.Donnee.setDistributionUniform()
		if(nbD == 1):
			self.Donnee.setDistributionAnneauUniform()
		if(nbD == 2):
			self.Donnee.setDistributionUniformMouvent()
		if(nbD == 3):
		#CERCLE DENSITE UNIFORM
			self.Donnee.setDistributionAnneauUniform(distMin = 0.0, N = 3000)
		if(nbD == 4):
			self.Donnee.setDistributionDisqueDensiteIn()
		if(nbD == 5):
			self.Donnee.setDistributionDisqueDensiteOu()
		if(nbD == 6):	
			self.Donnee.setDistributionDeuxGroupeRec()
		self.Voisinage = Voisinage()

	def init(self):
		return self.proto, self.text;

	def saveImage(self,Donne,N):
		self.Donnee = Donne
		for i in range(0,N):
			x = self.Donnee.getVecteurEntre()
			self.epochs(i, x)
			
		fig = plt.figure(facecolor='white')
		ax = fig.add_subplot(111)
		#Fixation de la fenetre
		ax.set_xlim(-0.6,0.6)
		ax.set_ylim(-0.6,0.6)
		
		#Affichage des données
		ax.scatter(self.Donnee.Data[:, 0], self.Donnee.Data[:, 1], s=4, c= "blue")
		#plt.savefig("T"+self.Donnee.type+"Donne")
		#Affichage du point selectionner
		ax.scatter(x[0],x[1], s = 4, c= "blue")
		ax.set_aspect(aspect='equal')
		#Affichage des liens de voisinage
		for i in range(len(self.Voisinage.listVoisin)):
			n1 = self.Voisinage.listVoisin[i].n1
			n2 = self.Voisinage.listVoisin[i].n2
			ax.plot([self.weights[n1][0], self.weights[n2][0]], [self.weights[n1][1], self.weights[n2][1]], 'm-')
			
		#Affichage des positions des neurones
		ax.scatter(self.weights[:,0],self.weights[:,1], s = 50, c= "red")
		
		#Box avec le nombre de frame
		textstr = 'Sample %d \n Distortion %f \n Neurones %d' % (N, self.calculeDistortion(), len(self.weights)) 
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		#ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
		plt.title(self.titre)
		plt.savefig("T"+self.Donnee.type+"_"+self.titre)	
			
	def updatefig(self,frame):
		N = 50
		if((frame*N)%(self.tFinal/4)==0):
			self.Donnee.deplacement()

		for i in range(0,N):
			x = self.Donnee.getVecteurEntre()
			self.epochs(50*frame+i, x)
		print(frame*N)
		
		x = self.Donnee.getVecteurEntre()
		self.epochs(frame, x)
		#self.proto.set_offsets(self.weights)
		
		self.ax.clear()
		#Fixation de la fenetre
		self.ax.set_xlim(-0.6,0.6)
		self.ax.set_ylim(-0.6,0.6)
		
		#Affichage des données
		self.ax.scatter(self.Donnee.Data[:, 0], self.Donnee.Data[:, 1], s=4)
		
		#Affichage du point selectionner
		#self.ax.scatter(x[0],x[1], s = 20, c= "black")
		
		#Affichage des liens de voisinage
		for i in range(len(self.Voisinage.listVoisin)):
			n1 = self.Voisinage.listVoisin[i].n1
			n2 = self.Voisinage.listVoisin[i].n2
			self.ax.plot([self.weights[n1][0], self.weights[n2][0]], [self.weights[n1][1], self.weights[n2][1]], 'g-')
			
		#Affichage des positions des neurones
		self.ax.scatter(self.weights[:,0],self.weights[:,1], s = 20, c= "red")
		
		#Box avec le nombre de frame
		textstr = 'Sample %d' % (frame*N, )
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
		if(50*frame+i>999):
			plt.savefig("TEST")
		
		return self.proto, self.text, self.ax
		
	def affichage(self, parser, t, tf):
		self.tFinal = tf
		parser.add_argument('--movie',
		help='Should we generate a movie',
					action='store_true')
		parser.add_argument('--fps', type=int, default=25,
					help='Frame rate')
		parser.add_argument('--frames', type=int, default=int(self.tFinal/50),
                    help='Number of frames')
		args = parser.parse_args()

		# Extract the arguments
		save_movie = t
		frames = args.frames
		fps = args.fps
		# Setup the plot
		if save_movie:
			matplotlib.use("Agg")
		fig = plt.figure(facecolor='white')

		self.ax = fig.add_subplot(111)
		self.ax.scatter(self.Donnee.Data[:, 0], self.Donnee.Data[:, 1], s=4)
		self.proto = self.ax.scatter(self.weights[:, 0], self.weights[:, 1], s=20,
                       animated=True)
		plt.xlim(-1.0, 1.0)
		plt.ylim(-1.0, 1.0)
		self.text = self.ax.annotate('0',
                       xy=(-0.95, -0.95),
                       fontsize=10)

		plt.gca().set_aspect('equal')
		plt.title(self.titre)
		ani = anim.FuncAnimation(fig, self.updatefig,  init_func=self.init, frames=frames, interval=1000/fps)
		if save_movie:
			ani.save(self.titre+"_"+self.Donnee.type+'.mp4', writer='ffmpeg')
		else:
			plt.show()

	def calculeDistortion(self):
		sum = 0
		for i in range( self.Donnee.N):
			dist = np.linalg.norm(self.weights - self.Donnee.Data[i], axis=1)
			sum +=  np.min(dist)**2

		return sum/self.Donnee.N

