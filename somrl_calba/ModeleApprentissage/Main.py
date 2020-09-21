from Model import *
from SOM import *
from DSOM import *
from DSOMA import *
from NeuralGas import *
from GrowingNeuralGas import *
import argparse

#Choix du modele
modele  = "SOM"

#Choix repartition des données
#0 repartition uniforme rectangulaire
#1 repartition uniforme annulaire
#2 repartition uniforme rectangulaire mouvante à 4 emplacement
#3 repartition uniforme circulaire
#4 disque avec densité forte au centre
#5 disque avec densité faible au centre
#6 2 groupe rectangulaire

nbD = 1
Donnee = Donnee()
#Donnee.setDistributionUniform(N = 2000)
Donnee.setDistributionAnneauUniform(distMin = 0.0, distMax = 0.5, N = 1000)
Donnee.maxDistDonnee()
N = 10000

# Sample some points for drawing
parser = argparse.ArgumentParser()
    
if(modele == "SOM"):
        m = SOM(nbD)
if(modele == "NG"):  
        m = NG(nbD)
if(modele == "GNG"):  
        m = GNG(nbD)
if(modele == "DSOM"):  
        m = DSOM(nbD)

"""
m = SOM(nbD , tFinal =N)
m.saveImage(Donnee,N)
m = NG(nbD, tFinal =N)
m.saveImage(Donnee,N)
m = GNG(nbD, tFinal =N)
m.saveImage(Donnee,N)
m = DSOM(nbD, tFinal =N)
m.saveImage(Donnee,N)
"""

#m = DSOM(nbD)
m.affichage(parser, True, tf= 10000)
