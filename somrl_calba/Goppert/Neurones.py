

class Neurones:
    
	def __init__(self,Vect, error=0):
		self.VecteurDonnee = Vect
		self.error = error
		
	def getVecteurDonnee(self):
		return self.VecteurDonnee
		
	def setVecteurDonnee(self,Vect):
		self.VecteurDonnee = Vect