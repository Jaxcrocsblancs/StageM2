

class Voisin:
	def __init__(self,n1,n2,init=1):
		self.n1 = n1;
		self.n2 = n2;
		self.Life = init;
	
	def upLife(self):
		self.Life += 1