import nltk

class ProductionBuilder:
	def __init__(self):
		self.seed = 1

	def buildNormal(self, lhs, rhs):
		return nltk.grammar.Production(lhs, rhs)

	def build(self, rhs):
		newKey = 'X' + str(self.seed)

		self.seed = self.seed + 1

		lhs = nltk.Nonterminal(newKey)

		return self.buildNormal(lhs, rhs)