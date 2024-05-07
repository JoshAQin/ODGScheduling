from graph import Graph
import random
from mkdir_path import *
mkdir_path("DFG")
mkdir_path("Test")
mkdir_path("Test/DFG")

NUM_GRAPH = 2000
TEST_NODE = [10, 20, 30, 40, 50, 70, 90, 110, 160, 200, 300, 400, 600, 800, 1000]
#istest = input('Is this a test generation? Yes or No')
istest = 'No'

class DFGGen(object):
	def __init__(self, num, tot_node=50, min_per_layer=1, max_per_layer=5, link_rate=0.5, mul_rate=0.3):
		res = "digraph {\n"
		res += "    node [fontcolor=black]\n"
		res += "    property [mul=%d,lf=%.1f]\n" % (2,random.uniform(1.0,2.0))
		nowNode = 0
		edges = []
		pre_layer = []
		while nowNode < tot_node:
			newNode = random.randint(min_per_layer, max_per_layer)
			if nowNode + newNode > tot_node:
				newNode = tot_node - nowNode
			cur_layer = []
			for i in range(nowNode,nowNode + newNode):
				cur_layer.append(i)
			for j in pre_layer:
				for k in cur_layer:
					if random.random() < link_rate:
						edges.append((j,k))
			pre_layer = cur_layer[:]
			nowNode += newNode
		for i in range(tot_node):
			if random.random() < mul_rate:
				typename = "mul"
			else:
				typename = "add"
			res += "    %d [ label = %s ];\n" % (i, typename)
		for (step,edge) in enumerate(edges):
			res += "    %d -> %d [ name = %d ];\n" % (edge[0],edge[1],step)
		res += "}\n"
		if istest=='No|no|n':
			output = open("./DFG/DFG_" + str(num) + ".dot","w")
			istest = False
		else if istest=='Yes|yes|y':
			output = open("./Test/DFG/test_DFG_" + str(num) + ".dot","w")
			istest = True
		else:
				print('Please input Yes or No')
				raise ValueError
		output.write(res)
		output.close()
if istest==False:
	for i in range(1,NUM_GRAPH+1):
		DFGGen(i,tot_node=random.randint(100,1000),mul_rate=random.uniform(0.3,0.5))
		if i % 100 == 0:
			print("Generated %d / %d DFGs." % (i,NUM_GRAPH))
else if istest==True:
	for i in range(0,10):
		DFGGen(i+1,tot_node=300,mul_rate=random.uniform(0.3,0.5))
		print("Generated %d / %d test DFGs." % (i+1,20))
	for i in range(10,20):
		DFGGen(i+1,tot_node=500,mul_rate=random.uniform(0.3,0.5))
		print("Generated %d / %d test DFGs." % (i+1,20))
