import pulp, time
from graph import Graph
from mkdir_path import *
mkdir_path("ILP_formulation")
mkdir_path("Sol")

class ILPSolver(object):
	def __init__(self, file_num, mul_delay=2, lf=1.0, constrained_MUL=0, constrained_ALU=0, istestfile=0):
		self.feasible = ' '
		self.schedule = dict()
		self.MUL = 0
		self.ALU = 0
		self.time = 0
		g = Graph(mul_delay)
		g.setLatencyFactor(lf)
		if istestfile == 0:
			with open("./DFG/dfg_%d.dot" % file_num) as infile:
				g.read(infile)
		else:
			with open("./Test/test_dfg_%d.dot" % file_num) as infile:
				g.read(infile)
		begin_time = time.time()
		g.initialize()
		prob = pulp.LpProblem("Time-Constrained Scheduling Problem",pulp.LpMinimize)
		M1 = pulp.LpVariable("MUL",lowBound=0,upBound=None,cat=pulp.LpInteger)
		M2 = pulp.LpVariable("ALU",lowBound=0,upBound=None,cat=pulp.LpInteger)
		Latency = pulp.LpVariable("MaxLatency",lowBound=1,upBound=None,cat=pulp.LpInteger)
		prob += (M1 + M2 + Latency), "Minimize the number of FUs and overall Latency"

		# Time frame constraints
		x = pulp.LpVariable.dicts("x",(range(len(g.adjlist)),range(g.getConstrainedL())),lowBound=0,upBound=1,cat=pulp.LpInteger)
		for (i,node) in enumerate(g.adjlist):
			prob += pulp.lpSum([x[i][t] for t in range(node.getASAP(),node.getALAP()+1)]) == 1, ""

		# Latency constraints
		for (i,node) in enumerate(g.adjlist):
			prob += (pulp.lpSum([(t+1) * x[i][t] for t in range(node.getASAP(),node.getALAP()+1)]) + node.delay - 1 - Latency <= 0), ""

		# Resource constraints
		rowR = []
		for i in range(g.getConstrainedL()):
			rowR.append({"ALU":[],"MUL":[]})
		for (i,node) in enumerate(g.adjlist):
			for t in range(node.getASAP(),node.getALAP()+node.delay):
				rowR[t][g.mapR(node.type)].append(i)
		for t in range(g.getConstrainedL()):
			for typeR in ["ALU","MUL"]:
				if len(rowR[t][typeR]) < 2:
					continue
				else:
					prob += pulp.lpSum([x[i][td] for i in rowR[t][typeR]
						for td in range(max(t-g.adjlist[i].delay+1,0),t+1)]) - (M1 if typeR == "MUL" else M2)<= 0, ""
		prob += M1-constrained_MUL<=0, ""
		prob += M2-constrained_ALU<=0, ""

		# Data dependency constraints
		for (i,node) in enumerate(g.adjlist):
			for vsucc in node.succ:
				prob += (pulp.lpSum([(t+1)*x[i][t] for t in range(node.getASAP(),node.getALAP()+1)])
					- pulp.lpSum([(t+1)*x[vsucc][t] for t in range(g.adjlist[vsucc].getASAP(),g.adjlist[vsucc].getALAP()+1)])
					<= (-1)*node.delay), ""

		if istestfile == 0:
			prob.writeLP("./ILP_formulation/dfg_%d.lp" % (file_num))
		else:	
			prob.writeLP("./Test/test_dfg_%d.lp" % (file_num))
		prob.solve()
		end_time = time.time()
		self.time = end_time - begin_time
		print("constrained_MUL = %d" %constrained_MUL)
		print("constrained_ALU = %d" %constrained_ALU)
		print("MUL = %d" % prob.variablesDict()["MUL"].varValue)
		print("ALU = %d" % prob.variablesDict()["ALU"].varValue)
		if istestfile == 0:
			out_file = open("./Sol/dfg_%d.sol" % file_num,"w")
		else:
			out_file = open("./Test/test_dfg_%d.sol" % file_num,"w")
		max_CLK = 0
		for v in sorted(prob.variables(),key=lambda x: int(x.name.split("_")[1]) if len(x.name.split("_")) != 1 else 0):
			if v.name[0] == "x" and v.varValue == 1:
				op = v.name.split("_")[1]
				CLK = v.name.split("_")[-1]
				self.schedule[int(op)] = int(CLK)
				out_file.write("%s op/CLK %s\n" % (op,CLK))
				max_CLK = max(max_CLK, int(CLK))
		self.TotLatency = max_CLK + 1
		LatVal = prob.variablesDict()["MaxLatency"].varValue
		Equal = self.TotLatency == LatVal
		print("TotLatency = %d / %d, Equal? %d\n\n" %(self.TotLatency, LatVal, Equal))
		out_file.write("%d MUL/ALU %d\n" %(prob.variablesDict()["MUL"].varValue, prob.variablesDict()["ALU"].varValue))
		out_file.write("1 TotLatency %d\n" % (self.TotLatency))
		out_file.close()
		self.MUL = prob.variablesDict()["MUL"].varValue
		self.ALU = prob.variablesDict()["ALU"].varValue
		self.feasible = pulp.LpStatus[prob.status]
		print("Status:", self.feasible)
		if self.feasible != 'Optimal':
			print(file_num,"is infeasible!")

	def getOptSchedule(self):
		return self.schedule
	def getResource(self):
		return (self.MUL, self.ALU)
	def getTotLatency(self):
		return self.TotLatency
	def getFeasibility(self):
		return self.feasible
	def getTime(self):
		return self.time
