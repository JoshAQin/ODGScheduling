import pulp
from pulp.apis import PULP_CBC_CMD
from graph import Graph
from mkdir_path import *
import pdb
import time
expanding_factor = 3

class SDCSolver(object):
	def __init__(self, file_num, mul_delay=2, lf=1.0, constrained_MUL=0, constrained_ALU=0):
		self.feasible = ' '
		self.schedule = dict()
		self.MUL = 0
		self.ALU = 0
		g = Graph(mul_delay)
		with open("./Test/test_dfg_%d.dot" % file_num) as infile:
			g.read(infile)
		g.setLatencyFactor(lf*expanding_factor)
		begin_time = time.time()
		g.initialize()
		ConstrainedL = g.getConstrainedL()
		prob = pulp.LpProblem("Time-Constrained Scheduling Problem",pulp.LpMinimize)
		Latency = pulp.LpVariable("MaxLatency",lowBound=1,upBound=None,cat=pulp.LpContinuous)
		prob += Latency, "Minimize the number of overall Latency"

		x = pulp.LpVariable.dicts("x",(range(g.vertex)),lowBound=1,upBound=(ConstrainedL + 1),cat=pulp.LpContinuous)

		# Latency constraints
		for (i,node) in enumerate(g.adjlist):
			#print("adding latency constraint for op ", i, "with delay", node.delay)
			#x-d+1 is the ending cycle
			prob += (x[i] - Latency <= 1 - node.delay), ""

		prob += (Latency <= ConstrainedL + 1), ""

		# Resource constraints
		g.topo_sort()
		topo_order = g.topo_order
		for (i,node_i) in enumerate(topo_order):
			num_MUL = 0
			num_ALU = 0
			if g.mapR(g.adjlist[node_i].type) == "MUL":
				num_MUL += 1
			else:
				num_ALU += 1
			#print("node: ", i, "rest: ", topo_order[i+1:])
			for vsucc in topo_order[i+1:]:
				if g.mapR(g.adjlist[vsucc].type) == "MUL":
					num_MUL += 1
				else:
					num_ALU += 1
				if num_MUL > constrained_MUL:
					#print("adding operation dependency for op ", node_i, "and", vsucc)
					num_MUL -= 1
					prob += (x[node_i] - x[vsucc] <= (-1)*g.adjlist[node_i].delay), "" 
				if num_ALU > constrained_ALU:
					#print("adding operation dependency for op ", node_i, "and", vsucc)
					num_ALU -= 1
					prob += (x[node_i] - x[vsucc] <= (-1)*g.adjlist[node_i].delay), "" 
		# print("Resource constraints generated.")

		# Data dependency constraints
		for (i,node) in enumerate(g.adjlist):
			prob += ((-1)*x[i] <= (-1)*node.getASAP() - 1), ""
			prob += (x[i] <= node.getALAP() + 1), ""

		for (i,node_p) in enumerate(g.adjlist):
			for vsucc_p in node_p.succ:
				#print("adding operation dependency for op ", i, "and", vsucc_p)
				prob += (x[i] - x[vsucc_p] <= (-1)*node_p.delay), ""
		# print("Data dependency constraints generated.")

		# print("Finish SDC generation.")
		prob.writeLP("./Test/test_dfg_%d_sdc.lp" % (file_num))
		prob.solve()
		end_time = time.time()
		self.time = end_time - begin_time
		out_file = open("./Test/test_dfg_%d_sdc.sol" % file_num,"w")
		max_cstep = 0
		MUL_list = [0] * (g.getConstrainedL() + 1)
		ALU_list = [0] * (g.getConstrainedL() + 1)
		print("Initialize FU lists!")
		for v in sorted(prob.variables(),key=lambda x: int(x.name.split("_")[1]) if len(x.name.split("_")) != 1 else 0):
			if v.name[0] == "x":
				op = int(v.name.split("_")[1])
				cstep = int(v.varValue)
				self.schedule[op] = cstep
				#print("Operation:", op, " is scheduled in ", cstep, "varValue=", v.varValue)
				delay = g.adjlist[op].delay
				if g.mapR(g.adjlist[op].type) == "MUL":
					for t in range(cstep,cstep+delay):
						MUL_list[t] += 1
						#print("Operation:", op, " is a MUL and occupies cycle ", t)
				if g.mapR(g.adjlist[op].type) == "ALU":
					for t in range(cstep,cstep+delay):
						ALU_list[t] += 1
						#print("Operation:", op, " is a ALU and occupies cycle ", t)
				out_file.write("%s op/cstep %s\n" % (op,cstep))
				max_cstep = max(max_cstep, cstep)
		print("Done!")
		self.TotLatency = max_cstep
		#print("TotLatency=", self.TotLatency, "Latency=", prob.variablesDict()["MaxLatency"].varValue)
		self.MUL = max(MUL_list)
		self.ALU = max(ALU_list)
		print("%d MUL/ALU %d" %(self.MUL, self.ALU))
		print("1 TotLatency %d" % (self.TotLatency))
		out_file.write("%d MUL/ALU %d\n" %(self.MUL, self.ALU))
		out_file.write("1 TotLatency %d\n" % (self.TotLatency))
		out_file.close()
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
