import re, sys
import numpy as np
from node import Node
import pdb

class Graph(object):
	def __init__(self, mul=2):
		self.mul_delay = mul
		self._LC = 1
		self.vertex = 0
		self.edge = 0
		self.adjlist = []
		self.edge_index = [[],[]]
		self.depth = 0
		self.order = []
		self.revOrder = []
		self.totLatency = 0
		self.finalLatency = 0
		self.prev_totLatency = 0
		self.numScheduledOp = 0
		# attr_mat[0]: Current schedule
		# attr_mat[1]: Current possible move
		# attr_mat[2]: All possible move
		self.attr_mat = []
		# reward and punishment
		self.reward = dict()
		self.reward["penalty"] = -1
		self.reward["illegal"] = -5
		self.reward["small"] = 0.1
		self.reward["ending"] = 0
		self.resource_constrint = 0
		self.vertex_attributes = []
		self.global_attributes = []
		self.topo_order_DFS = []
		self.topo_order = []
		self.last_schedule = -1
		self.best = False
		self.ODG_scheduling = False
		

	def setLatencyFactor(self,lc):
		self._LC = lc

	def getLatencyFactor(self):
		return self._LC

	def setConstrainedL(self,conL):
		self.CONSTRAINED_L = conL

	def getConstrainedL(self):
		return self.CONSTRAINED_L+1

	def getMulDelay(self):
		return self.mul_delay

	def setMAXRESOURCE(self,r):
		self.maxNr = {"MUL":r[0], "ALU":r[1]}

	def initialize(self):
		self.dfs() # obtain CONSTRAINED_L
		self.currNr = {"MUL":0, "ALU":0}
		self.bestNr = {"MUL":0x3f3f3f, "ALU":0x3f3f3f}
		self.nrt = {"MUL":np.array([0]*(self.CONSTRAINED_L+1)), "ALU":np.array([0]*(self.CONSTRAINED_L+1))}

	def read(self,infile):
		# print("Begin parsing...")
		for line in infile:
			if not ("label" in line or "name" in line):
				if "property" in line:
					res = re.split("=|,|\\].*",line)
					self.mul_delay = int(res[1])
					self.setLatencyFactor(float(res[3]))
				else:
					continue
			elif "label" in line:
				res = re.split(" *\\[ *label *= *| *\\];| +",line)
				op, op_type = res[1], res[2]
				self.add_vertex(op,op_type)
			else:
				res = re.split(" *\\[ *name *= *| *\\];| *-> *| +",line)
				src, des = res[1], res[2]
				self.add_edge(src,des)

	def mapR(self,type_,mode=0):
		if (type_ == "mul" or type_ == "MUL" or type_ == "div" or type_ == "DIV"):
			return ("MUL" if mode == 0 else 0)
		else:
			return ("ALU" if mode == 0 else 1)

	def add_vertex(self,name_,type_):
		delay = 1
		if self.mapR(type_) == "MUL":
			delay = self.mul_delay
		v = Node(self.vertex,name_,type_,delay)
		self.vertex += 1
		self.adjlist.append(v)

	def add_edge(self,src,des):
		self.edge_index[0].append(int(src))
		self.edge_index[1].append(int(des))
		for i in range(len(self.adjlist)):
			if self.adjlist[i].name == src:
				for j in range(len(self.adjlist)):
					if self.adjlist[j].name == des:
						self.adjlist[i].succ.append(j)
						self.adjlist[i].vertex_attribute[1] += 1
						self.adjlist[j].pred.append(i)
						self.adjlist[j].vertex_attribute[2] += 1
						self.edge += 1
						break

	def dfsASAP(self,num):
		if self.mark[num]:
			return
		if len(self.adjlist[num].pred) == 0:
			self.adjlist[num].setASAP(-1,0)
		else:
			for j in self.adjlist[num].pred:
				self.dfsASAP(j)
				self.adjlist[num].setASAP(j,self.adjlist[j].getASAP() + self.adjlist[j].delay)
		self.depth = max(self.adjlist[num].getASAP() + self.adjlist[num].delay - 1, self.depth)
		self.setConstrainedL(int((self.depth)*self._LC))
		self.mark[num] = True
		self.order.append(self.adjlist[num])

	def dfsALAP(self,num):
		if self.mark[num]:
			return
		self.topo_order_DFS += [num]
		if len(self.adjlist[num].succ) == 0:
			# CONSTRAINED_L is used here, dfsASAP must be done first
			self.adjlist[num].setALAP(-1, self.CONSTRAINED_L - self.adjlist[num].delay + 1)
		else:
			for j in self.adjlist[num].succ:
				self.dfsALAP(j)
				self.adjlist[num].setALAP(j,self.adjlist[j].getALAP() - self.adjlist[num].delay)
		self.mark[num] = True
		self.revOrder.append(self.adjlist[num])

	def dfs(self):
		# print("Begin DFS...")
		self.mark = np.zeros(self.vertex,dtype=bool)
		for i in range(len(self.adjlist)):
			if len(self.adjlist[i].succ) == 0:
				self.dfsASAP(i)
		self.mark = np.zeros(self.vertex,dtype=bool)
		for i in range(len(self.adjlist)):
			if len(self.adjlist[i].pred) == 0:
				self.dfsALAP(i)
		# print("Finish DFS.")
		# print("Constrained Latency is %d" % (self.CONSTRAINED_L+1))

	def topo_sort(self):
		for i in range(self.vertex):
			self.adjlist[i].num_pred = len(self.adjlist[i].pred)
		while len(self.topo_order) < self.vertex:
			visited = []
			for i in range(self.vertex):
				if i in self.topo_order:
					continue
				if self.adjlist[i].num_pred == 0:
					self.topo_order += [i]
					visited += [i]
			for j in visited:
				for k in self.adjlist[j].succ:
					self.adjlist[k].num_pred -= 1

	def initial_schedule(self):
		# clear previous attr_mat
		self.totLatency = 0
		self.numScheduledOp = 0
		self.currNr = {"MUL":0, "ALU":0}
		self.bestNr = {"MUL":0x3f3f3f, "ALU":0x3f3f3f}
		self.nrt = {"MUL":np.array([0]*(self.CONSTRAINED_L+1)), "ALU":np.array([0]*(self.CONSTRAINED_L+1))}
		for i in range(len(self.adjlist)):
			self.adjlist[i].initial()

		self.attr_mat = np.zeros((3,self.vertex,self.CONSTRAINED_L+1))
		for i in range(self.vertex):
			self.attr_mat[1:3,i,self.adjlist[i].getASAP():self.adjlist[i].getALAP() + self.adjlist[i].delay] = 1
		for i in range(self.vertex):
			self.schedule_node(i,self.adjlist[i].getASAP(),mode=0)
		#self.output_schedule()

	# mode = 0: initial schedule; mode = 1: reschedule
	def schedule_node(self,op,step,mode=1):
		self.best = False
		# if mode == 1:
		# 	print(f'scheduling node {op} in cstep {step}')
		if self.ODG_scheduling == True:
			if not self.test_node_illegal(op,step):
				return True, self.reward["illegal"], self.best#skip this schedule and retry

			if not self.test_val(op,step):
				self.adjlist[op].vertex_attribute[6]+=1
				return True, self.reward["penalty"], self.best#skip this schedule and retry

			if self.last_schedule != op: #refresh once agent jump the failed op
				self.adjlist[self.last_schedule].vertex_attribute[6] = 0
				self.last_schedule = op
		reward = 0
		tempR = self.mapR(self.adjlist[op].type)
		tempNum = self.mapR(self.adjlist[op].type,1)
		# remove old attr_mat
		oldOpNr = 0
		for d in range(self.adjlist[op].delay):
			oldOpNr += self.nrt[tempR][self.adjlist[op].cstep + d]
		if mode == 1:
			self.numScheduledOp += 1
			for d in range(self.adjlist[op].delay):
				# since the op initially placed here, so it should be at least WA
				self.attr_mat[0,op,self.adjlist[op].cstep + d] = 0
				self.nrt[tempR][self.adjlist[op].cstep + d] -= 1
		# current operation
		self.adjlist[op].schedule(step)
		delay = self.adjlist[op].delay
		for d in range(delay):
			self.nrt[tempR][step + d] += 1
		self.attr_mat[0,op,step:step+delay] = 1
		self.attr_mat[1,op,step:step+delay] = 0
		self.attr_mat[1,op,self.adjlist[op].getASAP():step] = 1
		self.attr_mat[1,op,step+delay:self.adjlist[op].getALAP()+delay] = 1
		# other influenced operations
		for vpred in self.adjlist[op].pred:
			tempALAP = self.adjlist[vpred].getALAP()
			d = self.adjlist[vpred].delay
			self.adjlist[vpred].setALAP(op,step - d)
			currALAP = self.adjlist[vpred].getALAP()
			self.attr_mat[1,vpred,min(tempALAP,currALAP)+d:max(tempALAP,currALAP)+d] = 0 if currALAP < tempALAP else 1
			if currALAP > tempALAP:
				#print("small reward is %f" %self.reward["small"])
				reward += self.reward["small"]
		for vsucc in self.adjlist[op].succ:
			tempASAP = self.adjlist[vsucc].getASAP()
			self.adjlist[vsucc].setASAP(op,step + self.adjlist[op].delay)
			currASAP = self.adjlist[vsucc].getASAP()
			self.attr_mat[1,vsucc,min(tempASAP,currASAP):max(tempASAP,currASAP)] = 0 if currASAP > tempASAP else 1
			if currASAP < tempASAP:
				#print("small reward is %f" %self.reward["small"])
				reward += self.reward["small"]

		self.adjlist[op].vertex_attribute[5]=step
		self.vertex_attributes=[]
		for i in range(len(self.adjlist)):
			self.adjlist[i].vertex_attribute[3] = self.adjlist[i].getASAP()
			self.adjlist[i].vertex_attribute[4] = self.adjlist[i].getALAP()
			#self.vertex_attributes[i] = self.adjlist[i].vertex_attribute
			self.vertex_attributes.append(self.adjlist[i].vertex_attribute)
		self.vertex_attributes=np.array(self.vertex_attributes)
		
		self.prev_totLatency = self.totLatency
		self.totLatency = max(self.totLatency, step + self.adjlist[op].delay) # step start from 0
		oldNr = self.currNr[tempR]
		self.currNr[tempR] = self.nrt[tempR].max()
		if mode != 0:
			#print("Updating Nrs")
			#if self.currNr["MUL"] != 0 and self.currNr["ALU"] != 0 and self.currNr["MUL"] + self.currNr["ALU"] <= self.bestNr["MUL"] + self.bestNr["ALU"]:
			if self.currNr["MUL"] + self.currNr["ALU"] + self.totLatency <= self.bestNr["MUL"] + self.bestNr["ALU"] + self.finalLatency:
				#print("Updating bestNrs")
				self.bestNr["MUL"], self.bestNr["ALU"] = self.currNr["MUL"], self.currNr["ALU"]
				self.finalLatency = self.totLatency
				self.best = True
				for op in range(self.vertex):
					self.adjlist[op].final_cstep = self.adjlist[op].cstep
		else:
			self.bestNr["MUL"], self.bestNr["ALU"] = self.currNr["MUL"], self.currNr["ALU"]
			self.finalLatency = self.totLatency
			for v in range(self.vertex):
				self.adjlist[v].final_cstep = self.adjlist[v].cstep
			self.best = True
		newOpNr = 0
		for d in range(self.adjlist[op].delay):
			newOpNr += self.nrt[tempR][self.adjlist[op].cstep + d]
		# early stop
		cnt = 0
		legal_move = self.getLegalMove()[0]
		for legal_op in legal_move:
			legal_op = self.adjlist[legal_op]
			typeR = self.mapR(legal_op.type)
			if (self.nrt[typeR][legal_op.cstep+1:legal_op.cstep+1+legal_op.delay] + 1 \
				> self.currNr[typeR]).any() and \
				(self.nrt[typeR][legal_op.cstep-1:legal_op.cstep-1+legal_op.delay] + 1 \
				> self.currNr[typeR]).any():
				cnt += 1
		if cnt >= len(legal_move):
			return False, self.reward["ending"], self.best
		# final reward
		reward += oldNr - self.currNr[tempR]
		reward += (oldOpNr - newOpNr)/self.adjlist[op].delay
		reward += (self.prev_totLatency - self.totLatency)
		return True, reward, self.best

	def test_node_illegal(self,op,step):
		if op < 0 or op >= self.vertex:
			#print("op %d exceed the num of vertex" %op)
			return False
		return True

	def test_val(self,op,step):
		# Latency Constraint
		if step < 0 or step + self.adjlist[op].delay - 1 > self.CONSTRAINED_L:
			#print("op %d exceeds the constrained latency" %op)
			return False
		# Resource Constraint
		tempR = self.mapR(self.adjlist[op].type)
		if self.nrt[tempR][step] + 1 > self.maxNr[tempR]:
			#print("op %d exceeds the constrained resource" %op)
			return False
		# Data dependency Constraint
		if step < self.adjlist[op].getASAP() or self.adjlist[op].getALAP() < step:
			#print("op %d exceeds its available clock cycles" %op)
			return False
		for vsucc in self.adjlist[op].succ:
			vsucc = self.adjlist[vsucc]
			if vsucc.cstep > -1 and step + self.adjlist[op].delay - 1 >= vsucc.cstep:
				return False
		for vpred in self.adjlist[op].pred:
			vpred = self.adjlist[vpred]
			if vpred.cstep > -1 and vpred.cstep + vpred.delay > step:
				return False
		return True

	def test_final(self):
		flag = True
		for v in self.adjlist:
			for vsucc in v.succ:
				vsucc = self.adjlist[vsucc]
				if v.cstep + v.delay - 1 >= vsucc.cstep:
					flag = False
					print("Schedule conflicts with Node %d(%s) and Node %d(%s)." % (v.num,v.name,vsucc.num,vsucc.name))
					return flag
		return flag

	def get_attr_mat(self):
		return self.attr_mat

	def getNrt(self):
		return self.nrt

	def getLegalMove(self):
		res = []
		res_dict = dict()
		cnt = 0
		for (op,row) in enumerate(self.get_attr_mat()[1,:,:]):
			if (row[:] == 1).any(): # backward!
				res.append(op)
				res_dict[cnt] = op
				cnt += 1
		return (res,res_dict)

	def output_adjlist(self):
		print("Adjacent List:")
		for v in self.adjlist:
			print("Node %d(%s):" % (v.num,v.name),end=" ")
			for op in v.succ:
				print(op+1,end=" ")
			print()

	def output_axap(self):
		print("AXAP:")
		for v in self.adjlist:
			print("Node %d(%s): [%d, %d]" % (v.num,v.name,v.getASAP(),v.getALAP()))

	def output_schedule(self):
		print("Schedule: ")
		for v in self.adjlist:
			print("Node %d(%s): %d" % (v.num,v.name,v.final_cstep))

	def output(self):
		print("# of operations: %d" % self.vertex)
		print("Latency factor: %f, CONSTRAINED_L: %d, Mul_delay: %d" % (self._LC,self.CONSTRAINED_L+1,self.mul_delay))
		print("Best # of resources: MUL: %d, ALU: %d" % (self.bestNr["MUL"], self.bestNr["ALU"]))
		print("Current # of resources: MUL: %d, ALU: %d" % (self.currNr["MUL"], self.currNr["ALU"]))
		print("Latency: %d" % self.totLatency)
		print("Final Latency: %d" % self.finalLatency)
		self.output_schedule()

	def get_totLatency(self):
		return self.totLatency

	def get_finalLatency(self):
		return self.finalLatency

	def print_DFG_settings(self):
		print("constrained_Latency:",self.getConstrainedL())
		print("Constrained resources: MUL: %d ALU: %d" % (self.maxNr["MUL"],self.maxNr["ALU"]))