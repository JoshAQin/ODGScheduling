from copy import deepcopy
import pdb

class OSPair(object):
	def __init__(self, op, step):
		super(OSPair, self).__init__()
		self.op = op
		self.step = step

	def __repr__(self):
		return "Op: %d, Step: %d" % (self.op + 1,self.step + 1)

class Node(object):
	def __init__(self, num_, name_, type_, delay_):
		super(Node, self).__init__()
		self.num = num_
		self.name = name_
		self.type = type_
		self.delay = delay_
		self.pred = []
		self.num_pred = []
		self.succ = []
		self.asap = []
		self.alap = []
		self.iasap = []
		self.ialap = []
		self.cstep = -1
		self.final_cstep = -1
		self.vertex_attribute = [delay_, 0, 0, 0, 0, 0, 0] 
		# 0:delay, 1:#predecessor, 2:#successor, 3:ASAP schedule, 4:ALAP schedule, 5:current schedule, 6:#fails

	def initial(self):
		self.asap = []
		self.alap = []
		for p in self.iasap:
			self.asap.append(deepcopy(p))
		for p in self.ialap:
			self.alap.append(deepcopy(p))
		self.asap.sort(key=lambda x:x.step,reverse=True)
		self.alap.sort(key=lambda x:x.step)
		self.cstep = -1

	def schedule(self, step):
		self.cstep = step

	def setASAP(self,op,asap_):
		flag = False
		for i in range(len(self.asap)):
			if self.asap[i].op == op:
				self.asap[i].step = asap_
				flag = True
		if not flag:
			self.asap.append(OSPair(op,asap_))
			self.iasap.append(OSPair(op,asap_))
		self.asap.sort(key=lambda x:x.step,reverse=True)

	def setALAP(self,op,alap_):
		flag = False
		for i in range(len(self.alap)):
			if self.alap[i].op == op:
				self.alap[i].step = alap_
				flag = True
		if not flag:
			self.alap.append(OSPair(op,alap_))
			self.ialap.append(OSPair(op,alap_))
		self.alap.sort(key=lambda x:x.step)

	def getASAP(self):
		return self.asap[0].step

	def getALAP(self):
		return self.alap[0].step
