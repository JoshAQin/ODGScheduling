from copy import deepcopy

class OSPair(object):
	def __init__(self, op, CLK):
		super(OSPair, self).__init__()
		self.op = op
		self.CLK = CLK

	def __repr__(self):
		return "Op: %d, CLK: %d" % (self.op + 1,self.CLK + 1)

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
		self.CLK_start = -1
		self.CLK_end = -1
		self.scheduled_CLK = -1
		self.vertex_attribute = [delay_, 0, 0, 0, 0, 0, 0] 
		# 0:delay, 1:#predecessor, 2:#successor, 3:ASAP schedule, 4:ALAP schedule, 5:current schedule, 6:#fails

	def initial(self):
		self.asap = []
		self.alap = []
		for p in self.iasap:
			self.asap.append(deepcopy(p))
		for p in self.ialap:
			self.alap.append(deepcopy(p))
		self.asap.sort(key=lambda x:x.CLK,reverse=True)
		self.alap.sort(key=lambda x:x.CLK)
		self.CLK_start = -1

	def schedule(self, CLK):
		self.CLK_start = CLK
		self.CLK_end = CLK + self.delay - 1

	def setASAP(self,op,asap_):
		flag = False
		for i in range(len(self.asap)):
			if self.asap[i].op == op:
				self.asap[i].CLK = asap_
				flag = True
		if not flag:
			self.asap.append(OSPair(op,asap_))
			self.iasap.append(OSPair(op,asap_))
		self.asap.sort(key=lambda x:x.CLK,reverse=True)

	def setALAP(self,op,alap_):
		flag = False
		for i in range(len(self.alap)):
			if self.alap[i].op == op:
				self.alap[i].CLK = alap_
				flag = True
		if not flag:
			self.alap.append(OSPair(op,alap_))
			self.ialap.append(OSPair(op,alap_))
		self.alap.sort(key=lambda x:x.CLK)

	def getASAP(self):
		return self.asap[0].CLK

	def getALAP(self):
		return self.alap[0].CLK
