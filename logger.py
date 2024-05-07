import re, os, logging

from mkdir_path import *
mkdir_path("Log")

class LogHandler(object):
	def __init__(self, name, level=logging.INFO):
		self.name = name
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(level)
		try:
			self.file_num = int(re.split("[_.]",sorted(os.listdir("./Log"),key=lambda x: x.split("_")[1])[-2])[1]) + 1
		except:
			self.file_num = 1
		self.file_name = "./Log/%s_%02d.log" % (self.name,self.file_num)
		self.handler = logging.FileHandler(self.file_name)
		self.formatter = logging.Formatter('%(asctime)s - %(message)s')
		self.handler.setFormatter(self.formatter)
		self.logger.addHandler(self.handler)

	def getLogger(self):
		return self.file_num, self.logger
