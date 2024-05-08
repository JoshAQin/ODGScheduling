import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
from policy import Policy
from mkdir_path import *
mkdir_path("Networks")

class Agent(object):
	def __init__(self, policy_network="", device="",lr=5e-4):
		super(Agent, self).__init__()
		self.device = device
		if policy_network == "":
			net = Policy().to(self.device)
			print("Build a new network!")
		else:
			try:
				net = torch.load("./Networks/" + policy_network).to(self.device)
				print("Loaded %s." % policy_network)
			except:
				net = Policy().to(self.device)
				print("No such network named %s. Rebuild a new network!" % policy_network)
		self.policy = net
		self.optimizer = optim.Adam(self.policy.parameters(),lr=lr)
		self.pi = Variable(torch.FloatTensor([math.pi])).to(self.device)
		self.prev_index = 99999

	def normal(self, x, mu, sigma2):
		a = ( -1 * (Variable(x)-mu).pow(2) / (2*sigma2) ).exp()
		b = 1 / ( 2 * self.pi * sigma2 ).sqrt()
		return a*b


	def get_action(self, g_data, legal_move, reschedule=False):
		mu0, sigma2 = self.policy(g_data)
		num_move = len(legal_move[0])
		mu = mu0 * num_move
		cnt=0
		while True: 
			eps = torch.randn(mu.size()).to(self.device)
			index = (mu + sigma2.sqrt()*eps).squeeze()
			log_prob = self.normal(index, mu, sigma2).log().squeeze()
			cnt += 1
			if index != self.prev_index:
				reschedule = False
			if reschedule==False:
				break
			if cnt > 100 and abs(sigma2) < 0.1:
				sigma2 = torch.tensor(100, dtype=float, device=self.device, requires_grad=True)
		self.prev_index = index

		if index<0:
			action = -1 * legal_move[1][min(round(-1 * index.item()),num_move-1)]
		else:
			action = legal_move[1][min(round(index.item()),num_move-1)]
		#print(f'index:{index},action:{action}')
		entropy = 0.5*( ( 2 * self.pi * sigma2 ).log() + 1 ).squeeze()
		#print(f'action: {action}; log_prob: {log_prob}; entropy: {entropy}')
		return action, log_prob, entropy


	def update_weight(self, all_log_probs, all_rewards, all_entropies, baseline=False):
		gamma = 0.99
		eps = np.finfo(np.float32).eps.item()
		tot_loss = []
		res_rewards, avg_reward = [], []
		# baseline `1/N\sum_{i=1}^N r(\tau)`
		for temp_rewards in zip(all_rewards):
			# a full trace \tau
			R = 0
			rewards = []
			for r in temp_rewards[::-1]:
				R = r + gamma * R
				rewards.insert(0, R)
			avg_reward.append(rewards[0]) # r(\tau)
			res_rewards.append(rewards)
		if baseline:
			avg_reward = np.array(avg_reward).mean()
			#print(f'avg_reward: {avg_reward}')
		else:
			avg_reward = 0
			#print(f'avg_reward: {avg_reward}')
		for log_probs, rewards, entropies in zip(all_log_probs,res_rewards,all_entropies):
			rewards = torch.tensor(np.array(rewards)).to(self.device).squeeze()
			#print(f'rewards: {rewards}')
			rewards = rewards - avg_reward # minus baseline
			#print(f'normalized rewards: {rewards}')
			loss = torch.Tensor([0]).float().to(self.device)
			for CLK, (log_prob, reward, entropy) in enumerate(zip(log_probs,rewards,entropies)):
				loss += -1 * log_prob * reward - 0.001 * entropy
			tot_loss.append(loss)
		# backpropagate
		self.optimizer.zero_grad()
		tot_loss = torch.cat(tot_loss).mean() # sum()
		tot_loss.backward()
		self.optimizer.step()
		res = tot_loss.item()
		del tot_loss
		return res

