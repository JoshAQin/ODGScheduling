import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.attr, self.hidden1, self.hidden2 = 7, 32, 64
		self.hidden3, self.hidden4 = 32, 8

		self.GNN_fwd1 = GATConv(in_channels=self.attr, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.GNN_bwd1 = GATConv(in_channels=self.attr, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.bi_agg1 = nn.Linear(self.hidden1 * 2, self.hidden1)
		self.GNN_fwd2 = GATConv(in_channels=self.hidden1, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.GNN_bwd2 = GATConv(in_channels=self.hidden1, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.bi_agg2 = nn.Linear(self.hidden1 * 2, self.hidden1)
		self.GNN_fwd3 = GATConv(in_channels=self.hidden1, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.GNN_bwd3 = GATConv(in_channels=self.hidden1, out_channels=self.hidden1, heads=2, concat=False, add_self_loops=False)
		self.bi_agg3 = nn.Linear(self.hidden1 * 2, self.hidden1)
		self.GNN_fwd4 = GATConv(in_channels=self.hidden1, out_channels=self.hidden2, heads=2, concat=False, add_self_loops=False)
		self.GNN_bwd4 = GATConv(in_channels=self.hidden1, out_channels=self.hidden2, heads=2, concat=False, add_self_loops=False)
		self.bi_agg4 = nn.Linear(self.hidden2 * 2 + self.attr + self.hidden1 * 3, self.hidden2)

		self.node_self_attn = nn.Linear(self.hidden2, 1)
		self.readout = self._node_self_attn_readout
		self.MLP = nn.Sequential(
			nn.Linear(self.hidden2, self.hidden3),
			nn.LeakyReLU(True),
			nn.Linear(self.hidden3,self.hidden4),
			nn.LeakyReLU(True),
		)
		self.to_mu = nn.Linear(self.hidden4, 1)
		self.to_sigma = nn.Linear(self.hidden4, 1)
		self._initialize_weights()

	def _node_self_attn_readout(self, h, batch):
		attn_weights = self.node_self_attn(h)
		attn_weights = F.softmax(attn_weights, dim=-1)
		return global_add_pool(attn_weights * h, batch)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0.1, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, g_data):
		backward_edge_index = g_data.edge_index[[1,0]]
		x_orig = g_data.x.clone().to("cuda:0")
		x_fwd1 = self.GNN_fwd1(g_data.x, g_data.edge_index)
		x_bwd1 = self.GNN_bwd1(g_data.x, backward_edge_index)
		x_1 = F.leaky_relu(self.bi_agg1(torch.cat([x_fwd1, x_bwd1], dim = -1)))
		x_fwd2 = self.GNN_fwd2(x_1, g_data.edge_index)
		x_bwd2 = self.GNN_bwd2(x_1, backward_edge_index)
		x_2 = F.leaky_relu(self.bi_agg2(torch.cat([x_fwd2, x_bwd2], dim = -1)))
		x_fwd3 = self.GNN_fwd3(x_2, g_data.edge_index)
		x_bwd3 = self.GNN_bwd3(x_2, backward_edge_index)
		x_3 = F.leaky_relu(self.bi_agg3(torch.cat([x_fwd3, x_bwd3], dim = -1)))
		x_fwd4 = self.GNN_fwd4(x_3, g_data.edge_index)
		x_bwd4 = self.GNN_bwd4(x_3, backward_edge_index)
		x = F.leaky_relu(self.bi_agg4(torch.cat([x_fwd4, x_bwd4, x_orig, x_1, x_2, x_3], dim = -1)))
		x = self.readout(x, g_data.batch)
		x = self.MLP(x)
		mu = torch.tanh(self.to_mu(x))
		#print(f'mu:{mu.item()}')
		sigma2 = F.softplus(self.to_sigma(x))
		#print(f'sigma:{sigma2.item()}')
		return mu, sigma2