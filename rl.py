import time, sys, argparse, gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.data as gdata
from math import ceil, pow, log10
from logger import LogHandler
from graph import Graph
from agent import Agent
from ilp_solver import ILPSolver
from sdc_solver import SDCSolver
from mkdir_path import *
mkdir_path("Log")
mkdir_path("Loss")

parser = argparse.ArgumentParser(description="ODG-Based HLS Scheduler")
parser.add_argument("--mode", type=int, default=0, help="Scheduling mode: Training (-1), Multi-DFG (0), or Single-DFG (test_file_idx) mode? (default: 0)")
parser.add_argument("--latency_factor", type=float, default=1.5, help="Latency constraint scaling factor (default: 1.5)")
parser.add_argument("--resource_factor", type=float, default=1, help="resource constraint scaling factor (default: 1)")
parser.add_argument("--mul_delay", type=int, default=2, help="MUL delay (default: 2)")
parser.add_argument("--cuda_idx", type=int, default=1, help="Cuda index (default: 1)")
parser.add_argument("--policy_network", type=str, default="pre_trained_policy_network.pkl", help="Pre-trained policy network. (default: None)")
parser.add_argument("--episodes", type=int, default=50, help="Max training episodes (default: 50)")
parser.add_argument("--timesteps", type=int, default=1000, help="Max training timestep in one episode (default: 1000)")
parser.add_argument("--input_graphs", type=int, default=2000, help="Number of training graphs (default: 2000)")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--null_reward", type=int, default=-5, help="Reward of a failed schedule (default: -5)")
parser.add_argument("--comparison_factor", type=int, default=5, help="Comparison factor for performance-based early stop in scheduling  (default: 5)")

args = parser.parse_args()

device = torch.device(("cuda:%d" % (args.cuda_idx-1)) if args.cuda_idx != 0 else "cpu")
agent = Agent(policy_network=args.policy_network,device=device,lr=args.learning_rate)

if args.mode == -1:
	logger_num, logger = LogHandler("training").getLogger()
	logger.info("ODG-Based HLS Scheduler")
	print("Logger num: %d" % logger_num)
	file_name = "_rl_" + time.strftime("%Y%m%d_") + str(logger_num)
	logger.info("NLLLoss + Adam")
	logger.info("Batch size: %d, Learning rate: %f" % (args.batch_size,args.learning_rate))
	try:
		graph_best_reward = np.load("./Networks/graph_best_reward.npy").item()
		epi_best_reward = np.load("./Networks/epi_best_reward.npy").item()
		print("Loaded graph_best_reward:",graph_best_reward,"Loaded epi_best_reward:",epi_best_reward)
	except:
		graph_best_reward = -1000
		epi_best_reward = -1000
		print("Initialize best_reward as:",-1000)
else:
	logger_num, logger = LogHandler("test_%s" % args.mode).getLogger()
	logger.info("ODG-Based HLS Scheduler")


fig = plt.figure()

def train(episode): # Monte Carlo REINFORCE
	global graph_best_reward
	global epi_best_reward
	res_loss, res_reward = [], []
	for i_graph in range(args.input_graphs//args.batch_size):
		all_log_probs, all_rewards, all_entropies = [], [], []
		# simulate batch_size graphs
		for minibatch in range(args.batch_size):
			log_probs, rewards, entropies = [], [], []
			graph = Graph()
			file=open("./DFG/dfg_%d.dot" % (i_graph*args.batch_size+minibatch+1),"r")
			print("Scheduling DFG: dfg_%d.dot"% (i_graph*args.batch_size+minibatch+1))
			logger.info("Scheduling DFG: dfg_%d.dot"% (i_graph*args.batch_size+minibatch+1))
			graph.read(file)
			graph.initialize()
			graph.initial_schedule()
			graph.ODG_scheduling = True
			MUL_ASAP = graph.currNr["MUL"]
			ALU_ASAP = graph.currNr["ALU"]
			graph.setMAXRESOURCE([MUL_ASAP,ALU_ASAP])
			
			y=torch.tensor([0], dtype=torch.long)
			batch=torch.tensor([0], dtype=torch.long)
		
			# A full trajectory tau
			for timestep in range(args.timesteps):
				g_data=gdata.Data(x=torch.tensor(graph.vertex_attributes, dtype=torch.float),\
				edge_index=torch.tensor(graph.edge_index, dtype=torch.long),\
				y=y,batch=batch).to(device)
				legalMove = graph.getLegalMove()
				if len(legalMove[0]) == 0:
					if hasattr(torch.cuda, 'empty_cache'):
						torch.cuda.empty_cache()
					break


				action, log_prob, entropy = agent.get_action(g_data,legalMove)
				if action > -1 * graph.vertex and action < 0:
					node = -1 * action
					CLK = graph.adjlist[node].CLK_start - 1
				elif action >= 0 and action < graph.vertex:
					node = action
					CLK = graph.adjlist[node].CLK_start + 1
				else:#invalid
					node = action
					CLK = 0

				if graph.adjlist[node].vertex_attribute[6] > 10:
					action, log_prob, entropy = agent.get_action(g_data,legalMove,reschedule=True)
					if action > -1 * graph.vertex and action < 0:
						node = -1 * action
						CLK = graph.adjlist[node].CLK_start - 1
					elif action >= 0 and action < graph.vertex:
						node = action
						CLK = graph.adjlist[node].CLK_start + 1
					else:#invalid
						node = action
						CLK = 0

				fes, reward, isbest = graph.schedule_node(node, CLK)

				log_probs.append(log_prob)
				rewards.append(reward)
				entropies.append(entropy)
				del g_data, log_prob, reward, entropy
				if hasattr(torch.cuda, 'empty_cache'):
					torch.cuda.empty_cache()
				if fes == False:
					break
			if not rewards:
				rewards.append(args.null_reward)
				log_probs.append(1)
				entropies.append(1)
			all_log_probs.append(log_probs)
			all_rewards.append(np.array(rewards).astype(np.float64))
			all_entropies.append(entropies)
			file.close()
			del graph, log_probs, rewards, entropies
			if hasattr(torch.cuda, 'empty_cache'):
				torch.cuda.empty_cache()
		# update policy
		loss = agent.update_weight(all_log_probs,all_rewards,all_entropies,baseline=True) # be careful that the rewards are not aligned
		avg_reward = np.array([x.sum() for x in all_rewards]).mean()
		res_loss.append(loss)
		res_reward.append(avg_reward)
		if i_graph % 10 == 0:
			print("Train - Episode %d, Batch: %d, Loss: %f, Reward: %f" % (episode,i_graph,loss,avg_reward))
			logger.info("Train - Episode %d, Batch: %d, Loss: %f, Reward: %f" % (episode,i_graph,loss,avg_reward))
		if graph_best_reward < avg_reward:
			graph_best_reward = avg_reward
			torch.save(agent.policy,"./Networks/policy" + file_name + "_graph_best.pkl")
			save_reward = np.array(graph_best_reward)
			np.save("./Networks/graph_best_reward",save_reward)
		del all_log_probs, all_rewards, all_entropies, loss, avg_reward
		if hasattr(torch.cuda, 'empty_cache'):
			torch.cuda.empty_cache()
		gc.collect

	mean_res_loss = np.array(res_loss).mean()
	mean_res_reward = np.array(res_reward).mean()
	if epi_best_reward < mean_res_reward:
		epi_best_reward = mean_res_reward
		torch.save(agent.policy,"./Networks/policy" + file_name + "_epi_best.pkl")
		save_reward = np.array(epi_best_reward)
		np.save("./Networks/epi_best_reward",save_reward)
	del res_loss, res_reward
	if hasattr(torch.cuda, 'empty_cache'):
		torch.cuda.empty_cache()
	gc.collect
	return (mean_res_loss, mean_res_reward)

def test(file_num):
	print("Testing file test_dfg_%d.dot:" %(file_num))
	logger.info("Testing file test_dfg_%d.dot:" %(file_num))
	begin_time = time.time()
	nrt, nrta, nrts, step, lat = [], [], [], [], []
	graph = Graph(args.mul_delay)
	graph.setLatencyFactor(args.latency_factor)
	graph.read(open("./Test/test_dfg_%d.dot" % file_num,"r"))
	graph.initialize()
#1.ASAP
	graph.initial_schedule()
	MUL_ASAP = graph.currNr["MUL"]
	ALU_ASAP = graph.currNr["ALU"]
	TotLatency_ASAP = graph.get_finalLatency()
	graph.setMAXRESOURCE([MUL_ASAP,ALU_ASAP])
	graph.print_DFG_settings()
	logger.info("ASAP # of resources: MUL: %d, ALU: %d, Tot: %d. TotLatency: %d" % (MUL_ASAP, ALU_ASAP, MUL_ASAP+ALU_ASAP, TotLatency_ASAP))



#2. ILP
	try:# Existing solution
		count = len(open("./Test/test_dfg_%d.sol" % file_num,'r').readlines())
		sol = open("./Test/test_dfg_%d.sol" % file_num,"r")
		ops = dict()
		i = 0
		for line in sol:
			i = i + 1
			if(i < count - 1):
				op, CLK = map(int,line.split(" op/CLK "))
				ops[op] = CLK
			elif(i == count - 1):
				MUL_ILP, ALU_ILP = map(int,line.split(" MUL/ALU "))
			else:
				_, TotLatency_ILP = map(int,line.split(" TotLatency "))
	except:# Solving with LP solver
		print("file: test_dfg_%d.sol not found! regenerating...\n" %(file_num))
		ilp = ILPSolver(file_num,graph.getMulDelay(),graph.getLatencyFactor(),MUL_ASAP,ALU_ASAP,istestfile=1)
		ops = ilp.getOptSchedule()
		MUL_ILP, ALU_ILP =ilp.getResource()
		TotLatency_ILP = ilp.getTotLatency()
		feasibility = ilp.getFeasibility()
		if feasibility != 'Optimal':
			logger.info("ILP is ")
			logger.info(feasibility)
		logger.info("ILP_Time: {:.4f}".format(ilp.getTime()))
	logger.info("ILP # of resources: MUL: %d, ALU: %d, Tot: %d. TotLatency: %d" % (MUL_ILP, ALU_ILP, MUL_ILP+ALU_ILP, TotLatency_ILP))


#3. SDC
	constrained_MUL = ceil(MUL_ASAP * args.resource_factor)
	constrained_ALU = ceil(ALU_ASAP * args.resource_factor)
	try:# Existing solution
		count = len(open("./Test/test_dfg_%d_sdc.sol" % file_num,'r').readlines())
		sol = open("./Test/test_dfg_%d_sdc.sol" % file_num,"r")
		ops = dict()
		i = 0
		for line in sol:
			i = i + 1
			if(i < count - 1):
				op, CLK = map(int,line.split(" op/CLK "))
				ops[op] = CLK
			elif(i == count - 1):
				MUL_SDC, ALU_SDC = map(int,line.split(" MUL/ALU "))
			else:
				_, TotLatency_SDC = map(int,line.split(" TotLatency "))
	except:# Solving with LP solver
		print("file: test_dfg_%d_sdc.sol not found! regenerating..." %(file_num))
		sdc = SDCSolver(file_num,graph.getMulDelay(),graph.getLatencyFactor(), constrained_MUL, constrained_ALU)
		ops = sdc.getOptSchedule()
		MUL_SDC, ALU_SDC =sdc.getResource()
		TotLatency_SDC = sdc.getTotLatency()
		feasibility = sdc.getFeasibility()
		if feasibility != 'Optimal':
			logger.info("SDC is ")
			logger.info(feasibility)
		logger.info("SDC_Time: {:.4f}".format(sdc.getTime()))
	logger.info("SDC # of resources: MUL: %d, ALU: %d, Tot: %d. TotLatency: %d" % (MUL_SDC, ALU_SDC, MUL_SDC+ALU_SDC, TotLatency_SDC))

#4. ODG
	graph.ODG_scheduling = True
	step.append(0)
	nrt.append(graph.currNr["MUL"])
	nrta.append(graph.currNr["ALU"])
	nrts.append(graph.currNr["MUL"] + graph.currNr["ALU"])
	lat.append(graph.totLatency)
	timestep = 0
	action=torch.tensor([0], dtype=torch.long)
	batch=torch.tensor([0], dtype=torch.long)
	# Max Timesteps
	NL = graph.vertex ** 2
	iter = (NL if NL > args.timesteps else args.timesteps)
	max_fail_num = iter / 10
	fail_num = 0
	ODG_begin_time = time.time()
	temp_consumption = []
	cum_consumption = 0
	early_stop_flag = False
	ASAP_consumption = MUL_ASAP + ALU_ASAP + TotLatency_ASAP
	while timestep < iter:
		g_data=gdata.Data(x=torch.tensor(graph.vertex_attributes, dtype=torch.float),\
			edge_index=torch.tensor(graph.edge_index, dtype=torch.long),\
			y=action,batch=batch).to(device)
		legalMove = graph.getLegalMove()
		if len(legalMove[0]) == 0:
			print(f'No legal move! Stop in Timestep {timestep}/{iter}!')
			break
		action, log_prob, entropy = agent.get_action(g_data, legalMove)
		if action > -1 * graph.vertex and action < 0:
			node = -1 * action
			CLK = graph.adjlist[node].CLK_start - 1
		elif action >= 0 and action < graph.vertex:
			node = action
			CLK = graph.adjlist[node].CLK_start + 1
		else: #invalid
			node = action
			CLK = 0
		if graph.adjlist[node].vertex_attribute[6] > 10:
			action, _, _ = agent.get_action(g_data, legalMove, reschedule=True)
			if action > -1 * graph.vertex and action < 0:
				node = -1 * action
				CLK = graph.adjlist[node].CLK_start - 1
			elif action >= 0 and action < graph.vertex:
				node = action
				CLK = graph.adjlist[node].CLK_start + 1
			else: #invalid
				node = action
				CLK = 0
		fes, _, _ = graph.schedule_node(node, CLK)

		if fes == False:
			fail_num += 1
			if action >= graph.vertex:
				print("Timestep %d: op %d exceed max idx, not available!" % (timestep+1,action))
		else:
			if timestep % graph.vertex == 0:
				#print(f'Timestep {int(timestep/graph.vertex)}n: reschedule op {action} to CLK {graph.adjlist[action].CLK_start}')
				print(f'Timestep {int(timestep/graph.vertex)}n: Resource: {graph.bestNr["MUL"] + graph.bestNr["ALU"]}, Latency: {graph.get_finalLatency()}')
			step.append(timestep+1)
			nrt.append(graph.currNr["MUL"])
			nrta.append(graph.currNr["ALU"])
			nrts.append(graph.currNr["MUL"] + graph.currNr["ALU"])
			lat.append(graph.totLatency)
		timestep += 1
		# Early Stop
		if fail_num > max_fail_num:
			print(f'Exceed failure tolerance! Early stop at Timestep {timestep}/{iter}!')
			break
		ODG_curr_consumption = graph.currNr["MUL"] + graph.currNr["ALU"] + graph.get_totLatency()
		ASAP_resource_consumption = MUL_ASAP + ALU_ASAP
		ODG_resource_consumption = graph.bestNr["MUL"] + graph.bestNr["ALU"]
		stop_factor = pow(10, ceil(log10(graph.vertex)) - 3)
		stop_step = args.comparison_factor * stop_factor * graph.vertex
		cum_num = args.comparison_factor * graph.vertex
		cum_consumption = 0 if timestep % cum_num == 0 else cum_consumption + ODG_curr_consumption
		if timestep % cum_num == cum_num - 1:
			temp_consumption.append(cum_consumption/float(cum_num))
		
		if timestep >= stop_step:
			if ODG_curr_consumption > ASAP_consumption and ASAP_resource_consumption - ODG_resource_consumption < 3:
				early_stop_flag = True
			elif timestep >= 3 * cum_num and \
				temp_consumption[-1] >= temp_consumption[-2] and temp_consumption[-2] >= temp_consumption[-3]:
				early_stop_flag = True
			if early_stop_flag:
				print(f'Performance degrades! Early stop at Timestep {timestep}/{iter}!')
				break

	end_time = time.time()
	MUL_ODG = graph.bestNr["MUL"]
	ALU_ODG = graph.bestNr["ALU"]
	Latency_ODG = graph.get_finalLatency()
	logger.info("ODG Resource: %d" %(MUL_ODG+ALU_ODG))
	logger.info("ODG TotLatency: %d" %(Latency_ODG))
	logger.info("ODG_Time: {:.4f}".format(end_time - ODG_begin_time))
	print("ODG Resource: %d" %(MUL_ODG+ALU_ODG))
	print("ODG TotLatency: %d" %(Latency_ODG))
	print("ODG_Time: {:.4f}".format(end_time - ODG_begin_time))
	print(f'Testing finished.\n\n')
	return (MUL_ODG, ALU_ODG, Latency_ODG)

def visualization(results):
	res_r = np.array([x[0] for x in results])
	res_l = np.array([x[1] for x in results])
	np.save("./Loss/" + "reward" + file_name + ".npy",res_r)
	np.save("./Loss/" + "loss" + file_name + ".npy",res_l)
	ax1 = fig.add_subplot(111)
	lns1 = ax1.plot(range(len(res_r)),res_r,label="Reward",color="b")
	ax2 = ax1.twinx()
	lns2 = ax2.plot(range(len(res_l)),res_l,label="Loss",color="r")
	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc=0)
	fig.savefig("./Loss/" + "fig" + file_name + ".jpg")
	fig.clf()

# Testing
if args.mode != -1:
	print("Begin testing...")
	agent.policy.eval()
	res = []
	if args.mode == 0:
		for i in range(1,11):
			res.append(test(i))
	else:
		res.append(test(args.mode))
	for i,x in enumerate(res):
		print("Graph %d: Resource: MUL:%d ALU:%d Tot:%d, Latency:%d" % (i,x[0],x[1],x[0]+x[1],x[2]))
	sys.exit()

logger.info("Begin training...")
startTime = time.time()
results = []
for episode in range(1,args.episodes+1):
	results.append(train(episode))
	if hasattr(torch.cuda, 'empty_cache'):
		torch.cuda.empty_cache()
	visualization(results)
	logger.info("Train Episode %d: Avg. Loss: %f, Avg. Reward: %f" % (episode,results[-1][0],results[-1][1]))
	print("Train Episode %d: Avg. Loss: %f, Avg. Reward: %f" % (episode,results[-1][0],results[-1][1]))
	torch.save(agent.policy,"./Networks/policy" + file_name +".pkl")
	usedTime = time.time() - startTime
	print("Finish %d / %d. Total time used: %f min. Rest of time: %f min."
		% (episode,args.episodes,usedTime/60,usedTime/60*args.episodes/episode-usedTime/60))
logger.info("Finish training.")