import tensorflow as tf
from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError

results_path="primal_results"
environment_path="saved_environments"
if not os.path.exists(results_path):
    os.makedirs(results_path)

class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path,grid_size):
        self.grid_size=grid_size
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=config)
        self.network=ACNet("global",5,None,False,grid_size,"global")
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess,ckpt.model_checkpoint_path)
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.size=gym.SIZE
        self.env=gym
        
    def step_all_parallel(self):
        action_probs=[None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        #parallel inference
        actions=[]
        inputs=[]
        goal_pos=[]
        for agent in range(1,self.num_agents+1):
            o=self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        #compute up to LSTM in parallel
        h3_vec = self.sess.run([self.network.h3], 
                                         feed_dict={self.network.inputs:inputs,
                                                    self.network.goal_pos:goal_pos})
        h3_vec=h3_vec[0]
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                         feed_dict={self.network.inputs:[inputs[a]],
                                                    self.network.h3:[h3_vec[a]],
                                                    self.network.state_in[0]:rnn_state[0],
                                                    self.network.state_in[1]:rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state
        #now finish in parallel
        policy_vec=self.sess.run([self.network.policy], 
                                         feed_dict={self.network.rnn_out:rnn_out})
        policy_vec=policy_vec[0]
        for agent in range(1,self.num_agents+1):
            action=np.argmax(policy_vec[agent-1])
            self.env._step((agent,action))
          
    def find_path(self,max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution=[]
        step=0
        while((not self.env._complete()) and step<max_step):
            timestep=[]
            for agent in range(1,self.env.num_agents+1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            self.step_all_parallel()
            step+=1
            #print(step)
        if step==max_step:
            raise OutOfTimeError
        for agent in range(1,self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))
        return np.array(solution)
    
def make_name(n,s,d,id,extension,dirname,extra=""):
    if extra=="":
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}{}".format(n,s,d,id,extension)
    else:
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}_{}{}".format(n,s,d,id,extra,extension)
    
def run_simulations(next,primal):
    #txt file: planning time, crash, nsteps, finished
    (n,s,d,id) = next
    environment_data_filename=make_name(n,s,d,id,".npy",environment_path,extra="environment")
    world=np.load(environment_data_filename)
    gym=mapf_gym.MAPFEnv(num_agents=n, world0=world[0],goals0=world[1])
    primal.set_env(gym)
    solution_filename=make_name(n,s,d,id,".npy",results_path,extra="solution")
    txt_filename=make_name(n,s,d,id,".txt",results_path)
    world=gym.getObstacleMap()
    start_positions=tuple(gym.getPositions())
    goals=tuple(gym.getGoals())
    start_time=time.time()
    results=dict()
    start_time=time.time()
    try:
        #print('Starting test ({},{},{},{})'.format(n,s,d,id))
        path=primal.find_path(256 + 128*int(s>=80) + 128*int(s>=160))
        results['finished']=True
        results['time']=time.time()-start_time
        results['length']=len(path)
        np.save(solution_filename,path)
    except OutOfTimeError:
        results['time']=time.time()-start_time
        results['finished']=False
    results['crashed']=False
    f=open(txt_filename,'w')
    f.write(json.dumps(results))
    f.close()

if __name__ == "__main__":
#    import sys
#    num_agents = int(sys.argv[1])

    primal=PRIMAL('model_primal',10)
    num_agents = 2

    while num_agents < 1024:
        num_agents *= 2

        print("Starting tests for %d agents" % num_agents)
        for size in [10,20,40,80,160]:
            if size==10 and num_agents>32:continue
            if size==20 and num_agents>128:continue
            if size==40 and num_agents>512:continue
            for density in [0,.1,.2,.3]:
                for iter in range(100):
                    run_simulations((num_agents,size,density,iter),primal)
print("finished all tests!")
