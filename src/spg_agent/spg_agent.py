import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../../spg'))
from spg.memory import Memory as ReplayBuffer  # TODO replace this too


sys.path.insert(0, os.path.join(head, '.'))
from spg_arch import SPGSequentialActor
from spg_arch import SPGSequentialCritic

from collections import deque
import numpy as np
import torch
import torch.optim as optim


class SPGAgent():

    def __init__(self, agent_config=None):

        self.agent_config = agent_config

        self.N, self.K = self.agent_config['N'], self.agent_config['K']
        embed_dim = self.agent_config['embed_dim']
        rnn_dim = self.agent_config['rnn_dim']

        # actor, critic
        self.actor = SPGSequentialActor(self.N, self.K, embed_dim, rnn_dim, 
                                        sinkhorn_rds=agent_config['sinkhorn_rds'], 
                                        sinkhorn_tau=agent_config['sinkhorn_tau'], 
                                        n_workers=agent_config['n_workers'], 
                                        bidirectional=True)

        self.critic = SPGSequentialCritic(self.N, self.K, embed_dim, rnn_dim, bidirectional=True)

        # objectives, optimizers, schedulers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.agent_config['actor_lr'])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.agent_config['critic_lr'])
        self.critic_obj = torch.nn.MSELoss()
        self.critic_aux_obj = torch.nn.MSELoss()

        actor_decay_step, actor_decay_rate = self.agent_config['actor_lr_decay_step'], self.agent_config['actor_lr_decay_rate']                                            
        critic_decay_step, critic_decay_rate = self.agent_config['critic_lr_decay_step'], self.agent_config['critic_lr_decay_rate']
        self.actor_scheduler = optim.lr_scheduler.MultiStepLR(self.actor_optim, range(actor_decay_step, actor_decay_step * 1000, actor_decay_step), gamma=actor_decay_rate) # idea of step scheduler: at each range val, decay lr by lr*gamma
        self.critic_scheduler = optim.lr_scheduler.MultiStepLR(self.critic_optim, range(critic_decay_step, critic_decay_step * 1000, critic_decay_step), gamma=critic_decay_rate)
                                                                
        self.replay_buffer = ReplayBuffer(self.agent_config['buffer_size'], action_shape=[self.N,self.N], observation_shape=[self.N,self.K], use_cuda=False)

        # exploration
        self.epsilon = self.agent_config['epsilon']
        epsilon_target = self.agent_config['epsilon_target']
        epsilon_decay_steps = self.agent_config['epsilon_decay_steps']
        self.epsilon_decay = (epsilon_target - self.epsilon) / epsilon_decay_steps 

        self.actor.train(); self.critic.train()

    def get_action(self, agent_state):
            
            M, P = self.actor(agent_state)

            # epsilon greedy exploration - 2-exchange adapted from TSP search strategies
            if np.random.rand() < self.epsilon:
                for _ in range(2):
                    # randomly choose two row idxs
                    idxs = np.random.randint(0, self.N, size=2)
                    # swap the two rows
                    tmp0 = P[:, idxs[0]].clone() # TODO this indexing is incorrect with batch dim
                    tmp1 = P[:, idxs[1]].clone()
                    P[:, idxs[0]] = tmp1
                    P[:, idxs[1]] = tmp0
                    tmp0 = M[:, idxs[0]].clone()
                    tmp1 = M[:, idxs[1]].clone()
                    M[:, idxs[0]] = tmp1
                    M[:, idxs[1]] = tmp0

            self.epsilon += self.epsilon_decay

            action = torch.matmul(P.squeeze(0), agent_state.squeeze(0)).byte().numpy() # remove batch, as an integer array

            terminal_idx = np.squeeze(np.argwhere(action[:,0]==1)[0]) # 1st noop, noop is 1
                                                                      # TODO don't have this hardcoded

            return dict(agent_action=action[:terminal_idx, 0], M=M, P=P) # TODO require M, P for replay buffer but havent seen reward

    def observe(self, agent_state, agent_action, agent_reward):

        # store in replay -- TODO add this to training loop in a separate call?  
        M, P = agent_action['M'], agent_action['P']
        R = torch.tensor(agent_reward, requires_grad=False).view(1,-1) # add a batch_dim
        self.replay_buffer.append(agent_state.data, P.data.byte(), M.data, R.data) # don't think .data is required?

        if self.replay_buffer.nb_entries > 16:
            s_batch, P_batch, M_batch, r_batch = self.replay_buffer.sample(16)
            P_batch = P_batch.float()
            targets = r_batch

            # critic update
            hard_Q = self.critic(s_batch, P_batch).squeeze(2) # [b, 1, 1] to [b, 1]
            critic_loss = self.critic_obj(hard_Q, targets) # loss
            soft_Q = self.critic(s_batch, M_batch).squeeze(2)
            critic_aux_loss = self.critic_aux_obj(soft_Q, hard_Q.detach()) # de-biasing loss
            self.critic_optim.zero_grad() 
            (critic_loss + critic_aux_loss).backward() 
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, norm_type=2) #
            self.critic_optim.step()
            self.critic_scheduler.step()                 
            self.critic_optim.zero_grad()                

            # actor update
            self.actor_optim.zero_grad()
            soft_action, _ = self.actor(s_batch, round=False)
            soft_critic_loss = self.critic(s_batch, soft_action).squeeze(2).mean()
            actor_loss = -soft_critic_loss
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0, norm_type=2)
            self.actor_optim.step()
            self.actor_scheduler.step()