# SPG
import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../')) # /src
from spg_agent.spg_agent import SPGAgent

# DQN
from rlgraph.agents import Agent

sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.task import Task # update
from lift.lift.rl_model.task_graph import TaskGraph


class PostgresAgent():
    """
        Wrapper for DQN agent (from rlgraph) or SPG agent,
        a standard api that can be called by system controller
    
    """

    def __init__(self, dqn=True, agent_config=None, experiment_config=None, schema=None):
        
        self.agent_config = agent_config
        
        self.dqn = dqn
        if self.dqn:

            # update: wrapper wraps (Lift) TaskGraph which has a (Lift) Task which has a (Rlgraph) Agent
            self.task_graph = TaskGraph()
            task = Task(self.agent_config, 
                        state_space=schema.get_states_spec(), 
                        action_space=schema.get_actions_spec())
            self.task_graph.add_task(task)

            # TODO DELETE
            # self.agent = Agent.from_spec(
            #     self.agent_config,
            #     state_space=states_spec,
            #     action_space=actions_spec
            # )
        
        else:             
            agent_config['N'], agent_config['K'] = schema.N, schema.K
            self.agent = SPGAgent(agent_config=agent_config)
            
    def get_action(self, agent_state):

        if self.dqn:
            return self.task_graph.act_task("", states=agent_state.as_array(), apply_preprocessing=True)
            
            #self.agent.get_action(agent_state.as_array(), apply_preprocessing=True)

        else:
            return self.agent.get_action(agent_state)


    def observe(self, agent_state, agent_action, agent_reward, next_agent_state=None, terminal=None):
        """
        Args:
            agent_state: both
            agent_action: both
            agent_reward: both
            next_agent_state: DQN-specific
            terminal: DQN-specific
        """
        
        if self.dqn:
            self.task_graph.observe_task("", agent_state.as_array(), agent_action, [], agent_reward, next_agent_state.as_array(), terminal)
            #self.agent.observe(agent_state.as_array(), agent_action, [], agent_reward, next_agent_state.as_array(), terminal)
        
        else:
            self.agent.observe(agent_state, agent_action, agent_reward)