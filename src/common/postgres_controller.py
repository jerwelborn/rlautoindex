#!/usr/local/bin/python3

import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.controller.system_controller import SystemController

sys.path.insert(0, os.path.join(head, '..')) # src
from dqn_agent.postgres_converter import PostgresConverter as PostgresDQNConverter
from dqn_agent.postgres_schema import PostgresSchema as PostgresDQNSchema
from spg_agent.postgres_converter import PostgresConverter as PostgresSPGConverter
from spg_agent.postgres_schema import PostgresSchema as PostgresSPGSchema


from postgres_agent import PostgresAgent
from postgres_data_source import PostgresDataSource
from postgres_system_environment import Action, PostgresSystemEnvironment
from tpch_workload import TPCHWorkload

import csv
import numpy as np
import pdb
import pickle
import time

class PostgresSystemController(SystemController):
    """
    Abstraction for training + testing 
    
    Encapsulates a slew of other abstractions: agent + system + agent-system auxiliaries like converter and schema
    
    Some of these are shared by DQN + SPG agents, while some are not shared, so there's a bit of brittleness in how this is architected,
    though should be fairly straightforward to follow.

    """

    def __init__(self, 
                 dqn,
                 agent_config, 
                 experiment_config, 
                 schema_config,
                 result_dir):
        """
        
        Args:
            dqn (bool): DQN agent, else SPG agent; required where rlgraph breaks abstraction barrier (e.g. resetting)
            ...

             TODO 'config' and 'spec' conflated throughout code
        """
        self.dqn = dqn

        super().__init__(agent_config=agent_config, 
                         experiment_config=experiment_config,
                         result_dir=result_dir)    

        #
        # schema, for agent state and action representations 
        # 
        schema_config['tables'] = experiment_config['tables']
        self.schema_config = schema_config
        
        if self.dqn:
            self.schema = PostgresDQNSchema(schema_config)
        else:
            self.schema = PostgresSPGSchema(schema_config)

        self.states_spec = self.schema.get_states_spec()
        self.actions_spec = self.schema.get_actions_spec()
        self.system_spec = self.schema.get_system_spec()

        if self.dqn: self.agent_config['network_spec'][0]['vocab_size'] = self.system_spec['vocab_size'] # TODO

        #
        # converter
        #
        if self.dqn:
            self.converter = PostgresDQNConverter(experiment_config=experiment_config, schema=self.schema)
        else:
            self.converter = PostgresSPGConverter(experiment_config=experiment_config, schema=self.schema)

        #
        # workload
        # 
        n_selections = experiment_config.get("n_selections", 3)
                
        self.n_executions = experiment_config['n_executions']
        self.n_train_episodes = experiment_config["n_train_episodes"]
        self.n_test_episodes = experiment_config["n_test_episodes"]
        self.n_queries_per_episode = experiment_config["n_queries_per_episode"]

        workload_spec = {
            "tables": self.schema_config['tables'],
            "scale_factor": 1, # TODO specify this somewhere, not just in tpch_util 
            "n_selections": n_selections
        }
        self.workload = TPCHWorkload(spec=workload_spec)

        #
        # workload ser/des
        #
        self.data_source = PostgresDataSource(workload_spec)

        #
        # environment
        #
        self.system_environment = PostgresSystemEnvironment(tables=self.experiment_config['tables'])

        #
        # agent
        #
        self.agent_api = PostgresAgent(
            dqn=self.dqn,
            agent_config=self.agent_config,
            experiment_config=self.experiment_config,
            schema = self.schema
        )
        
        if self.dqn:
            self.task_graph = self.agent_api.task_graph # have to have this exposed...
            self.set_update_schedule(self.agent_config["update_spec"])

            # TODO:
            # per description in docstring, we can't hide all rlgraph agent behind a shared dqn/spg api:
            # self.set_update_schedule(), self.agent.timesteps, self.update_if_necessary() called from super class 
            # require access to rlpgraph *directly*


    def generate_workload(self, export=False):
        """
        """
        n_train_queries = self.n_train_episodes * self.n_queries_per_episode
        n_test_queries = self.n_test_episodes * self.n_queries_per_episode

        train_queries = [self.workload.generate_query_template() for _ in range(n_train_queries)]
        test_queries = [self.workload.generate_query_template() for _ in range(n_test_queries)]

        if export:
            self.data_source.export_data(train_queries, self.result_dir, label='train')
            self.data_source.export_data(test_queries, self.result_dir, label='test')

        return train_queries, test_queries
    
    def restore_workload(self):
        """
        """
        train_queries = self.data_source.import_data(self.result_dir, label="train")
        test_queries = self.data_source.import_data(self.result_dir, label="test")

        return train_queries, test_queries

    def train(self, queries):
        """
            trains in same MDP-style as 3.1 in Sutton-Barto

            Args:
                queries (list): SQLQueries, each episode exposes a slice of queries (query templates) 
                label (str): indicates whether spg or dqn

            TODO:
                recording of meta-data for an episode / episodes reduces code readability, not sure if there's a clean workaround
        """

        # meta-data
        result_dir = os.path.join(self.result_dir, 'dqn' if self.dqn else 'spg')
        if not os.path.exists(result_dir): os.makedirs(result_dir)

        # record per step 
        times = [] # total time, roughly decomposes into...
        agent_times = [] # time spent retrieving action, updating based on retrieved action
        query_times = [] # time spent on queries
        system_times = [] # time spent on transitioning state (i.e. indexing)
        rewards = [] # for trend over time

        # record per episode of steps
        index_set_stats = [] # distribution of action decisions over steps, of # of query attributes per step, the total size

        if self.dqn: self.task_graph.get_task("").unwrap().timesteps = 0 # TODO belongs in rlgraph

        for episode_idx in range(self.n_train_episodes):
            
            self.logger.info('Starting episode {}/{}'.format(episode_idx+1, self.n_train_episodes))

            # meta-data
            action_results = dict.fromkeys(['noop', 'duplicate_index', 'index'], 0)
            action_results_szs = []

            # queries + context
            episode_queries = queries[episode_idx * self.n_queries_per_episode:
                              (episode_idx + 1) * self.n_queries_per_episode]
            terminal_query_idx = len(episode_queries) - 1
            context = [] # list of list of attributes, corresponding to compound indices
        
            if self.dqn: self.task_graph.reset()
            self.system_environment.reset()
            
            for query_idx, query in enumerate(episode_queries):

                start = time.monotonic()

                if query_idx+1 % 5 == 0:
                    self.logger.info('Completed {}/{} queries'.format(query_idx+1,self.n_queries_per_episode))

                ##
                # get state
                ##
                acting_start = time.monotonic() 
                agent_state = self.converter.system_to_agent_state(query=query, 
                                                                   system_context=dict(indices=context))

                ##
                # get action for state
                ##
                agent_action = self.agent_api.get_action(agent_state)

                system_action = self.converter.agent_to_system_action(actions=agent_action, 
                                                                      meta_data=dict(query_cols=query.query_cols))
                acting_time = time.monotonic() - acting_start 
 
                ## 
                # take action (i.e. add index) and record reward (i.e. from query under updated index set)
                ##
                system_time, action_result = self.system_environment.act(dict(index=system_action, table=query.query_tbl))
                action_results[action_result] += 1
                action_results_szs.append(len(system_action))

                query_string, query_string_args = query.sample_query()
                query_time = self.system_environment.execute(query_string=query_string, query_string_args=query_string_args)
    
                index_set_size, _ = self.system_environment.system_status()
                
                ##
                # update agent
                ##
                updating_start = time.monotonic()
                agent_reward = self.converter.system_to_agent_reward(meta_data=dict(runtime=query_time,
                                                                                    index_size=index_set_size)) # TODO replace meta_data arg
                context.append(system_action)
                next_agent_state = terminal = None # required because bootstrapping
                if self.dqn: 
                    terminal = query_idx == terminal_query_idx
                    next_query = episode_queries[query_idx + 1 if not terminal else query_idx]
                    next_agent_state = self.converter.system_to_agent_state(query=next_query, 
                                                                        system_context=dict(indices=context))
                
                self.agent_api.observe(agent_state, agent_action, agent_reward, next_agent_state=next_agent_state, terminal=terminal)
                self.update_if_necessary()

                updating_time = time.monotonic() - updating_start

                # record results per step of episode
                times.append(time.monotonic() - start)
                agent_times.append(acting_time + updating_time)
                query_times.append(query_time)
                system_times.append(system_time)

                rewards.append(agent_reward)

            # record results per episode
            for key in action_results.keys(): action_results[key] = action_results[key] / self.n_queries_per_episode
            index_set_stats.append((action_results, np.mean(action_results_szs), index_set_size)) 

            # pretty print a summary
            self.logger.info("Completed episode: " \
                             "actions taken: " + ('{}:{:.2f} ' * len(tuple(action_results.items()))).format(*tuple(field for tup in action_results.items() for field in tup)) + \
                             "avg reward per step: {:.2f} ".format(np.mean(rewards[:-self.n_queries_per_episode])) + \
                             "avg query runtime: {:.2f} ".format(np.mean(query_times[:-self.n_queries_per_episode])) + \
                             "index set size {}/{:.0f}".format(len(self.system_environment.index_set), index_set_size))  
                                 

        # record results for all episodes 
        np.savetxt(os.path.join(result_dir, 'train_times.txt'), np.asarray(times), delimiter=',') # TODO replace np.savetxt with np.save
        np.savetxt(os.path.join(result_dir, 'train_agent_times.txt'), np.asarray(agent_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_system_times.txt'), np.asarray(system_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_query_times.txt'), np.asarray(query_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_rewards.txt'), np.asarray(rewards), delimiter=',')
        
        with open(os.path.join(result_dir, 'train_index_set_stats.csv'), 'a') as f:
            writer = csv.writer(f)
            for episode in index_set_stats:
                writer.writerow([*tuple(episode[0].values()), episode[1], episode[2]])
        
    def act(self, queries):
        """
        Emulate an episode on queries (i.e. test queries) without training to accumulate system actions

        Args:
            queries (list): SQLQuery objects 

        Returns:
            system actions (dict): records actions taken per query / query_idx
        """

        context = []
        system_actions = {}

        for query_idx, query in enumerate(queries):

            agent_state = self.converter.system_to_agent_state(query=query, 
                                                               system_context=dict(indices=context))
            agent_action = self.agent_api.get_action(agent_state)
            
            system_action = self.converter.agent_to_system_action(actions=agent_action, 
                                                                  meta_data=dict(query_cols=query.query_cols))
            self.system_environment.act(dict(index=system_action, table=query.query_tbl))

            system_actions[query_idx] = system_action
            context.append(system_action)
        
        return system_actions

    def evaluate(self, queries, actions={}, export=False):
        """
        Evaluates the execution of a set of queries (i.e. test queries).
        Assumes indices of interest are already set up. 

        Args:
            queries (list): SQLQuery objects to execute and evaluate
        """

        runtimes = []
        for query in queries:
            
            runtimes_per_query = []

            for _ in range(self.n_executions):
                
                query_string, query_string_args = query.sample_query()
                query_time = self.system_environment.execute(query_string, query_string_args)                
                
                runtimes_per_query.append(query_time)

            runtimes.append(runtimes_per_query)

        index_set_size, index_set = self.system_environment.system_status()


        if export:
            tag = 'default' if actions == {} else 'dqn' if self.dqn else 'spg' # yikes
            result_dir = os.path.join(self.result_dir, tag)
            if not os.path.exists(result_dir): os.makedirs(result_dir)

            # runtimes
            np.savetxt(os.path.join(result_dir, 'test_query_times.txt'), np.asarray(runtimes), delimiter=',')
            
            # index set, index set size 
            with open(os.path.join(result_dir, 'test_index_set_stats.csv'), 'wb') as f:
                pickle.dump([index_set_size, index_set], f)
             
            # queries w/ any action taken 
            with open(os.path.join(result_dir, 'test_by_query.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                for query_idx, query in enumerate(queries):

                    writer.writerow([query.query_string, actions.get(query_idx, ''), np.mean(runtimes[query_idx])])




# TODO split out script
def main(argv):

    import gflags
    import json

    #
    # logging
    # 
    import logging # import before rlgraph
    format_ = '%(levelname)s / %(module)s / %(message)s\n'
    formatter = logging.Formatter(format_)
    if logging.root.handlers == []:
    	handler = logging.StreamHandler(stream=sys.stdout) # no different from basicConfig()
    	handler.setFormatter(formatter)
    	handler.setLevel(logging.DEBUG)
    	root.addHandler(handler)
    else:
	# handler has set up by default in rlgraph
        logging.root.handlers[0].setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.info('Starting controller script...')


    #
    # parsing - TODO command-line arg semantics aren't super clean
    #
    FLAGS = gflags.FLAGS
    
    # baselines rely on controller, which requires an agent, so set up dqn by default  
    gflags.DEFINE_boolean('dqn', True, 'dqn or spg, dqn by default for non-agent')
    gflags.DEFINE_string('config', 'conf/dqn.json', 'config for agent, agent representations')
    gflags.DEFINE_string('experiment_config', '', 'config for experiment')
    gflags.DEFINE_boolean('generate_workload', True, 'set True to build train and test workloads')
    gflags.DEFINE_string('result_dir', '../res/', 'base directory for workload and workload results')
    gflags.DEFINE_boolean('with_agent', True, 'set True for agent, False for non-agent baseline')
    
    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.config, 'r') as fh:
        config = json.load(fh)
    with open(FLAGS.experiment_config, 'r') as fh:
        experiment_config = json.load(fh)
    agent_config, schema_config = config['agent'], config['schema']
    logger.info('agent config: {}'.format(agent_config))
    logger.info('schema config: {}'.format(schema_config))
    logger.info('experiment config: {}'.format(experiment_config))

    result_dir = FLAGS.result_dir

    # build workload? or restore built workload?
    # if FLAGS.generate_workload:
    #     # if creating workload i.e. if starting a set of experiments, add a timestamp
    #     timestamp = time.strftime("%m-%d-%y_%H:%M", time.localtime())
    #     result_dir = os.path.join(result_dir, timestamp)
    #     if not os.path.exists(result_dir): os.makedirs(result_dir)

    
    # n.b. controller is required for agent-advised index or baseline index, b/c wraps system, workload
    controller = PostgresSystemController(
        dqn=FLAGS.dqn,
        agent_config=agent_config,
        schema_config=schema_config,
        experiment_config=experiment_config,
        result_dir=result_dir
    )

    # TODO crude way to suppress stdout from system_environment, this should be in a config somewhere 
    controller.system_environment.logger.setLevel(logging.WARNING) 

    # generate workload for first of a few experiments...
    if FLAGS.generate_workload:
        train, test = controller.generate_workload(export=True) 
    else:
        train, test = controller.restore_workload()
        # TODO! copy workload config?

    # ...which are one of
    
    # evaluate default i.e. 1º keys 
    if not FLAGS.with_agent: 
        controller.reset_system() 
        controller.evaluate(test, actions={}, export=True)

    # train agent, then evaluate agent's index
    elif FLAGS.with_agent:
        
        logger.info('TRAINING')
        controller.train(train)
        controller.system_environment.reset()

        logger.info('TESTING')
        system_actions = controller.act(test)
        controller.evaluate(test, actions=system_actions, export=True)
    
    logger.info('RAN TO COMPLETION')

if __name__ == "__main__":
    main(sys.argv)
