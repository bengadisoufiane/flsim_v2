import node
import load_data
import logging
import numpy as np
from sklearn.decomposition import PCA
import pickle
import random
import sys
from rl.util import *
import torch
import utils.dists as dists  # pylint: disable=no-name-in-module
from gym import Env
from gym.spaces import Box, Discrete
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
alpha = 2

class Random(object):
    """Basic federated learning server."""

    def __init__(self, config):
        self.config = config
        self.accuracies_list=[]
        self.weights_list=[]
        self.updated_weights_list=[]

    # Set up server
    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_nodes = self.config.nodes.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated server
        self.load_data()
        self.load_model()
        self.make_nodes(total_nodes)

    def load_data(self):
        import fl_model  # pylint: disable=import-error

        # Extract config for loaders
        config = self.config

        # Set up data generator
        generator = fl_model.Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels



        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]



    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path,False)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_nodes(self, num_nodes):



        nodes = []
        for node_id in range(num_nodes):



            fl_config = self.config

            # Initialize nodes
            new_node = {
                "random": node.Random(fl_config,node_id),

            }[fl_config.server]
            new_node.boot()
            nodes.append(new_node)

        self.nodes = nodes



    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            self.accuracies_list.append(accuracy)
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def round(self):
        import fl_model  # pylint: disable=import-error

        # Select nodes to participate in the round
        sample_nodes = self.selection()

        # Configure sample nodes
        self.configuration(sample_nodes)

        # Run nodes using multithreading for better parallelism
        threads = [node.run() for node in sample_nodes]
        [t for t in threads]


        # Recieve node updates
        reports = self.reporting(sample_nodes)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        self.weights = [self.flatten_weights(weight) for weight in weights]

        self.weights_list.extend(self.weights)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        self.updated_weights = self.flatten_weights(updated_weights)

        self.updated_weights_list.append(self.updated_weights)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model,True)

        # Test global model accuracy
        if self.config.nodes.do_test:  # Get average accuracy from node reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy

    def state(self):

        weight_vecs = self.model_weights(nodes)




    # Federated learning phases
    def selection(self):
        # Select devices to participate in round
        # Select node randomly
        nodes_per_round = self.config.nodes.per_round
        sample_nodes = [node for node in random.sample(self.nodes, nodes_per_round)]

        return sample_nodes

    def configuration(self, sample_nodes):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected node for federated learning task
        for node in sample_nodes:
            if loading == 'dynamic':
                self.set_node_data(node)  # Send data partition to node

            # Extract config for node
            config = self.config

            # Continue configuraion on node
            node.configure(config)

    def reporting(self, sample_nodes):
        # Recieve reports from sample node


        reports = [node.get_report() for node in sample_nodes]
        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_nodes)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_node_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]


        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_node_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_node_data(self, node):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for node
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, node.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to node
        node.set_data(data, self.config)

    def save_model(self, model, path,flag):
        path_global = path + '/global'
        torch.save(model.state_dict(), path_global)
        if flag:
            for node in self.nodes:
                path_node = path+'/node_'+str(node.node_id)
                torch.save(model.state_dict(), path_node)

        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.node_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))
    def reset(self):
        logging.info('Reset FL Server...')
        self.state = [1 for i in range((self.config.nodes.total+1)*self.config.nodes.total)]
        self.first_action=True
        self.boot()
        return self.state

class Policy(object):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:
    - `select_action`
    # Arguments
        agent (rl.core.Agent): Agent used
    """

    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy
        # Returns
            Configuration as dict
        """
        return {}


class SoftmaxPolicy(Policy):
    """ Implement softmax policy for multinimial distribution
    Simple Policy
    - takes action according to the pobability distribution
    """

    def __init__(self,nb_actions):
        self.nb_actions = nb_actions

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            probs (np.ndarray) : Probabilty for each action
        # Returns
            action
        """

        # action = np.random.choice(range(nb_actions),2, p=q_values)
        action = np.argsort(q_values)[-self.nb_actions:]
        return action

