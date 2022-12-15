import load_data
from sklearn.decomposition import PCA
import utils.dists as dists  # pylint: disable=no-name-in-module
import logging
import sys
import pytz
from threading import Thread
from datetime import datetime
import client
import torch
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import time
import random
from rl.util import *
alpha = 2




def current_time():
    tz_NY = pytz.timezone('America/New_York')
    datetime_NY = datetime.now(tz_NY)
    return datetime_NY.strftime("%m_%d_%H:%M:%S")


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages


def calculate_reward(accuracy, target_accuracy):
    return round((accuracy - target_accuracy) * 1000, 2)


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages




class Node(object):
    """Basic federated learning node."""

    def __init__(self, config,node_id):
        self.latency = None
        self.config = config
        self.node_id = node_id
        self.accuracies_list=[]
    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.node_id, len(self.data), set([label for _, label in self.data]))




    # Set up node
    def boot(self):
        logging.info('Booting {} node...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Set up simulated node
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)

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

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.debug('Labels ({}): {}'.format(
            len(labels), labels))

        # Set up data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[self.config.loader]

        logging.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, model_path)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Make simulated clients
        clients = []


        for client_id in range(num_clients):

            # Create new client
            new_client = client.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    # Run federated learning
    def pca(self):
        X = [self.updated_weights] + self.weights
        #X = np.transpose(X)

        pca = PCA(n_components=self.config.clients.total)
        return pca.fit(X).transform(X)


    def step(self,action):
        import fl_model  # pylint: disable=import-error
        start = time.perf_counter()
        # Configure sample clients
        target_accuracy = self.config.fl.target_accuracy
        sample_clients = self.selection(action)
        self.configuration(sample_clients)
        self.first_action = False

        # Run clients using multithreading for better parallelism
        #threads = [client.run() for client in sample_clients]
        #[t for t in threads]
        #[t.join() for t in threads]
        threads = [Thread(target=client.run()) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(self.clients)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        self.weights = [self.flatten_weights(weight) for weight in weights]

        # Perform weight aggregation
        logging.info('Aggregating updates')

        self.reports = self.reporting(sample_clients)
        updated_weights = self.aggregation(self.reports)

        # Generate report for server
        self.report = Report(self)
        self.report.weights = updated_weights
        self.report.num_samples = sample_clients[0].get_report().num_samples


        self.updated_weights=self.flatten_weights(updated_weights)
        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        self.state=self.pca().flatten()


        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
            self.accuracy = accuracy
        else:  # Test updated model on node
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)
            self.accuracy = accuracy

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        self.accuracies_list.append(accuracy)


        # Break loop when target accuracy is met
        if target_accuracy and (accuracy >= target_accuracy):
            logging.info('Target accuracy reached.')
            done = True
        else:
            done = False
        # latency
        latency = [client.latency for client in sample_clients]

        latency = np.array([float(i)/max(latency) for i in latency]).mean()

        reward = alpha ** (accuracy - target_accuracy) - 1 - latency


        info = {"accuracy":self.accuracy,
                "target_accuracy":target_accuracy}

        # Return step information
        self.latency = time.perf_counter() - start

        return self.state, reward, done, info

    def state(self):

        weight_vecs = self.model_weights()




    # Federated learning phases

    def selection(self,action):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select node randomly
        if self.first_action:
            sample_clients = [client for client in self.clients]
        else:
            # epsilon greedy
            p = np.random.random()
            if p < self.config.clients.eps:
                sample_clients = [client for client in random.sample(self.clients, clients_per_round)]
            else:
                sample_clients = [self.clients[action_id] for  action_id in action]


        return sample_clients

    def configuration(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config,self.node_id)


    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_client_updates(self, reports):
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
        updates = self.extract_client_updates(reports)

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

    # node operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition.get('size'):
                partition_size = self.config.data.partition.get('size')
            elif self.config.data.partition.get('range'):
                start, stop = self.config.data.partition.get('range')
                partition_size = random.randint(start, stop)

        # Extract data partition for client
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/node_'+str(self.node_id)
        torch.save(model.state_dict(), path)
        logging.info('Saved node model: {}'.format(path))

    def save_reports(self, round, reports):
        import fl_model  # pylint: disable=import-error

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            fl_model.extract_weights(self.model))


    """Simulated federated learning node."""



    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv

    # Federated learning phases
    def set_data(self, data, config):
        # Extract from config
        do_test = self.do_test = config.nodes.do_test
        test_partition = self.test_partition = config.nodes.test_partition

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        data = self.data
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

    def configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        model_path = self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent global model
        path = model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)


    def get_report(self):
        # Report results to server.
        return self.upload(self.report)



    def test(self):
        # Perform model testing
        raise NotImplementedError
    def reset(self):
        logging.info('Reset FL node...')
        self.state = [1 for i in range((self.config.clients.total+1)*self.config.clients.total)]
        self.first_action=True

        return self.state


class Report(object):
    """Federated learning node report."""

    def __init__(self, node):
        self.node_id = node.node_id







