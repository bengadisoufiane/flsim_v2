import logging
import random
from server import Random
import node
from threading import Thread
from utils.kcenter import GreedyKCenter  # pylint: disable=no-name-in-module


class Kcenter(Random):
    """Federated learning server that performs KCenter profiling during selection."""

    # Run federated learning
    def run(self):
        # Perform profiling on all nodes
        self.profiling()

        # Designate space for storing used node profiles
        self.used_profiles = []

        # Continue federated learning
        super().run()
    def make_nodes(self, num_nodes):



        nodes = []
        for node_id in range(num_nodes):



            fl_config = self.config

            # Initialize nodes
            new_node = {
                "kcenter": node.Kcenter(fl_config,node_id),

            }[fl_config.server]
            new_node.boot()
            nodes.append(new_node)

        self.nodes = nodes

    # Federated learning phases
    def selection(self):
        # Select devices to participate in round

        profiles = self.profiles
        k = self.config.nodes.per_round

        if len(profiles) < k:  # Reuse nodes when needed
            logging.warning('Not enough unused nodes')
            logging.warning('Dumping nodes for reuse')
            self.profiles.extend(self.used_profiles)
            self.used_profiles = []

        # Shuffle profiles
        random.shuffle(profiles)

        # Cluster nodes based on profile weights
        weights = [weight for _, weight in profiles]
        KCenter = GreedyKCenter()
        KCenter.fit(weights, k)

        logging.info('KCenter: {} nodes, {} centers'.format(
            len(profiles), k))

        # Select nodes marked as cluster centers
        centers_index = KCenter.centers_index
        sample_profiles = [profiles[i] for i in centers_index]
        sample_nodes = [node for node, _ in sample_profiles]

        # Mark sample profiles as used
        self.used_profiles.extend(sample_profiles)
        for i in sorted(centers_index, reverse=True):
            del self.profiles[i]

        return sample_nodes

    def profiling(self):
        # Use all nodes for profiling
        nodes = self.nodes

        # Configure nodes for training
        self.configuration(nodes)

        # Train on nodes to generate profile weights
        threads = [node.run() for node in self.nodes]
        [t for t in threads]

        # Recieve node reports
        reports = self.reporting(nodes)

        # Extract weights from reports
        weights = [report.weights for report in reports]
        weights = [self.flatten_weights(weight) for weight in weights]

        # Use weights for node profiles
        self.profiles = [(node, weights[i])
                         for i, node in enumerate(nodes)]
        return self.profiles

