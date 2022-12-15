import warnings

from tensorflow.keras.callbacks import Callback
import argparse
import config
import os
import server
import logging
from rl.util import *
from tensorflow.keras.models import Sequential
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import rl.callbacks
import matplotlib.ticker as mticker
alpha = 2
accuracy = 0.8
target_accuracy = 0.9

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Flatten(input_shape=(100, 24)))
    model.add(Dense(actions, activation='softmax'))
    return model


def build_agent(model, actions,nb_actions):
    policy = SoftmaxPolicy(nb_actions)
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='/Users/soufiane/PycharmProjects/flsim/configs/MNIST/mnist.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='accuracy', value=0.70, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_step_end(self, epoch, logs={}):
        current = logs["info"]["accuracy"]
        target_accuracy = logs["info"]["target_accuracy"]
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= target_accuracy:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.rewards = []

    def on_episode_begin(self, episode, logs):
        self.rewards = []

    def on_step_end(self, step, logs):
        self.rewards.append(logs['reward'])

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

def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)


    if fl_config.server=="random":

        # Initialize random server
        fl_server = {
            "random": server.Random(fl_config),

        }[fl_config.server]
        fl_server.boot()

        # Run federated learning
        fl_server.run()


        X = fl_server.updated_weights_list + fl_server.weights_list

        pca = PCA(n_components=2)
        y=pca.fit(X).transform(X)
        for i in range(len(y)):
            if i<len(fl_server.updated_weights_list):
                plt.scatter(y[i][0],y[i][1], marker="o")
            else:
                plt.scatter(y[i][0],y[i][1], marker="v")

        plt.title('Fig. 9: PCA on model weights of FL training with \n MNIST. w1, w2 ..., wn are the global model weights.')
        plt.xlabel("C1")
        plt.ylabel("C2")
        plt.show()

        plt.plot(fl_server.accuracies_list,'-bo', label=f'sigma={fl_config.clients.eps}')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 6: Accuracy v.s. communication rounds \n on different levels of non-IID MNIST data.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")

        plt.legend()
        plt.show()



        plt.plot(fl_server.accuracies_list,'-bo',label='K = {} , round # = {} '.format(fl_config.clients.per_round, len(fl_server.accuracies_list)))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 10: FL training on MNIST with different levels of parallelism.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")
        plt.legend()
        plt.show()

    elif fl_config.server=="kcenter":

        # Initialize random server
        fl_server = {
            "kcenter": server.Kcenter(fl_config),

        }[fl_config.server]
        fl_server.boot()

        # Run federated learning
        fl_server.run()


        X = fl_server.updated_weights_list + fl_server.weights_list

        pca = PCA(n_components=2)
        y=pca.fit(X).transform(X)
        for i in range(len(y)):
            if i<len(fl_server.updated_weights_list):
                plt.scatter(y[i][0],y[i][1], marker="o")
            else:
                plt.scatter(y[i][0],y[i][1], marker="v")

        plt.title('Fig. 9: PCA on model weights of FL training with \n MNIST. w1, w2 ..., wn are the global model weights.')
        plt.xlabel("C1")
        plt.ylabel("C2")
        plt.show()

        plt.plot(fl_server.accuracies_list,'-bo', label=f'sigma={fl_config.clients.eps}')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 6: Accuracy v.s. communication rounds \n on different levels of non-IID MNIST data.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")

        plt.legend()
        plt.show()



        plt.plot(fl_server.accuracies_list,'-bo',label='K = {} , round # = {} '.format(fl_config.clients.per_round, len(fl_server.accuracies_list)))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 10: FL training on MNIST with different levels of parallelism.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")
        plt.legend()
        plt.show()

    elif fl_config.server=="basic":

        # Initialize server dqn
        fl_server = {
            "basic": server.Server(fl_config),

        }[fl_config.server]

        # Select node to participate in the round
        states = (1, (fl_config.nodes.total+1)*fl_config.nodes.total)
        actions = fl_config.nodes.total
        model = build_model(states, actions)

        dqn = build_agent(model, actions,fl_config.nodes.per_round)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        # Run federated learning

        episodes = fl_config.episodes
        scores = []
        for episode in range(1, episodes + 1):
            #fl_server.reset()

            callbacks = [
                EarlyStoppingByLossVal(monitor='accuracy', value=0.70,
                                       verbose=5),
                EpisodeLogger()
            ]
            dqn.fit(fl_server, nb_steps=fl_config.fl.rounds, visualize=False, callbacks=callbacks, verbose=5,nb_max_episode_steps=0)
            score = sum(callbacks[1].rewards)
            scores.append(score)
            print('Episode:{} Score:{}'.format(episode, score))

        X = fl_server.updated_weights_list + fl_server.weights_list

        pca = PCA(n_components=2)
        y=pca.fit(X).transform(X)
        for i in range(len(y)):
            if i<len(fl_server.updated_weights_list):
                plt.scatter(y[i][0],y[i][1], marker="o")
            else:
                plt.scatter(y[i][0],y[i][1], marker="v")

        plt.title('Fig. 9: PCA on model weights of FL training with \n MNIST. w1, w2 ..., wn are the global model weights.')
        plt.xlabel("C1")
        plt.ylabel("C2")
        plt.show()

        plt.plot(fl_server.accuracies_list,'-bo', label=f'sigma={fl_config.clients.eps}')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 6: Accuracy v.s. communication rounds \n on different levels of non-IID MNIST data.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")

        plt.legend()
        plt.show()



        plt.plot(fl_server.accuracies_list,'-bo',label='K = {} , round # = {} '.format(fl_config.clients.per_round, len(fl_server.accuracies_list)))
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 10: FL training on MNIST with different levels of parallelism.')
        plt.xlabel("Communication Round (#)")
        plt.ylabel("FL Accuracy (%)")
        plt.legend()
        plt.show()


        plt.plot(scores,'-bo')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('Fig. 5: Training the DRL agent.')
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()






    # Delete global model
    os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
