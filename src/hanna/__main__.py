import argparse
import numpy as np
import tensorflow as tf

from hanna.svm import SVM


class Agent():
    def __init__(self, _domain):
        model = tf.keras.model.Sequential([
            tf.keras.layers.Dense(49)
            tf.keras.layers.Dense(100)
            tf.keras.layers.Dense(1)
        ])

        self.nn = model.compile(loss="mse", optimizer="sgd")
        self.domain = _domain

    def train(self, x, y):
        done = False

        while not done:
            self.nn.fit(x, y, epochs=1, verbose=0)

            errors = [1 if pred != y[i] else 0 for i, pred in self.nn.predict(x)]

            if sum(errors) == 0:
                done = True

    def perceive(self, example):
        return self.nn.predict(example)


def distance(a1, a2, examples):
    return np.sum([(a1.perceive(ex) - a2.perceive(ex)) ** 2 for ex in examples]) / len(examples)


def iteration(args):
    """ Single iteration of the loop. """

    # -- world space
    n = 10
    # -- gather n samples from inside the agents domain
    # -- gather n samples from outside the agents domain

    for agent in agents:
        x = np.append(x_inside, x_outside)

        y_inside = np.ones(n) * 1 if agent.domain == 'row' else -1
        y_outside = np.ones(n) * 1 if agent.domain == 'random' else -1

        y = np.append(y_inside, y_outside)
        # -- individual space

        # -- train the neural net until it converges.
        # -- in other words till it tells the difference between the two cultures.
        agent.train(x, y)

    # -- do we build a full plan here?

    # -- agent communicates its position by point towards the nearest ideal in the world space.
    # -- this position is calculated the distances between the samples and what the agents sees as the current mean

    # -- D(a1, a2) = Σ i = 1 : numExamples [pa1(exi) – pa2(exi)]2 / numExamples
    # -- take two agents and check their perceiveions each example examples, square the difference, sum it and divide by the total examples.

    social_distances = np.zeros((args.agents, args.agents))

    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            social_distances[i, j] = distance(a1, a2, examples)  # which examples?

    # what is the boundary of who is close enough?

    # -- social space

    # -- where does the agent share its current ideal?
    # -- select new design from all the designs of the field.


def main(args=None):
    """ Application' main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sigma", type=int, default=5, dest="sigma",
                        help="Sigma paramter for the SVM kernel function. Default: 5")
    parser.add_argument("-a", "--agents", type=int, default=4, dest="agents",
                        help="The number of agents in the simulation. Default: 4")
    args = parser.parse_args()

    svm = SVM(x, y, args.sigma)
    # agents = [Agent() for _ in range(args.agents)]

    # for round in range(args.rounds):
    #     iteration(agents)

    return 0
