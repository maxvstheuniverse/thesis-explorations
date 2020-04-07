import argparse
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from ae_naming_game.Agent import Agent


tf.keras.backend.set_floatx('float64')


def play(game_id, agents):
    """ Play a game, provide a game id and all agents. """
    speaker, listener = random.choices(agents, k=2)
    obj = random.randint(0, 7)
    success, alignment = 0, 0

    word, invented = speaker.speak(obj)
    interpreted_obj = listener.listen(obj, word)

    # speaker_vocab, listener_vocab = speaker.print_vocabulary(), listener.print_vocabulary()

    if interpreted_obj == obj:
        if len(listener.vocabulary[obj]) == 1 and len(speaker.vocabulary[obj]) == 1:
            alignment = 1
        success = 1

        speaker.adopt(obj, word)
        listener.adopt(obj, word)
    else:
        listener.add_word(obj, word)

    return [game_id, obj, success, alignment,
            speaker.identifier,  # speaker_vocab,
            listener.identifier]  # listener_vocab,
            #word if invented else '']


def run(args):
    """ Run the simulation. """
    agents = [Agent(i, args.objects) for i in range(args.agents)]
    games = range(1, args.games + 1)

    return [play(game_id, agents) for game_id in games]


def export(results, t, args):
    """ Export the results. """
    t = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime(t))

    # -- export logs
    with open(os.path.join(os.getcwd(), "output", f"log-{t}.csv"), "w") as file:
        header = ";".join(["Game", "Object", "Success", "Alignment Success",
                           "Speaker",  # "Vocabulary",
                           "Listener"])  # "Vocabulary",
                           # "Word Invented"])

        file.write("\n".join([header] + [";".join(map(lambda x: str(x), row)) for row in results]))
        file.close()

    # -- export plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    games = range(args.games)
    window = 40

    results_t = list(zip(*results))  # -- transpose results list
    success, alignment = results_t[2], results_t[3]

    y_success = [sum(success[j] for j in range(i - window, i) if j > 0) / window for i in games]
    y_alignment = [sum(alignment[j] for j in range(i - window, i) if j > 0) / window for i in games]

    ax.set_title(f"Naming Game (Agents {args.agents}, Games: {args.games}, Objects: {args.objects})")
    ax.plot(games, y_success, label="Success")
    ax.plot(games, y_alignment, label="Alignment")
    ax.set_ylabel("%")
    ax.set_xlabel('Games')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, args.games)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/plot-{t}.svg")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=10, dest="agents",
                        help="The number of agents in the simulation. Default: 10")
    parser.add_argument("-g", "--games", type=int, default=2000, dest="games",
                        help="The number of games played. Default: 2000")
    parser.add_argument("-o", "--objects", type=int, default=8, dest="objects",
                        help="The number of objects used in the game. Default: 8")
    args = parser.parse_args()

    #  Start simulation
    start_time = time.time()
    results = run(args)
    print(f"Finished {args.games} games in {time.time() - start_time:0.03f} seconds!")

    export(results, start_time, args)
    return 0