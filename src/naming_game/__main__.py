import argparse
import random
import os
import matplotlib.pyplot as plt
import time

from naming_game.Agent import Agent


def run_simulation(num_agents, num_games, num_objects):
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    # -- Setup
    agents = [Agent(i, num_objects) for i in range(num_agents)]
    games = range(1, num_games + 1)

    successes = []
    alignments = []

    words_invented = 0
    invention_rate = 0

    logs = []

    # -- Run
    for game in games:
        speaker, listener = random.choices(agents, k=2)
        obj = random.randint(0, 7)
        success = 0
        alignment = 0

        speaker_vocab = speaker.get_vocabulary()
        listener_vocab = listener.get_vocabulary()

        word, invented = speaker.speak(obj)
        interpreted_obj = listener.listen(obj, word)

        if invented:
            speaker.add_word(obj, word)
            words_invented += 1

        if interpreted_obj == obj:
            if len(listener.vocabulary[obj]) == 1 and len(speaker.vocabulary[obj]) == 1:
                alignment = 1
            success = 1

            speaker.adopt(obj, word)
            listener.adopt(obj, word)
        else:
            listener.add_word(obj, word)

        invention_rate = words_invented / game

        successes.append(success)
        alignments.append(alignment)

        metrics = [game, obj, success, alignment,
                   speaker.identifier, speaker.print_vocabulary(),
                   listener.identifier, listener.print_vocabulary(),
                   word if invented else '', invention_rate]

        logs.append(";".join(map(lambda x: str(x), metrics)))

    # -- Output

    # -- export log
    with open(os.path.join(os.getcwd(), "output", f"log-{t}.csv"), "w") as file:
        header = ";".join(["Game", "Object", "Success", "Alignment Success",
                           "Speaker", "Vocabulary", "Listener", "Vocabulary",
                           "Word Invented", "Word Invention Rate"])
        file.write("\n".join([header] + logs))
        file.close()

    # -- export plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    window = 40

    y_success = [sum(successes[j] for j in range(i - window, i) if j > 0) / window for i in range(num_games)]
    y_alignment = [sum(alignments[j] for j in range(i - window, i) if j > 0) / window for i in range(num_games)]

    ax.set_title(f"Naming Game (Agents {len(agents)}, Games: {num_games}, Objects: {num_objects})")
    ax.plot(games, y_success, label="Success")
    ax.plot(games, y_alignment, label="Alignment")
    ax.set_ylabel("%")
    ax.set_xlabel('Games')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, num_games)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/plot-{t}.svg")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=10, dest="num_agents",
                        help="The number of agents in the simulation.")
    parser.add_argument("-g", "--games", type=int, default=100, dest="num_games",
                        help="The number of games played. Default: 100")
    parser.add_argument("-o", "--objects", type=int, default=8, dest="num_objects",
                        help="The number of objects used in the game. Default: 8")
    args = parser.parse_args()

    #  Start simulation
    start_time = time.time()
    run_simulation(args.num_agents, args.num_games, args.num_objects)
    print(f"Finished {args.num_games} games in {time.time() - start_time:0.03f} seconds!")
    return 0
