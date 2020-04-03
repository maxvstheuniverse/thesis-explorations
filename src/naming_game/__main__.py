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

    metrics = {"logs": [], "success": [], "alignment": [], "words_invented": 0, "invention_rate": 0,
               "time": t, "games": num_games, "agents": num_agents, "objects": num_objects}

    # -- Run
    for game in games:
        speaker, listener = random.choices(agents, k=2)
        obj = random.randint(0, 7)
        success = 0
        alignment = 0

        word, invented = speaker.speak(obj)
        interpreted_obj = listener.listen(obj, word)

        if invented:
            speaker.add_word(obj, word)
            metrics['words_invented'] += 1

        if interpreted_obj == obj:
            if len(listener.vocabulary[obj]) == 1 and len(speaker.vocabulary[obj]) == 1:
                alignment = 1
            success = 1

            speaker.adopt(obj, word)
            listener.adopt(obj, word)
        else:
            listener.add_word(obj, word)

        metrics['success'].append(success)
        metrics['alignment'].append(alignment)
        metrics['invention_rate'] = metrics['words_invented'] / game

        log = [game, obj, success, alignment,
               speaker.identifier, speaker.print_vocabulary(),
               listener.identifier, listener.print_vocabulary(),
               word if invented else '', metrics["invention_rate"]]

        metrics["logs"].append(log)

    return metrics


def export(metrics):
    # -- export logs
    with open(os.path.join(os.getcwd(), "output", f"log-{metrics['time']}.csv"), "w") as file:
        header = ";".join(["Game", "Object", "Success", "Alignment Success",
                           "Speaker", "Vocabulary", "Listener", "Vocabulary",
                           "Word Invented", "Word Invention Rate"])

        file.write("\n".join([header] + [";".join(map(lambda x: str(x), log)) for log in metrics["logs"]]))
        file.close()

    # -- export metrics
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    games = range(metrics["games"])
    window = 40

    y_success = [sum(metrics["success"][j] for j in range(i - window, i) if j > 0) / window for i in games]
    y_alignment = [sum(metrics["alignment"][j] for j in range(i - window, i) if j > 0) / window for i in games]

    ax.set_title(f"Naming Game (Agents {metrics['agents']}, Games: {metrics['games']}, Objects: {metrics['objects']})")
    ax.plot(games, y_success, label="Success")
    ax.plot(games, y_alignment, label="Alignment")
    ax.set_ylabel("%")
    ax.set_xlabel('Games')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, metrics["games"])

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/plot-{metrics['time']}.svg")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=10, dest="num_agents",
                        help="The number of agents in the simulation.")
    parser.add_argument("-g", "--games", type=int, default=2000, dest="num_games",
                        help="The number of games played. Default: 100")
    parser.add_argument("-o", "--objects", type=int, default=8, dest="num_objects",
                        help="The number of objects used in the game. Default: 8")
    args = parser.parse_args()

    #  Start simulation
    start_time = time.time()
    metrics = run_simulation(args.num_agents, args.num_games, args.num_objects)
    print(f"Finished {args.num_games} games in {time.time() - start_time:0.03f} seconds!")

    export(metrics)
    return 0
