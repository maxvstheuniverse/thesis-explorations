import argparse


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=10, dest="agents",
                        help="The number of agents in the simulation. Default: 10")
    parser.add_argument("-g", "--games", type=int, default=2000, dest="games",
                        help="The number of games played. Default: 2000")
    parser.add_argument("-o", "--objects", type=int, default=8, dest="objects",
                        help="The number of objects used in the game. Default: 8")
    args = parser.parse_args()

    return 0
