# @Luka
#
# Entry point — runs all three experiments in sequence and saves figures to results/figures/.
# Run: python main.py  (from project root)
#
# Experiments 2 and 3 can run as soon as network.py, metrics.py, and Milica's files are done.
# Uncomment each block below once its dependencies are implemented.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/data",    exist_ok=True)


def main():
    print("=== Figure 1: Reproduce paper MSD plot ===")
    # Requires: algorithm.py, game.py (Milica) + network.py, metrics.py (Luka)
    # from experiments.reproduce_figure1 import reproduce_figure1
    # reproduce_figure1()

    print("=== Figure 2: MSD vs topology ===")
    # Requires: same as Figure 1
    # from experiments.topology_test import topology_test
    # topology_test()

    print("=== Figure 3: MSD floor vs step size ===")
    # Requires: same as Figure 1
    # from experiments.stepsize_test import stepsize_test
    # stepsize_test()

    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
