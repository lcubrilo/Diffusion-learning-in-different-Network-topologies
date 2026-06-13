# @Luka
#
# Entry point — runs all three experiments in sequence and saves figures to results/figures/.
# Run: python main.py  (from project root)

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/data",    exist_ok=True)


def main():
    print("=== Topology visualization ===")
    from experiments.visualize_topology import visualize_topology
    visualize_topology()

    print("=== Figure 1: Reproduce paper MSD plot ===")
    # Requires: algorithm.py, game.py (Milica) + network.py, metrics.py (Luka)
    from experiments.reproduce_figure1 import reproduce_figure1
    reproduce_figure1()

    print("=== Figure 2: MSD vs topology ===")
    # Requires: same as Figure 1
    from experiments.topology_test import topology_test
    topology_test()

    print("=== Figure 3: MSD floor vs step size ===")
    # Requires: same as Figure 1
    #from experiments.stepsize_test import stepsize_test #commented out because it just works, and exported figure is good
    #stepsize_test()

    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
