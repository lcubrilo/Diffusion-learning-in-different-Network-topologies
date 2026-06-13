# @Luka
#
# Entry point — runs all three experiments in sequence and saves figures to results/figures/.
# Run: python main.py            (Figure 1 = single-run moments, fast)
#      python main.py mc         (Figure 1 = Monte-Carlo moments, slow; → figure1_mc.png)
#      python main.py both       (Figure 1 = both versions, separate files)

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/data",    exist_ok=True)

# Figure 1 moment panels: "regular" (single noisy run, fast, → figure1.png),
# "mc" (Monte-Carlo averaged, slow, → figure1_mc.png), or "both".
# Overridable from the command line: e.g. `python main.py mc`.
FIGURE1_MODE = "regular"


def main(figure1_mode=FIGURE1_MODE):
    print("=== Topology visualization ===")
    from experiments.visualize_topology import visualize_topology
    visualize_topology()

    print(f"=== Figure 1: Reproduce paper MSD plot (mode={figure1_mode}) ===")
    # Requires: algorithm.py, game.py (Milica) + network.py, metrics.py (Luka)
    from experiments.reproduce_figure1 import run_figure1
    run_figure1(figure1_mode)

    print("=== Figure 2: MSD vs topology ===")
    # Requires: same as Figure 1
    from experiments.topology_test import topology_test
    topology_test()

    print("=== Figure 3: MSD floor vs step size ===")
    #Requires: same as Figure 1
    from experiments.stepsize_test import stepsize_test #commented out because it just works, and exported figure is good
    stepsize_test()

    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else FIGURE1_MODE
    main(figure1_mode=mode)
