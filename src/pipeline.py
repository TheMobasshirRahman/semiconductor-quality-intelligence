"""
Rebuild every artifact from the raw SECOM files, in dependency order.

    python -m src.pipeline            # full run (~10 min; feature selection + model dominate)
    python -m src.pipeline --from spc # resume from a given step

Each step runs as its own `python -m src.<step>` process so saved models pickle their
classes under the package name (loadable by the dashboard), exactly as when run by hand.
"""
import argparse
import subprocess
import sys
import time

STEPS = ["data_loader", "preprocessing", "spc", "feature_selection", "model", "anomaly"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="first step to run")
    args = parser.parse_args(argv)

    for step in STEPS[STEPS.index(args.start):]:
        print(f"\n=== {step} " + "=" * (60 - len(step)), flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-W", "ignore", "-m", f"src.{step}"], check=True)
        print(f"--- {step} done in {time.time() - t0:.0f}s", flush=True)
    print("\nAll artifacts rebuilt. Start the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
