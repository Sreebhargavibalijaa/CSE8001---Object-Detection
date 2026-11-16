"""CLI: open-loop load against the deployed k3s pipeline (same as profiling.k8s_profile).

  python -m loadgen calibrate | stages | e2e | preflight
"""

from profiling.k8s_profile import main

if __name__ == "__main__":
    main()
