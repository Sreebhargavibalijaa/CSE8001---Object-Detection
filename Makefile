.PHONY: profile profile-k8s slices

PYTHON ?= mot_pipeline (Object_Detection + Tracking)/venv/bin/python3

# Milestone 2: every T5-T7 table and plot from the k3s deployment (see README).
profile:
	chmod +x run_profile_k8s.sh
	./run_profile_k8s.sh

profile-k8s: profile

slices:
	"$(PYTHON)" "mot_pipeline (Object_Detection + Tracking)/build_slices.py"
