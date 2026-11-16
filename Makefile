.PHONY: profile profile-k8s slices report

PYTHON ?= mot_pipeline (Object_Detection + Tracking)/venv/bin/python3

profile:
	chmod +x run_profile.sh
	./run_profile.sh

profile-k8s:
	chmod +x run_profile_k8s.sh
	./run_profile_k8s.sh

slices:
	"$(PYTHON)" "mot_pipeline (Object_Detection + Tracking)/build_slices.py"

report:
	PYTHONPATH=. "$(PYTHON)" scripts/write_report.py
