.PHONY: profile slices report

PYTHON ?= mot_pipeline (Object_Detection + Tracking)/venv/bin/python3

profile:
	chmod +x run_profile.sh
	./run_profile.sh

slices:
	"$(PYTHON)" "mot_pipeline (Object_Detection + Tracking)/build_slices.py"

report:
	PYTHONPATH=. "$(PYTHON)" scripts/write_report.py
