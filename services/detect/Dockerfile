# Build from repository root after the shared image exists:
#   docker build -f services/Dockerfile -t mot17-serve:local .
#   docker build -f services/detect/Dockerfile -t mot17-detect:local .
FROM mot17-serve:local
CMD ["python", "-m", "services.detect.server"]
