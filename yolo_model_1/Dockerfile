# 1. Base Image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install OS-level dependencies for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file
COPY requirements.txt .

# 5. Install CPU-Only PyTorch AND TorchVision (THE REAL FIX)
# This installs all the big libraries from the CPU-only source.
# We add "torchvision" and "torchaudio" here.
RUN pip install --no-cache-dir --timeout=600 \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 6. Install the rest of the app requirements
# This will now see that torch AND torchvision are already
# installed (the CPU versions) and will NOT download them again.
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# 7. Set up a non-root user
RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser/app

# 8. Copy the rest of the application code (app and models)
COPY --chown=appuser:appuser . .

# 9. Expose the port
EXPOSE 8000

# 10. Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]