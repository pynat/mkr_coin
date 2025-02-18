FROM continuumio/miniconda3:latest

# Create conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Set default shell to use created conda environment (corrected environment name)
SHELL ["conda", "run", "-n", "mkr", "/bin/bash", "-c"]

# Copy pickle model into container
COPY final_xgboost_model.pkl /app/final_xgboost_model.pkl

# Copy application code to container
COPY . /app

# Set working directory
WORKDIR /app

# List the files in /app for debugging purposes
RUN ls -l /app

# Expose application port
EXPOSE 5001

# Define the command to run your app
CMD ["conda", "run", "-n", "mkr", "python", "/app/predict.py"]