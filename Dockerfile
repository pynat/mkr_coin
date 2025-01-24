FROM continuumio/miniconda3:latest

# Create conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Set default shell to use created conda environment (corrected environment name)
SHELL ["conda", "run", "-n", "mkr", "/bin/bash", "-c"]

# Copy application code to container
COPY . /app

# Copy pickle model into container
COPY final_xgboost_model.pkl /app/final_xgboost_model.pkl

# Expose application port
EXPOSE 5001

# Define the command to run your app
CMD ["python", "predict.py"]