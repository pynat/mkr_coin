FROM continuumio/miniconda3
#FROM mambaorg/micromamba:latest


# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
RUN wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm ta-lib-0.4.0-src.tar.gz

# Create conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Prepare pip requirements
COPY requirements.txt .
RUN conda run -n mkr pip install -r requirements.txt

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