# Use a micromamba base image
FROM mambaorg/micromamba:1.5.1 as builder

# Set the working directory
WORKDIR /app

# Copy environment and configuration files
COPY environment.yml .
COPY config/config.yml ./config/config.yml

# Create the micromamba environment
RUN micromamba create -f environment.yml -y && \
    micromamba clean --all --yes

# Copy the rest of the application files
COPY . .

# Set the entrypoint for the container
# This will be a script that handles data syncing and running the main application
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Activate the environment and set the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/bin/_activate_current_env.sh && /app/entrypoint.sh"]

# Set a default command (can be overridden)
CMD ["--mode=train"]
