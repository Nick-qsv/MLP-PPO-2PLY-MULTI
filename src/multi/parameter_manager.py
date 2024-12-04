import multiprocessing
import torch
import numpy as np
import boto3
import io
import os
from config import (
    S3_BUCKET_NAME,
    S3_MODEL_PREFIX,
    INITIAL_TEMPERATURE,
    FINAL_TEMPERATURE,
    MAX_UPDATES,
)
from agents.policy_network import BackgammonPolicyNetwork


class ParameterManager:
    """
    Manages shared parameters and versioning for synchronization between the trainer and workers.

    This class uses shared objects (lock, parameters, version) created by a multiprocessing.Manager()
    in the main process. It ensures thread-safe updates and provides methods for workers to access
    the latest parameters and save/load the model.

    Args:
        lock (multiprocessing.Lock): A lock to ensure thread-safe updates.
        version (multiprocessing.Value): A shared integer representing the version number.
        parameters (multiprocessing.Manager.dict): A shared dictionary storing the model parameters.
    """

    # Temperature decay parameters
    INITIAL_TEMPERATURE = INITIAL_TEMPERATURE
    FINAL_TEMPERATURE = FINAL_TEMPERATURE
    MAX_UPDATES = MAX_UPDATES

    def __init__(self, lock, version, parameters):
        self.lock = lock
        self.version = version
        self.parameters = parameters

        # Initialize parameters if empty
        with self.lock:
            if not bool(self.parameters):  # Check if the parameters dictionary is empty
                policy_network = BackgammonPolicyNetwork()
                state_dict = policy_network.state_dict()
                # Convert tensors to CPU to ensure compatibility
                state_dict = {k: v.cpu() for k, v in state_dict.items()}
                self.parameters.update(state_dict)
                self.version.value = 1  # Set initial version

        self.s3_bucket_name = S3_BUCKET_NAME
        self.s3_model_prefix = S3_MODEL_PREFIX

    def get_parameters(self, device=None):
        """
        Retrieves the current state_dict of parameters.

        Args:
            device (torch.device, optional): The device to move the parameters to.

        Returns:
            dict: A state_dict with parameter names as keys and tensors as values.
        """
        # Reconstruct the state_dict from stored NumPy arrays
        state_dict = {}
        for key, array in self.parameters.items():
            state_dict[key] = torch.tensor(array, device=device)
        return state_dict

    def get_version(self):
        """
        Retrieves the current version number.

        Returns:
            int: The version number.
        """
        return self.version.value

    def set_parameters(self, new_state_dict):
        """
        Updates the shared parameters with a new state_dict and increments the version number.

        Args:
            new_state_dict (dict): A state_dict containing the new parameters.
        """
        with self.lock:
            # Update the parameters with new NumPy arrays
            for key, tensor in new_state_dict.items():
                self.parameters[key] = tensor.cpu().numpy()
            # Increment the version number
            self.version.value += 1

    def get_temperature(self):
        """
        Calculates the current temperature based on the version number.

        Returns:
            float: The current temperature.
        """
        current_version = self.get_version()
        if current_version <= 1:
            return self.INITIAL_TEMPERATURE
        elif current_version >= 1 + self.MAX_UPDATES:
            return self.FINAL_TEMPERATURE
        else:
            decay_fraction = (current_version - 1) / self.MAX_UPDATES
            temperature = (
                self.INITIAL_TEMPERATURE
                - (self.INITIAL_TEMPERATURE - self.FINAL_TEMPERATURE) * decay_fraction
            )
            return temperature

    # --- Modified methods for saving and loading ---

    def save_model_local(self, filename=None):
        """
        Save the model parameters locally.

        Args:
            filename (str): Optional filename. Defaults to 'models/ppo_backgammon.pth'.
        """
        state_dict = self.get_parameters()

        if not os.path.exists("models"):
            os.makedirs("models")
        if filename is None:
            filename = os.path.join("models", "ppo_backgammon.pth")
        else:
            filename = os.path.join("models", filename)

        torch.save(state_dict, filename)
        print(f"Model saved locally to {filename}")

    def load_model_local(self, filename=None):
        """
        Load the model parameters from a local file.

        Args:
            filename (str): Optional filename. Defaults to 'models/ppo_backgammon.pth'.
        """
        if filename is None:
            filename = os.path.join("models", "ppo_backgammon.pth")
        else:
            filename = os.path.join("models", filename)

        if os.path.isfile(filename):
            state_dict = torch.load(filename, map_location=torch.device("cpu"))
            self.set_parameters(state_dict)
            print(f"Model loaded locally from {filename}")
        else:
            print(f"No saved model found locally at {filename}")

    def save_model_s3(self, filename=None):
        """
        Save the model parameters to S3.

        Args:
            filename (str): Optional filename. Defaults to 'ppo_backgammon_s3.pth'.
        """
        # Initialize s3_client here
        s3_client = boto3.client("s3")

        state_dict = self.get_parameters()

        if filename is None:
            filename = "ppo_backgammon_s3.pth"

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)  # Reset buffer position to the beginning

        s3_key = os.path.join(self.s3_model_prefix, filename)
        print(f"Uploading model to s3://{self.s3_bucket_name}/{s3_key}...")

        try:
            s3_client.upload_fileobj(buffer, self.s3_bucket_name, s3_key)
            print(f"Model saved to s3://{self.s3_bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Failed to save model to S3: {e}")

    def load_model_s3(self, filename=None):
        """
        Load the model parameters from S3.

        Args:
            filename (str): Optional filename. Defaults to 'ppo_backgammon_s3.pth'.
        """
        # Initialize s3_client here
        s3_client = boto3.client("s3")

        if filename is None:
            filename = "ppo_backgammon_s3.pth"

        s3_key = os.path.join(self.s3_model_prefix, filename)
        buffer = io.BytesIO()

        try:
            s3_client.download_fileobj(self.s3_bucket_name, s3_key, buffer)
            buffer.seek(0)
            state_dict = torch.load(buffer, map_location=torch.device("cpu"))
            self.set_parameters(state_dict)
            print(f"Model loaded from s3://{self.s3_bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Failed to load model from S3: {e}")

    def save_model(self, filename=None, to_s3=False):
        """
        Save the model parameters either locally or to S3.

        Args:
            filename (str): Optional filename.
            to_s3 (bool): If True, save to S3; otherwise, save locally.
        """
        if to_s3:
            self.save_model_s3(filename)
        else:
            self.save_model_local(filename)

    def load_model(self, filename=None, from_s3=False):
        """
        Load the model parameters either locally or from S3.

        Args:
            filename (str): Optional filename.
            from_s3 (bool): If True, load from S3; otherwise, load locally.
        """
        if from_s3:
            self.load_model_s3(filename)
        else:
            self.load_model_local(filename)
