import multiprocessing
import torch
import numpy as np
import boto3
import io
import os
from config import S3_BUCKET_NAME, S3_MODEL_PREFIX
from agents.policy_network import BackgammonPolicyNetwork


class ParameterManager:
    """
    Manages shared parameters and versioning for synchronization between the trainer and workers.

    This class uses a multiprocessing.Manager() to store a shared state_dict of the model parameters
    and a version number. It ensures thread-safe updates and provides methods for workers to access
    the latest parameters and save/load the model.

    Args:
        policy_network_class (nn.Module): The class of the policy network whose parameters are managed.

    Attributes:
        parameters (multiprocessing.managers.DictProxy): A proxy dictionary storing the model parameters.
        version (multiprocessing.managers.ValueProxy): A proxy integer representing the version number.
        lock (multiprocessing.managers.AcquirerProxy): A lock to ensure thread-safe updates.
    """

    def __init__(self, policy_network_class=BackgammonPolicyNetwork):
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()

        # Initialize the version number to 1
        self.version = self.manager.Value("i", 1)

        # Initialize the policy network and get its state_dict
        initial_network = policy_network_class()
        state_dict = initial_network.state_dict()

        # Convert tensors to NumPy arrays for serialization
        self.parameters = self.manager.dict()
        for key, tensor in state_dict.items():
            self.parameters[key] = tensor.cpu().numpy()

        # Initialize S3 client attributes (set them according to your configuration)
        self.s3_client = boto3.client("s3")  # or pass as an argument
        self.s3_bucket_name = S3_BUCKET_NAME
        self.s3_model_prefix = S3_MODEL_PREFIX

    def get_parameters(self):
        """
        Retrieves the current state_dict of parameters.

        Returns:
            dict: A state_dict with parameter names as keys and tensors as values.
        """
        # Reconstruct the state_dict from stored NumPy arrays
        state_dict = {}
        for key, array in self.parameters.items():
            state_dict[key] = torch.tensor(array)
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

    # --- New methods for saving and loading ---

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
        if not self.s3_client:
            print("S3 client not initialized. Cannot save to S3.")
            return

        state_dict = self.get_parameters()

        if filename is None:
            filename = "ppo_backgammon_s3.pth"

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)  # Reset buffer position to the beginning

        s3_key = os.path.join(self.s3_model_prefix, filename)
        print(f"Uploading model to s3://{self.s3_bucket_name}/{s3_key}...")

        try:
            self.s3_client.upload_fileobj(buffer, self.s3_bucket_name, s3_key)
            print(f"Model saved to s3://{self.s3_bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Failed to save model to S3: {e}")

    def load_model_s3(self, filename=None):
        """
        Load the model parameters from S3.

        Args:
            filename (str): Optional filename. Defaults to 'ppo_backgammon_s3.pth'.
        """
        if not self.s3_client:
            print("S3 client not initialized. Cannot load from S3.")
            return

        if filename is None:
            filename = "ppo_backgammon_s3.pth"

        s3_key = os.path.join(self.s3_model_prefix, filename)
        buffer = io.BytesIO()

        try:
            self.s3_client.download_fileobj(self.s3_bucket_name, s3_key, buffer)
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
