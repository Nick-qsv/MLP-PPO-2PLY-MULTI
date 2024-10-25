import multiprocessing
import torch
import numpy as np


class ParameterManager:
    """
    Manages shared parameters and versioning for synchronization between the trainer and workers.

    This class uses a multiprocessing.Manager() to store a shared state_dict of the model parameters
    and a version number. It ensures thread-safe updates and provides methods for workers to access
    the latest parameters.

    Args:
        policy_network_class (nn.Module): The class of the policy network whose parameters are managed.

    Attributes:
        parameters (multiprocessing.managers.DictProxy): A proxy dictionary storing the model parameters.
        version (multiprocessing.managers.ValueProxy): A proxy integer representing the version number.
        lock (multiprocessing.managers.AcquirerProxy): A lock to ensure thread-safe updates.
    """

    def __init__(self, policy_network_class):
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
