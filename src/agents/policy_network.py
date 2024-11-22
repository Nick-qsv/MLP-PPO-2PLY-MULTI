import torch.nn as nn
import torch.nn.functional as F
import torch


class BackgammonPolicyNetwork(nn.Module):
    """
    A Policy Network for Backgammon.

    This network evaluates a batch of board states (each representing a potential action)
    and outputs a corresponding state value for each action, treating
    each board state independently.

    Architecture:
        - Input Layer:
            Accepts board states represented as tensors of shape (batch_size, 198),
            where each row corresponds to a potential action's resulting board state.
        - Shared Encoder:
            A fully connected layer that extracts features from each board state.
        - Value Head:
            Outputs a state value estimate per action.

    Parameters:
        input_size (int): The size of the input feature vector.
                          Default is 198, corresponding to the board state representation.
        hidden_size (int): The number of neurons in the hidden fully connected layer.
                           Default is 128.

    Attributes:
        fc1 (nn.Linear): The fully connected layer mapping inputs to hidden representations.
        value_head (nn.Linear): The layer mapping hidden representations to state value estimates.


    """

    def __init__(self, input_size=198, hidden_size=128):
        """
        Initializes the BackgammonPolicyNetwork.

        Args:
            input_size (int, optional): Size of the input feature vector. Default is 198.
            hidden_size (int, optional): Number of neurons in the hidden layer. Default is 128.
        """
        super(BackgammonPolicyNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, x):
        """
        Forward pass for a batch of action states.

        Args:
            x (torch.Tensor): Input tensor containing batch of action state features.
                              Shape: (batch_size, 198)

        Returns:
            logits (torch.Tensor): Action logits for each action state.
                                   Shape: (batch_size,)
            state_values (torch.Tensor): State value estimates for each action state.
                                         Shape: (batch_size,)
        """
        # x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc1(x))
        state_values = self.value_head(x).squeeze(-1)
        return state_values
