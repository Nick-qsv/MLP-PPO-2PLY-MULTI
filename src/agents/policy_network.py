import torch.nn as nn
import torch.nn.functional as F
import torch


class BackgammonPolicyNetwork(nn.Module):
    """
    A Policy Network for Backgammon with Action Masking and Batch-wise Log Probabilities.

    This network evaluates a batch of board states (each representing a potential action)
    and outputs log probabilities for each action, considering an action mask to exclude
    invalid actions. It also outputs a corresponding state value for each action, treating
    each board state independently.

    Architecture:
        - Input Layer:
            Accepts board states represented as tensors of shape (batch_size, 198),
            where each row corresponds to a potential action's resulting board state.
        - Shared Encoder:
            A fully connected layer that extracts features from each board state.
        - Action Head:
            Outputs a single logit per action.
        - Value Head:
            Outputs a state value estimate per action.

    Parameters:
        input_size (int): The size of the input feature vector.
                          Default is 198, corresponding to the board state representation.
        hidden_size (int): The number of neurons in the hidden fully connected layer.
                           Default is 128.
        use_sigmoid (bool): If True, use sigmoid activation; if False, use ReLU. Default is False (ReLU).

    Attributes:
        fc1 (nn.Linear): The fully connected layer mapping inputs to hidden representations.
        action_head (nn.Linear): The layer mapping hidden representations to a single logit.
        value_head (nn.Linear): The layer mapping hidden representations to state value estimates.

    Example:
        >>> model = BackgammonPolicyNetwork(input_size=198, hidden_size=128)
        >>> board_features = torch.randn(32, 198)  # Batch of 32 board states (actions)
        >>> action_mask = torch.randint(0, 2, (32,))  # Random action mask
        >>> log_probs, state_values = model(board_features, action_mask)
    """

    def __init__(self, input_size=198, hidden_size=128, use_sigmoid=False):
        """
        Initializes the BackgammonPolicyNetwork.

        Args:
            input_size (int, optional): Size of the input feature vector. Default is 198.
            hidden_size (int, optional): Number of neurons in the hidden layer. Default is 128.
            use_sigmoid (bool): If True, use sigmoid activation; if False, use ReLU. Default is False (ReLU).
        """
        super(BackgammonPolicyNetwork, self).__init__()
        self.use_sigmoid = use_sigmoid
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, x, action_mask):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor containing batch of board state features.
                              Shape: (batch_size, 198)
            action_mask (torch.Tensor): Tensor containing action masks.
                                        Shape: (batch_size,), elements are 1 (valid action) or 0 (invalid action)

        Returns:
            log_probs (torch.Tensor): Log probabilities over the batch of actions.
                                      Invalid actions have log probabilities set to a large negative value.
                                      Shape: (batch_size,)
            state_values (torch.Tensor): Tensor containing state value estimates for each action.
                                         Shape: (batch_size,)
        """
        if self.use_sigmoid:
            x = torch.sigmoid(self.fc1(x))  # Sigmoid activation for hidden layer
        else:
            x = torch.relu(self.fc1(x))  # ReLU activation for hidden layer

        logits = self.action_head(x).squeeze(-1)  # Action logits
        state_values = self.value_head(x).squeeze(-1)  # State value estimates

        # Adjust logits for invalid actions
        mask = action_mask.to(dtype=torch.bool, device=logits.device)
        masked_logits = logits.masked_fill(~mask, -1e9)

        # Compute log probabilities across the batch dimension
        log_probs = F.log_softmax(masked_logits, dim=0)

        return log_probs, state_values
