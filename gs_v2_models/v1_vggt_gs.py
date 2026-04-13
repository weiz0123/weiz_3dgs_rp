import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicMLP, self).__init__()
        
        # 1. First linear layer (Input -> Hidden)
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # 2. Activation function (introduces non-linearity)
        self.relu = nn.ReLU()
        
        # 3. Second linear layer (Hidden -> Output)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define how the data flows through the network
        out = self.fc1(x)      # Pass through first layer
        out = self.relu(out)   # Apply activation
        out = self.fc2(out)    # Pass through output layer
        return out

# ==========================================
# Example of how to use it
# ==========================================

if __name__ == "__main__":
    # Define the dimensions
    INPUT_DIM = 10   # e.g., 10 features per sample
    HIDDEN_DIM = 32  # Number of neurons in the hidden layer
    OUTPUT_DIM = 2   # e.g., 2 classes for binary classification
    
    # Initialize the model
    model = BasicMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    # Create some dummy input data (Batch Size of 5, 10 features each)
    dummy_input = torch.randn(5, INPUT_DIM)
    
    # Pass the data through the model
    predictions = model(dummy_input)
    
    print("Input shape:", dummy_input.shape)
    print("Output shape:", predictions.shape)
    print("\nOutput values:\n", predictions)