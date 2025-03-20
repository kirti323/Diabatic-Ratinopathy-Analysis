import torch

# Assuming `model` is your trained model and `test_loader` is your DataLoader for test data
def test_model(model, test_loader):
    # Ensure the model is in evaluation mode
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize counters
    correct_predictions = 0
    total_predictions = 0
    
    # No need to track gradients for validation, hence wrap in torch.no_grad to save memory and computations
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Permute the images to be of shape [batch_size, channels, height, width]
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            
            # Total number of labels
            total_predictions += labels.size(0)
            
            # Total correct predictions
            correct_predictions += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy