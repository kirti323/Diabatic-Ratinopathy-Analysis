import torch
import time
import collections
import resource
import GPUtil
from GPUtil import showUtilization as gpu_usage
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def print_gpu_memory_usage(when_use):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name} @ {when_use}")
        print(f"Total VRAM: {gpu.memoryTotal / 1024:.2f} GB")
        print(f"Used VRAM: {gpu.memoryUsed / 1024:.2f} GB")
        print(f"Free VRAM: {gpu.memoryFree / 1024:.2f} GB")
        print(f"GPU Load: {gpu.load * 100:.2f}%\n")

def current_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10, criterion_name='CrossEntropyLoss', segmentation_name = "intraretinal"):
    # Initialize a dictionary to store training and validation statistics
    stats = defaultdict(list)
    
    # Create Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Intialize the Tensorboard Writer
    #writer = SummaryWriter()
    
    # Log the model graph
    #images, labels = next(iter(train_loader))
    #writer.add_graph(model, images)
    
    # Start time
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Epoch start time
        epoch_start_time = time.time()
        
        for inputs, labels in train_loader:
            # Move the inputs and labels to the GPU
            inputs, labels = (inputs.cuda()).float(), labels.cuda()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss using the original outputs
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Convert outputs to 0 or 1 for evaluation or logging
            binary_outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
            
            # Accumulate loss
            running_loss += loss.item()
        
        # Record the training loss for this epoch
        average_loss = running_loss / len(train_loader)
        stats['train_loss'].append(average_loss)
        
        # log training in Tensorboard
        #writer.add_scalar('Loss/train', average_loss, epoch)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = (inputs.cuda()).float(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        
        # Record the validation loss for this epoch
        average_valid_loss = valid_loss / len(valid_loader)
        stats['valid_loss'].append(average_valid_loss)
        
        # log validation in Tensorboard
        #writer.add_scalar('Loss/valid', average_valid_loss, epoch)
        
        # Get total for Epoch
        epoch_time = time.time() - epoch_start_time
        
        # Create a print statement for this particular run.
        print(f'Epoch: {epoch + 1}/{epochs} | Training Loss: {average_loss:.4f} | Validation Loss: {average_valid_loss:.4f} | Time: {epoch_time:.2f} seconds')
    
    return stats