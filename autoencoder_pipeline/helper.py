import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, embedding_size):
        super(Autoencoder, self).__init__()
        
        # Define the first layer of the encoder layers, starting with the input size and ending with the embedding size
        encoder_layers = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()] 
        
        # Define the first layer of the decoder layers, starting with the embedding size and ending with the input size
        decoder_layers = [nn.Linear(embedding_size, layer_sizes[-1]), nn.ReLU()]
        
        # Define layers for strength prediction
        strength_layers = [nn.Linear(embedding_size, 20), nn.ReLU(), nn.Linear(20, 1)]
        
        # Add intermediate layers for both encoder and decoder
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(layer_sizes[-i - 1], layer_sizes[-i - 2]))
            decoder_layers.append(nn.ReLU())

        # Add the final linear layer for the encoder to output the embedding
        encoder_layers.append(nn.Linear(layer_sizes[-1], embedding_size, bias=False))
        
        # Add the final linear layer for the decoder to output the reconstructed input
        decoder_layers.append(nn.Linear(layer_sizes[0], input_size))

        #after this step the full encoder and decoder are defined
        
        # Define the encoder, decoder, and strength prediction as sequential modules
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.strength = nn.Sequential(*strength_layers)
        
    def get_strength(self, x):
        x = self.encoder(x)
        return self.strength(x)
    
    # Forward pass of the autoencoder, returns the reconstructed input and strength prediction
    def forward(self, x):
        x = self.encoder(x)
        strength = self.strength(x)
        x = self.decoder(x)
        return x, strength
    
    def encode(self, x):
        return self.encoder(x)

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, embedding_size):
        super(VariationalAutoencoder, self).__init__()
        
        # Define the encoder layers, starting with the input size and ending with the embedding size
        encoder_layers = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]
        
        # Define the decoder layers, starting with the embedding size and ending with the input size
        decoder_layers = [nn.Linear(embedding_size, layer_sizes[-1]), nn.ReLU()]
        
        # Define layers for strength prediction
        strength_layers = [nn.Linear(embedding_size, 20), nn.ReLU(), nn.Linear(20, 1)]
        
        # Add intermediate layers for both encoder and decoder
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(layer_sizes[-i - 1], layer_sizes[-i - 2]))
            decoder_layers.append(nn.ReLU())
        
        # Add the final linear layer for the decoder to output the reconstructed input
        decoder_layers.append(nn.Linear(layer_sizes[0], input_size))
        
        # Define the encoder, decoder, and strength prediction as sequential modules
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.strength = nn.Sequential(*strength_layers)
        
        # Define linear layers for mean and log variance for the variational autoencoder
        self.mu = nn.Linear(layer_sizes[-1], embedding_size)
        self.logvar = nn.Linear(layer_sizes[-1], embedding_size)
    
    # Function to get the strength prediction for a given input
    def get_strength(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        return self.strength(mu)
    
    # Forward pass of the variational autoencoder, returns the reconstructed input and strength prediction
    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        strength = self.strength(mu)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        x = self.decoder(x)
        return x, strength
    
    # Function to encode the input data and return the mean of the latent space
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        return mu

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(model, train_loader, test_loader, reconstruction_weight, strength_weight, strength_index, device, num_epochs=10, train_strength=True):
    """
    Function to train the autoencoder model.

    Args:
        model (nn.Module): The autoencoder model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        reconstruction_weight (float): Weight for the reconstruction loss.
        strength_weight (float): Weight for the strength prediction loss.
        strength_index (int): Index of the column containing strength values in the classification data.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        num_epochs (int, optional): Number of epochs for training. Defaults to 10.
        train_strength (bool, optional): Whether to train strength prediction. Defaults to True.
    """
    # Define loss criteria
    mse_criterion = nn.MSELoss()
    strength_criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        # Initialize total loss for this epoch
        total_loss = 0
        model.train()  # Set the model to train mode

        # Iterate through batches in the training loader
        for data in train_loader:
            readings, classifications = data[0].to(device), data[1]
            # Extract strength labels if needed
            if train_strength:
                strengths = classifications[:, strength_index].unsqueeze(1).to(device)
            
            # Forward pass!
            reconstruction, pred_strength = model(readings)
            reconstruction_loss = mse_criterion(reconstruction, readings)
            
            # If training strength prediction, compute the loss
            if train_strength:
                # Identify instances where strength labels are missing (-1)
                non_labelled_strength_indexes = np.where(classifications[:, strength_index] == -1)[0]

                # Apply strength labels to the predictions for consistent loss computation
                pred_strength[non_labelled_strength_indexes] = strengths[non_labelled_strength_indexes]

                # Compute strength prediction loss
                strength_loss = strength_criterion(pred_strength, strengths)
                
                # Combine reconstruction loss and strength loss based on provided weights
                loss = reconstruction_weight * reconstruction_loss + strength_weight * strength_loss
            else:
                # If not training strength prediction, use only the reconstruction loss
                loss = reconstruction_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate the total loss for this epoch
            total_loss += loss.item()
        
        # Compute average training loss for this epoch
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation!
        avg_val_loss = 0
        avg_reconstruction_loss = 0
        avg_strength_loss = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in test_loader:
                readings, classifications = data[0].to(device), data[1]
                if train_strength:
                    strengths = classifications[:, strength_index].unsqueeze(1).to(device)
                
                # Forward pass during evaluation
                reconstruction, pred_strength = model(readings)
                reconstruction_loss = mse_criterion(reconstruction, readings)
                
                # If training strength prediction, compute and accumulate strength loss
                if train_strength:
                    non_labelled_strength_indexes = np.where(classifications[:, strength_index] == -1)[0]
                    pred_strength[non_labelled_strength_indexes] = strengths[non_labelled_strength_indexes]
                    strength_loss = strength_criterion(pred_strength, strengths)
                    avg_strength_loss += strength_loss.item()
                
                # Accumulate reconstruction loss
                avg_reconstruction_loss += reconstruction_loss.item()
                
                # Calculate total validation loss
                if train_strength:
                    avg_val_loss += reconstruction_weight * reconstruction_loss.item() + strength_weight * strength_loss.item()
                else:
                    avg_val_loss += reconstruction_loss.item()
        
        # Compute average validation losses
        avg_val_loss /= len(test_loader)
        avg_reconstruction_loss /= len(test_loader)
        avg_strength_loss /= len(test_loader)

        # Log epoch information
        print('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}, Validation Reconstruction Loss: {:.6f}, Validation Strength Loss: {:.6f}'.format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss, avg_reconstruction_loss, avg_strength_loss))

def avg(group, group_size): 
    group['GroupNumber'] = np.array(range(len(group.index))) // group_size
    res = group.groupby('GroupNumber').mean()
    return res

def load_data(data,model_inputs,ignore_indexes=['Event'],group_size=10,transform='scale'):
    index_names = [x for x in data.index.names if x not in ignore_indexes]
    data = data.groupby(index_names).apply(avg, group_size)

    data_np = data.to_numpy()
    #convert the index to a numpy array
    data_labels = data.index.to_numpy()
    data_index_names = np.array(data.index.names)
    data_columns = np.array(data.columns)
    if transform == 'scale':
        data_np = (data_np-512)/512
    elif transform == 'normalize':
        data_np = (data_np - data_np.mean(axis=0))/data_np.std(axis=0)

    #find in the index which column is 'GroupNumber' and remove it from the index and the data
    group_number_index = list(data_index_names).index('GroupNumber')
    data_index_names = np.delete(data_index_names,group_number_index)
    #remove that column from the index
    data_labels = np.array([list(x) for x in data_labels])
    data_labels = np.delete(data_labels,group_number_index,1)

    #the expected data format follows the order of the model_inputs, transform the data to match the order and put a None if the data doesn't contain that column
    data_model_input = np.array([data_np[:,list(data_columns).index(x)] if x in data_columns else np.zeros(data_np.shape[0]) for x in model_inputs]).T


    #get unique values of each column in the index
    unique_values = [list(np.unique(data_labels[:,i])) for i in range(data_labels.shape[1])]
    index_order = [unique_values[i] for i in range(len(unique_values))]
    data_labels_indexed = np.array([np.array([index_order[i].index(x) for i,x in enumerate(row)]) for row in data_labels])

    #find missing columns
    missing_columns = [x for x in model_inputs if x not in data_columns]
    
    #convert the data to a tensor
    data_tensor = torch.tensor(data_model_input,dtype=torch.float32)
    data_labels_tensor = torch.tensor(data_labels_indexed,dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(data_tensor,data_labels_tensor)
    return dataset, index_order, data_index_names, data_columns, data_labels, missing_columns

def get_embedding(model, test_loader, device):
    """
    Function to generate embeddings and associated metadata from the provided test dataset using the given autoencoder model.

    Args:
        model (nn.Module): The autoencoder model to generate embeddings.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        embeddings (numpy.ndarray): Array of embeddings generated by the model.
        labels (numpy.ndarray): Array of labels associated with each sample.
        markers (numpy.ndarray): Array of markers associated with each sample.
        strengths (numpy.ndarray): Array of strength predictions associated with each sample.
    """
    embeddings = []
    markers = []
    labels = []
    strengths = []
    
    # Switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # Encode input data and predict strength
            reconstruction = model.encode(data[0].to(device))
            strength = model.get_strength(data[0].to(device))
            
            # Append results to lists
            embeddings.append(reconstruction.cpu().numpy())
            labels.append(data[1].numpy())
            markers.append(data[0].numpy())
            strengths.append(strength.cpu().numpy())
    
    # Stack lists into numpy arrays
    strengths = np.vstack(strengths)
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)
    markers = np.vstack(markers)
    
    return embeddings, labels, markers, strengths


def format_data_into_experiment(embeddings, data_labels, data_index_names, strengths, ignore_columns=[]):
    """
    Function to format data into experiment-friendly format.

    Args:
        embeddings (numpy.ndarray): Array of embeddings.
        data_labels (numpy.ndarray): Array of labels associated with each sample.
        data_index_names (numpy.ndarray): Names of the data indexes.
        strengths (numpy.ndarray): Array of strength predictions associated with each sample.
        ignore_columns (list, optional): List of columns to ignore during formatting. Defaults to [].

    Returns:
        output (list): List of dictionaries containing experiment data.
        updated_data_index_names (numpy.ndarray): Names of the data indexes after removing 'Time'.

    Example:
    {
        'label': array(['OT-1', 'A2', '10pM'], dtype='<U32'),
        'times': array([ 4., 12., 24., 30., 36., 48., 60., 72.]), -> 8 time points
        'embeddings': 
            array([[ 2.812331  , -1.6377858 ], -> 8 coordinates representing each time point
            [ 2.4986808 , -2.1473103 ],
            [ 0.82419896, -1.4354441 ],
            [ 0.41825223, -1.2032889 ],
            [-0.25648448, -0.93741745],
            [-0.44599652, -0.49643898],
            [-0.6142367 ,  2.623618  ],
            [-1.1161239 ,  3.1469705 ]], dtype=float32),
        'embeddings_std': array([[0.47875398, 0.49482682],  -> 8 coordinates representing each time point
            [0.3167876 , 0.28064564],
            [0.31661785, 0.22188072],
            [0.27403387, 0.19103782],
            [0.36644462, 0.3634829 ],
            [0.26989996, 0.19326805],
            [0.18511224, 0.7760166 ],
            [0.2788795 , 0.8915927 ]], dtype=float32),
        'strengths': array([5.2081647, 4.4613147, 3.389994 , 3.0422797, 2.353233 , 2.3307343, 2.4898236, 1.973127 ], dtype=float32),  -> 8 strengths representing each time point
        'strengths_std': array([0.23247074, 0.5587103 , 0.47756493, 0.3757039 , 0.2846382 ,
            0.38151413, 0.42299035, 0.4429074 ], dtype=float32)},
    """
    # Trim data_labels and strengths if longer than embeddings
    if len(embeddings) < len(data_labels):
        data_labels = data_labels[:len(embeddings)]
        strengths = strengths[:len(embeddings)]
    
    # Remove ignored columns
    for ignore_column in ignore_columns:
        ignore_index = list(data_index_names).index(ignore_column)
        data_labels = np.delete(data_labels, ignore_index, axis=1)
        data_index_names = np.delete(data_index_names, ignore_index)

    # Find the index of 'Time' in data_index_names
    time_index = list(data_index_names).index('Time')
    
    # Get unique labels and associated strengths
    unique_labels = np.unique(data_labels, axis=0)
    unique_labels_strengths = []
    unique_labels_embeddings = []
    unique_labels_strengths_std = []
    unique_labels_embeddings_std = []
    
    # Iterate over unique labels
    for unique_label in unique_labels:
        indexes = np.where((data_labels == unique_label).all(axis=1))
        embeddings_for_label = embeddings[indexes]
        strengths_for_label = strengths[indexes]
        average_strength = np.mean(strengths_for_label)
        average_strength_std = np.std(strengths_for_label)
        average_embedding = np.mean(embeddings_for_label, axis=0)
        average_embedding_std = np.std(embeddings_for_label, axis=0)
        unique_labels_strengths.append(average_strength)
        unique_labels_embeddings.append(average_embedding)
        unique_labels_strengths_std.append(average_strength_std)
        unique_labels_embeddings_std.append(average_embedding_std)

    unique_labels_embeddings = np.array(unique_labels_embeddings)
    unique_labels_strengths = np.array(unique_labels_strengths)
    unique_labels_embeddings_std = np.array(unique_labels_embeddings_std)
    unique_labels_strengths_std = np.array(unique_labels_strengths_std)

    # Get unique experiment labels
    unique_labels_times = unique_labels[:, time_index]
    experiment_labels = np.delete(unique_labels, time_index, axis=1)
    unique_experiment_labels = np.unique(experiment_labels, axis=0)

    output = []
    for unique_experiment_label in unique_experiment_labels:
        indexes = np.where((experiment_labels == unique_experiment_label).all(axis=1))
        embeddings_for_label = unique_labels_embeddings[indexes]
        strengths_for_label = unique_labels_strengths[indexes]
        embeddings_for_label_std = unique_labels_embeddings_std[indexes]
        strengths_for_label_std = unique_labels_strengths_std[indexes]
        times = unique_labels_times[indexes]
        
        # Convert times to doubles from strings
        times = times.astype(float)
        
        # Sort the embeddings_for_label and strengths_for_label by time
        sorted_indexes = np.argsort(times)
        times = times[sorted_indexes]
        embeddings_for_label = embeddings_for_label[sorted_indexes]
        strengths_for_label = strengths_for_label[sorted_indexes]
        
        # Append formatted data to output
        output.append({
            'label': unique_experiment_label,
            'times': times,
            'embeddings': embeddings_for_label,
            'embeddings_std': embeddings_for_label_std,
            'strengths': strengths_for_label,
            'strengths_std': strengths_for_label_std
        })
    
    # Remove 'Time' from data_index_names
    updated_data_index_names = np.delete(data_index_names, time_index)
    
    return output, updated_data_index_names

def save_params(model_inputs,group_size,batch_size,embedding_size,autoencoder_hidden_sizes,model_path):
    with open("params.txt", "w") as file:
       file.write(f"model_inputs: {model_inputs}\n")
       file.write(f"group_size: {group_size}\n")
       file.write(f"batch_size: {batch_size}\n")
       file.write(f"embedding_size: {embedding_size}\n")
       file.write(f"autoencoder_hidden_sizes: {autoencoder_hidden_sizes}\n")
       file.write(f"model_path: {model_path}\n")
    
def load_params():
    with open("params.txt", "r") as file:
        lines = file.readlines()
        model_inputs = eval(lines[0].split(": ")[1])
        group_size = int(lines[1].split(": ")[1])
        batch_size = int(lines[2].split(": ")[1])
        embedding_size = int(lines[3].split(": ")[1])
        autoencoder_hidden_sizes = eval(lines[4].split(": ")[1])
        model_path = lines[5].split(": ")[1].strip()
    return model_inputs,group_size,batch_size,embedding_size,autoencoder_hidden_sizes,model_path