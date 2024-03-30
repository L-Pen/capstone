import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self,input_size,layer_sizes,embedding_size):
        super(Autoencoder,self).__init__()
        encoder_layers = [nn.Linear(input_size,layer_sizes[0]),nn.ReLU()]
        decoder_layers = [nn.Linear(embedding_size,layer_sizes[-1]),nn.ReLU()]
        # classify_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,8)]
        # regression_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,2)]
        strength_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,1)]


        for i in range(len(layer_sizes)-1):
            encoder_layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(layer_sizes[-i-1],layer_sizes[-i-2]))
            decoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Linear(layer_sizes[-1],embedding_size,bias=False))
        #add instance norm to the encoder
        decoder_layers.append(nn.Linear(layer_sizes[0],input_size))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        # self.classify = nn.Sequential(*classify_layers)
        # self.regress = nn.Sequential(*regression_layers)
        self.strength = nn.Sequential(*strength_layers)
    def get_strength(self,x):
        x = self.encoder(x)
        return self.strength(x)
    def forward(self,x):
        x = self.encoder(x)
        strength = self.strength(x)
        x = self.decoder(x)
        return x, strength
    def encode(self,x):
        return self.encoder(x)

class VariationalAutoencoder(nn.Module):
    def __init__(self,input_size,layer_sizes,embedding_size):
        super(VariationalAutoencoder,self).__init__()
        encoder_layers = [nn.Linear(input_size,layer_sizes[0]),nn.ReLU()]
        decoder_layers = [nn.Linear(embedding_size,layer_sizes[-1]),nn.ReLU()]
        # classify_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,8)]
        # regression_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,2)]
        strength_layers = [nn.Linear(embedding_size,20),nn.ReLU(),nn.Linear(20,1)]


        for i in range(len(layer_sizes)-1):
            encoder_layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(layer_sizes[-i-1],layer_sizes[-i-2]))
            decoder_layers.append(nn.ReLU())
        
        decoder_layers.append(nn.Linear(layer_sizes[0],input_size))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        # self.classify = nn.Sequential(*classify_layers)
        # self.regress = nn.Sequential(*regression_layers)
        self.strength = nn.Sequential(*strength_layers)
        self.mu = nn.Linear(layer_sizes[-1],embedding_size)
        self.logvar = nn.Linear(layer_sizes[-1],embedding_size)
    def get_strength(self,x):
        x = self.encoder(x)
        mu = self.mu(x)
        return self.strength(mu)
    def forward(self,x):
        x = self.encoder(x)
        mu = self.mu(x)
        strength = self.strength(mu)
        logvar = self.logvar(x)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        x = self.decoder(x)
        return x, strength
    def encode(self,x):
        x = self.encoder(x)
        mu = self.mu(x)
        return mu

def train(model,train_loader,test_loader,reconstruction_weight,strength_weight,strength_index,device,num_epochs=10, train_strength=True):
    mse_criterion = nn.MSELoss()
    strength_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
    for epoch in range(num_epochs):
        avg_loss = 0
        model.train()
        for data in train_loader:
            readings,classifications = data[0].to(device),data[1]
            if train_strength:
                strengths = classifications[:,strength_index].unsqueeze(1).to(device)
            # ===================forward=====================
            reconstruction,pred_strength = model(readings)
            reconstruction_loss = mse_criterion(reconstruction, readings)
            if train_strength:
                #get indexes where strength is -1
                non_labelled_strength_indexes = np.where(classifications[:,strength_index] == -1)[0]
                #copy over the strength to the pred_strength so that the loss is 0 for the non labelled strength
                pred_strength[non_labelled_strength_indexes] = strengths[non_labelled_strength_indexes]
                strength_loss = strength_criterion(pred_strength,strengths)
        
            # ===================backward====================
            optimizer.zero_grad()
            if train_strength:
                loss = reconstruction_weight*reconstruction_loss +strength_weight*strength_loss
            else:
                loss = reconstruction_loss
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        #validate
        avg_loss = avg_loss/len(train_loader)
        avg_val_loss = 0
        avg_strength_loss = 0
        avg_reconstruction_loss = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                readings,classifications = data[0].to(device),data[1]
                # decode classifications[0] to one hot
                if train_strength:
                    strengths = classifications[:,strength_index].unsqueeze(1).to(device)
                # ===================forward=====================
                reconstruction,pred_strength = model(readings)
                reconstruction_loss = mse_criterion(reconstruction, readings)
                if train_strength:
                    non_labelled_strength_indexes = np.where(classifications[:,strength_index] == -1)[0]
                    pred_strength[non_labelled_strength_indexes] = strengths[non_labelled_strength_indexes]
                    strength_loss = strength_criterion(pred_strength,strengths)
                    avg_strength_loss += strength_loss.item()
                avg_reconstruction_loss += reconstruction_loss.item()
                # ===================log========================
                if train_strength:
                    avg_val_loss +=reconstruction_weight* reconstruction_loss.item()+ strength_weight*strength_loss.item()
                else:
                    avg_val_loss +=reconstruction_loss.item()
        avg_val_loss = avg_val_loss/len(test_loader)
        avg_reconstruction_loss = avg_reconstruction_loss/len(test_loader)
        avg_strength_loss = avg_strength_loss/len(test_loader)

        # ===================log========================
        print('epoch [{}/{}], train loss:{:.6f} val loss:{:.6f} val reconstruction loss {:.6f} val strength loss {:.6f}'.format(epoch + 1, num_epochs, avg_loss, avg_val_loss,avg_reconstruction_loss,avg_strength_loss))

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

def get_embedding(model,test_loader,device):
       embeddings = []
       markers = []
       labels = []
       strengths = []
       model.eval()
       with torch.no_grad():
              for data in test_loader:
                     reconstruction= model.encode(data[0].to(device))
                     strength = model.get_strength(data[0].to(device))
                     
                     # print(data[0].shape,data[1].shape,output.shape)
                     embeddings.append(reconstruction.cpu().numpy())
                     labels.append(data[1].numpy())
                     markers.append(data[0].numpy())
                     strengths.append(strength.cpu().numpy())
       strengths = np.vstack(strengths)
       embeddings = np.vstack(embeddings)
       labels = np.vstack(labels)
       markers = np.vstack(markers)
       return embeddings,labels,markers,strengths

def format_data_into_experiment(embeddings,data_labels,data_index_names,strengths,ignore_columns=[]):
    #data_labels might be longer than embeddings, so we need to remove the extra rows
    if len(embeddings) < len(data_labels):
        data_labels = data_labels[:len(embeddings)]
        strengths = strengths[:len(embeddings)]
    # create a list of dictionaries for each label
    for ignore_column in ignore_columns:
        ignore_index = list(data_index_names).index(ignore_column)
        data_labels = np.delete(data_labels, ignore_index, axis=1)
        data_index_names = np.delete(data_index_names, ignore_index)

    time_index =list(data_index_names).index('Time')
    unique_labels = np.unique(data_labels, axis=0)
    unique_labels_strengths = []
    unique_labels_embeddings = []
    unique_labels_strengths_std = []
    unique_labels_embeddings_std = []
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
        #convert times to doubles from strings
        times = times.astype(float)
        #sort the embeddings_for_label and strengths_for_label by time
        sorted_indexes = np.argsort(times)
        times = times[sorted_indexes]
        embeddings_for_label = embeddings_for_label[sorted_indexes]
        strengths_for_label = strengths_for_label[sorted_indexes]
        output.append({
            'label': unique_experiment_label,
            'times': times,
            'embeddings': embeddings_for_label,
            'embeddings_std': embeddings_for_label_std,
            'strengths': strengths_for_label,
            'strengths_std': strengths_for_label_std
        })
    return output, np.delete(data_index_names, time_index)

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