# How to use pipeline

## Get the dataset

- Ensure the dataset is just outside of the git repository

## Train the model

1. This is done through the `train_model.ipynb` notebook
2. Update any parameters for the autoencoder that you want changed (including the path to the dataset)
3. Put any filters on the data you want in the next cell (eg. only training on OT-1 or removing an antigen etc.)
4. Run the full train_model.ipynb, this will save your parameters to params.txt and the model to autoencoder_test.pt

## Use the model

1. This is done through the `analyze_model.ipynb` notebook
2. Adjust any settings in the first cell (settings will be specific to dataset you are using) but just follow the format (including the path to the dataset you want to analyze)
3. Run the full notebook, and observe the graphs generated below
