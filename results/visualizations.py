from pickle import load
from matplotlib.pylab import plt
from numpy import arange
 
# Load the training and validation loss dictionaries
train_loss = load(open('s_MLP_t_simglucose_data_insilico_seed_56_2024-02-18/train_loss.pkl', 'rb'))
val_loss = load(open('s_MLP_t_simglucose_data_insilico_seed_56_2024-02-18/ii_loss.pkl', 'rb'))

assert len(train_loss)==len(val_loss)
# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, len(train_loss)+1)
 
# Plot and label the training and ii loss values
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='II Loss',alpha=0.5)
 
# Add in a title and axes labels
plt.title('Regular and II Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, len(train_loss)+1, 5000))
 
# Display the plot
plt.legend(loc='best')
plt.show()