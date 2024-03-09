import os
from pickle import load
from matplotlib.pylab import plt
from numpy import arange

parent_folder = input("Which is your student model? [MLP_parallel, MLP_tree, MLP_joint, MLP_scaled]")
assert parent_folder in ["MLP_parallel", "MLP_tree", "MLP_joint", "MLP_scaled"]

date_exp = input("What is the date of the experiment (%YYYY-MM-DD)")

pred_h = input("What is your PH [30,45,60,120]")
assert pred_h in ["30","45","60", "120"]
# Load the training and validation loss dictionaries
train_loss = load(open(os.path.join('results',parent_folder,f's_{parent_folder}_t_simglucose_data_insilico_seed_56_{date_exp}_PH_{pred_h}','train_loss.pkl'), 'rb'))
val_loss = load(open(os.path.join('results',parent_folder,f's_{parent_folder}_t_simglucose_data_insilico_seed_56_{date_exp}_PH_{pred_h}','ii_loss.pkl'), 'rb'))

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