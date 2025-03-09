import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_loss(losses, epochs=None, title="Training and Validation Loss", 
                       figsize=(10, 6), save_path=None, show=True):
    train_losses = losses['train_loss']
    val_losses = losses['val_loss']
    
    if epochs is None:
        epochs = np.arange(1, len(train_losses) + 1)
    
    data = pd.DataFrame({
        'Epoch': np.concatenate([epochs, epochs]),
        'Loss': np.concatenate([train_losses, val_losses]),
        'Type': ['Training'] * len(train_losses) + ['Validation'] * len(val_losses)
    })
    
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    
    ax = sns.lineplot(x='Epoch', y='Loss', hue='Type', data=data, marker='o')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    return plt.gcf()