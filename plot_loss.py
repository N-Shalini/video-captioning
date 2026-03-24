import json
import matplotlib.pyplot as plt
import os

def plot_loss(history_path='models/loss_history.json', save_path='loss_curve.png'):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    epochs = range(1, len(train_loss) + 1)
    
    # Use a clean, modern aesthetic
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2.5)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2.5)
    
    plt.title('Video Captioning Model: Training vs Validation Loss', fontsize=16, pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Cross-Entropy Loss', fontsize=14)
    plt.xticks(epochs)
    
    # Highlight the best epoch
    if val_loss:
        best_epoch = val_loss.index(min(val_loss)) + 1
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.8, 
                    label=f'Best Model (Epoch {best_epoch})', linewidth=2)
        plt.scatter(best_epoch, min(val_loss), color='green', s=100, zorder=5)
        
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curve successfully saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_loss()
