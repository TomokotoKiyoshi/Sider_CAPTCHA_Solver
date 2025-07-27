"""
分析训练日志，找出最优的epoch
"""
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_training_log(log_file):
    """解析训练日志文件"""
    epochs = []
    train_losses = []
    val_losses = []
    maes = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取每个epoch的信息
    epoch_pattern = r'Epoch (\d+)/\d+:.*?Train Loss: ([\d.]+).*?Val Loss: ([\d.]+), MAE: ([\d.]+) pixels'
    matches = re.findall(epoch_pattern, content, re.DOTALL)
    
    for match in matches:
        epoch, train_loss, val_loss, mae = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        maes.append(float(mae))
    
    return epochs, train_losses, val_losses, maes

def find_best_models(epochs, val_losses, maes):
    """找出最优的模型"""
    results = []
    
    # 最低验证损失
    if val_losses:
        min_val_loss_idx = val_losses.index(min(val_losses))
        results.append({
            'criteria': 'Lowest Validation Loss',
            'epoch': epochs[min_val_loss_idx],
            'val_loss': val_losses[min_val_loss_idx],
            'mae': maes[min_val_loss_idx]
        })
    
    # 最低MAE
    if maes:
        min_mae_idx = maes.index(min(maes))
        results.append({
            'criteria': 'Lowest MAE',
            'epoch': epochs[min_mae_idx],
            'val_loss': val_losses[min_mae_idx],
            'mae': maes[min_mae_idx]
        })
    
    return results

def analyze_checkpoints_dir(checkpoint_dir):
    """分析训练日志"""
    checkpoint_path = Path(checkpoint_dir)
    
    # 如果是checkpoints根目录，查找最新的版本目录
    if checkpoint_path.name == 'checkpoints':
        version_dirs = []
        for dir_path in checkpoint_path.iterdir():
            if dir_path.is_dir():
                import re
                if re.match(r'^\d+\.\d+\.\d+$', dir_path.name) or dir_path.name.startswith('v'):
                    log_files = list(dir_path.glob('training_log_*.txt'))
                    if log_files:
                        version_dirs.append(dir_path)
        
        if version_dirs:
            # 按版本号排序，优先使用新版本格式
            def version_key(path):
                name = path.name
                if re.match(r'^\d+\.\d+\.\d+$', name):
                    parts = name.split('.')
                    return (0, int(parts[0]), int(parts[1]), int(parts[2]))
                elif name.startswith('v') and re.match(r'^v\d+\.\d+\.\d+$', name):
                    parts = name[1:].split('.')
                    return (1, int(parts[0]), int(parts[1]), int(parts[2]))
                else:
                    return (2, name)
            
            version_dirs.sort(key=version_key, reverse=True)
            checkpoint_path = version_dirs[0]
            print(f"Using latest version directory: {checkpoint_path}")
    
    # 查找training_log文件
    log_files = list(checkpoint_path.glob('training_log_*.txt'))
    if not log_files:
        # 尝试logs子目录
        logs_dir = checkpoint_path / 'logs'
        if logs_dir.exists():
            log_files = list(logs_dir.glob('training_log_*.txt'))
        
        if not log_files:
            print(f"No training log found in {checkpoint_path}")
            return
    
    # 使用最新的日志文件
    log_file = sorted(log_files)[-1]
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Analyzing: {log_file}")
    print("-" * 60)
    
    # 解析日志
    epochs, train_losses, val_losses, maes = parse_training_log(log_file)
    
    if not epochs:
        print("No training data found in log.")
        return
    
    # 显示训练进度
    print(f"Total epochs trained: {len(epochs)}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Final MAE: {maes[-1]:.2f} pixels")
    print()
    
    # 找出最优模型
    best_models = find_best_models(epochs, val_losses, maes)
    
    print("Best Models:")
    print("-" * 60)
    for model in best_models:
        print(f"{model['criteria']}:")
        print(f"  Epoch: {model['epoch']}")
        print(f"  Val Loss: {model['val_loss']:.4f}")
        print(f"  MAE: {model['mae']:.2f} pixels")
        print(f"  Checkpoint: src/checkpoints/epoch_{model['epoch']:03d}.pth")
        print()
    
    # 绘制训练曲线
    if len(epochs) > 1:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, maes)
        plt.xlabel('Epoch')
        plt.ylabel('MAE (pixels)')
        plt.title('Mean Absolute Error')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, val_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = logs_dir / 'training_curves.png'
        plt.savefig(plot_path)
        print(f"Training curves saved to: {plot_path}")
        plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing training checkpoints')
    args = parser.parse_args()
    
    # 如果没有指定，使用项目根目录的checkpoints
    if args.checkpoint_dir is None:
        project_root = Path(__file__).parent.parent.parent
        args.checkpoint_dir = str(project_root / 'src' / 'checkpoints')
    
    analyze_checkpoints_dir(args.checkpoint_dir)