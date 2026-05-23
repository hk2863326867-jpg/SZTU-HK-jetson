import os
import shutil

def clean_checkpoints():
    """
    清理检查点目录，只保留最佳模型
    """
    best_dir = 'D:/dien_checkpoints/best'
    save_dir = 'D:/dien_checkpoints/save'
    
    print('Cleaning checkpoint directory...')
    
    # 清理 save 目录（保留最佳模型，删除其他检查点）
    if os.path.exists(save_dir):
        print(f'Cleaning save directory: {save_dir}')
        # 删除整个 save 目录
        shutil.rmtree(save_dir)
        print('Save directory deleted')
        # 重新创建空的 save 目录
        os.makedirs(save_dir, exist_ok=True)
    
    # 检查最佳模型目录
    if os.path.exists(best_dir):
        print(f'Best model directory exists: {best_dir}')
        files = os.listdir(best_dir)
        print(f'Best model files: {files}')
    else:
        print(f'Best model directory not found: {best_dir}')
    
    print('Cleanup completed!')

if __name__ == "__main__":
    clean_checkpoints()