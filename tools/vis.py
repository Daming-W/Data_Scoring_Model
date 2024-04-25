import pandas as pd
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = '/root/Data_Scoring_Model/logger/log_04%H47'  # 确保路径和文件名正确

# 读取日志文件
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# 解析数据
data = {
    'Epoch': [],
    'Train Loss': [],
    'Train Accuracy': [],
    'Eval Loss': [],
    'Eval Accuracy': []
}

# 解析日志行
index= 0 
for line in lines:
    if "Train epoch loss" in line:
        index+=1
        parts = line.split(',')
        train_loss = float(parts[0].split(':')[-1].strip())
        train_acc = float(parts[1].split(':')[-1].strip())


        data['Epoch'].append(index)
        data['Train Loss'].append(train_loss)
        data['Train Accuracy'].append(train_acc)

    if "Eval" in line:
        eval_loss = float(parts[0].split(':')[-1].strip())
        eval_acc = float(parts[1].split(':')[-1].strip())
        data['Eval Loss'].append(eval_loss)
        data['Eval Accuracy'].append(eval_acc)

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘制loss图
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Train Loss'], 'bo-', label='Training Loss')
plt.plot(df['Epoch'], df['Eval Loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/root/Data_Scoring_Model/plots/training_validation_loss.png')
plt.show()

# 绘制accuracy图
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Train Accuracy'], 'bo-', label='Training Accuracy')
plt.plot(df['Epoch'], df['Eval Accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/root/Data_Scoring_Model/plots/training_validation_accuracy.png')
plt.show()
