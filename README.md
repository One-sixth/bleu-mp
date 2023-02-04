# bleu-mp

Multi-process BLEU evaluation tool.  
多进程BLEU评估工具。  

Modified from the bleu scoring tool of huggingface evaluation.  
改自 huggingface evaluate 的 bleu 评分工具。  
https://github.com/huggingface/evaluate/blob/main/metrics/bleu/bleu.py  

# Install / 安装
pip
```
pip install -U bleu-mp
```

dev
```
git clone https://github.com/One-sixth/bleu-mp
cd bleu-mp
pip install -e .
```

# New Features / 新特性
1. Faster!  
2. The built-in multi-process implementation of python is not used. I use my own multi-process implementation, which is very friendly to the compatibility of Windows and Linux. The memory occupied by each calculation subprocess is very low.  

---
1. 更快！  
2. 不使用python内置的多进程实现。使用我自己的多进程实现，从而对windows和linux的兼容性非常友好，每个计算子进程占用的内存非常低。  

# Features / 特性
Both string and integer sequences are supported for bleu calculation.  
同时支持 字符串和整数序列 进行bleu计算。  

# Speed test / 速度测试
Test code is in unittest/test.py.  
测试代码位于 unittest/test.py。  

CPU：i7-8750H
```
# short str / 短字符串
score (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 2200000, 2200000) (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 2200000, 2200000)
1  process cost time 16.979528665542603
10 process cost time 3.5354034900665283

# long str / 长字符串
score (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 22000000, 22000000) (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 22000000, 22000000)
1  process cost time 103.8217351436615
10 process cost time 22.66322374343872

# short int list / 短整数序列
score (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 800000, 800000) (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 800000, 800000)
1  process cost time 4.874496936798096
10 process cost time 1.1751139163970947

# long int list / 长整数序列
score (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 16000000, 16000000) (1.0, [1.0, 1.0, 1.0, 1.0], 1.0, 1.0, 16000000, 16000000)
1  process cost time 47.34107685089111
10 process cost time 10.046519994735718

```

# Warning / 警告
Don't input pytorch's tensor type. It will causes unnecessary memory consumption, and a lot of performance loss.  
Please convert to numpy array or list type first.  

不要传入 pytorch 的 tensor 类型，这会导致额外的内存消耗和大量的性能损失。  
请先转换到 numpy数组 或 list类型。  

# Demo / 示例
```python
from bleu_mp import compute_bleu

# str
pred_data = ['床前明月光，疑是地上霜', '举头望明月，低头思故乡'] * 1000
tgt_data = [['床前明月光，疑是地上霜'], ['举头望明月，低头思故乡', '静夜思']] * 1000
result = compute_bleu(pred_data, tgt_data)
print('bleu score', result[0])

# int list
pred_data = [[1, 2, 3, 4], [2, 3, 4, 5]] * 1000
tgt_data = [[[1, 2, 3, 4]], [[2, 3, 4, 5], [4, 5, 6]]] * 1000
result = compute_bleu(pred_data, tgt_data)
print('bleu score', result[0])
```

# Reference / 引用
https://github.com/huggingface/evaluate