
# 算法实现

### 协同过滤

##### Apriori


```python
import numpy as np
from pprint import pprint
```


```python
class Apriori:

    def __init__(self, min_support, min_confidence):
        self.min_support = min_support # 最小支持度
        self.min_confidence = min_confidence # 最小置信度

    def count(self, filename='apriori.txt'):
        self.total = 0 # 数据总行数
        items = {} # 物品清单

        # 统计得到物品清单
        with open(filename) as f:
            for l in f:
                self.total += 1
                for i in l.strip().split(','): # 以逗号隔开
                    if i in items:
                        items[i] += 1.
                    else:
                        items[i] = 1.

        # 物品清单去重，并映射到ID
        self.items = {i:j/self.total for i,j in items.items() if j/self.total > self.min_support}
        self.item2id = {j:i for i,j in enumerate(self.items)}

        # 物品清单的0-1矩阵
        self.D = np.zeros((self.total, len(items)), dtype=bool)

        # 重新遍历文件，得到物品清单的0-1矩阵
        with open(filename) as f:
            for n,l in enumerate(f):
                for i in l.strip().split(','):
                    if i in self.items:
                        self.D[n, self.item2id[i]] = True

    def find_rules(self, filename='apriori.txt'):
        self.count(filename)
        rules = [{(i,):j for i,j in self.items.items()}] # 记录每一步的频繁项集
        l = 0 # 当前步的频繁项的物品数

        while rules[-1]: # 包含了从k频繁项到k+1频繁项的构建过程
            rules.append({})
            keys = sorted(rules[-2].keys()) # 对每个k频繁项按字典序排序（核心）
            num = len(rules[-2])
            l += 1
            for i in range(num): # 遍历每个k频繁项对
                for j in range(i+1,num):
                    # 如果前面k-1个重叠，那么这两个k频繁项就可以组合成一个k+1频繁项
                    if keys[i][:l-1] == keys[j][:l-1]:
                        _ = keys[i] + (keys[j][l-1],)
                        _id = [self.item2id[k] for k in _]
                        support = 1. * sum(np.prod(self.D[:, _id], 1)) / self.total # 通过连乘获取共现次数，并计算支持度
                        if support > self.min_support: # 确认是否足够频繁
                            rules[-1][_] = support

        # 遍历每一个频繁项，计算置信度
        result = {}
        for n,relu in enumerate(rules[1:]): # 对于所有的k，遍历k频繁项
            for r,v in relu.items(): # 遍历所有的k频繁项
                for i,_ in enumerate(r): # 遍历所有的排列，即(A,B,C)究竟是 A,B -> C 还是 A,B -> C 还是 A,B -> C ？
                    x = r[:i] + r[i+1:]
                    confidence = v / rules[n][x] # 不同排列的置信度
                    if confidence > self.min_confidence: # 如果某种排列的置信度足够大，那么就加入到结果
                        result[x+(r[i],)] = (confidence, v)

        return sorted(result.items(), key=lambda x: -x[1][0]) # 按置信度降序排列
```


```python
# 输出代表的意思是，前n-1项可以推出第n项，例如：A3+F4-->H4 confidence=0.879... support=0.0784...
model = Apriori(0.06, 0.75)
pprint(model.find_rules('./Data/apriori.txt'))
```

    [(('A3', 'F4', 'H4'), (0.8795180722891566, 0.07849462365591398)),
     (('C3', 'F4', 'H4'), (0.875, 0.07526881720430108)),
     (('B2', 'F4', 'H4'), (0.7945205479452054, 0.06236559139784946)),
     (('C2', 'E3', 'D2'), (0.7543859649122807, 0.09247311827956989)),
     (('D2', 'F3', 'H4', 'A2'), (0.7532467532467533, 0.06236559139784946))]
    
