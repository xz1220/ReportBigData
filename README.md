<h1 style="text-align: center"> 推荐系统专题报告 </h1>
<div style="text-align: center"><middle>邢政     2017060801025</middle></div>
<div style="text-align: center"><middle>12 - 7 - 2019</middle></div>

___

>实验报告简介，主要说明做了什么


## 推荐系统简介

随着信息技术的不断发展，如今互联网已经成为了人们日常生活中密不可分的一部分。人们每天都会在互联网上进行各种各样的活动例如看电影、购物、阅读新闻时事等，但随着互联网上信息的越来越多，人们发现越来越难以从互联网上的海量信息中找出最适合自己的那些，例如当你登录Netflix想要看一部电影放松时却不知道哪一部符合自己的口味。推荐系统的出现正是为了解决这种“信息过载”的问题，它会预测用户的需求并推荐给用户其最可能喜欢的内容，缓解了人们从海量信息中做出选择的烦恼。

在推荐系统发展的早期，常见的推荐方法是简单的依据物品的销量、话题的点击量或新闻的阅读量等进行排序，然后选取排在最前面的N个物品组成排行榜并推荐给用户。这种方法具有非常不错的效果，直到今天我们仍能经常在各大网站上看到类似的功能。但另一方面，这种方法也存在着巨大的缺陷，即只有少量的排在前列的物品能够得到推荐，更多的物品则被埋没不为人知，根据营销中的“长尾理论”我们知道细小市场的累积所产生的利润同样是巨大的，因此如何充分利用已有资源（物品），并使得推荐尽可能准确，成为了推荐系统领域研究的主要目标，由此个性化推荐系统应运而生。个性化推荐系统，顾名思义即指根据用户的个性与偏好来产生推荐内容，由于不同用户的个性与偏好存在差异，因此对其推荐的内容也会有所不同，这样一方面可以使得更多的物品得到推荐，利于获取更多小的细分市场的利润；另一方面，由于推荐是根据用户的偏好产生的，因此推荐成功的概率也更高。

现如今，推荐系统已在互联网中得到了广泛的应用，并给应用它的企业带来了丰厚的利润。据报道，推荐系统给亚马逊带来了35%的销售收入，给Netflix带来了高达75%的消费，并且Youtube主页上60%的浏览来自推荐服务。因此，有关推荐系统的研究具有十分深远的意义与巨大的实用价值。


## 经典的协同过滤方法

**协同过滤**作为推荐算法中最经典的类型，包括在线的协同和离线的过滤两部分。所谓在线协同，就是通过在线数据找到用户可能喜欢的物品，而离线过滤，则是过滤掉一些不值得推荐的数据，比如推荐值评分低的数据，或者虽然推荐值高但是用户已经购买的数据。

协同过滤的模型一般为m个物品，m个用户的数据，只有部分用户和部分数据之间是有评分数据的，其它部分评分是空白，此时我们要用已有的部分稀疏数据来预测那些空白的物品和数据之间的评分关系，找到最高评分的物品推荐给用户。

一般来说，协同过滤推荐分为三种类型。第一种是**基于用户**的协同过滤，第二种是**基于项目**的协同过滤，第三种是**基于模型**的协同过滤。

### 基于用户和基于项目的协同过滤

基于用户(user-based)的协同过滤主要考虑的是用户和用户之间的相似度，只要找出相似用户喜欢的物品，并预测目标用户对对应物品的评分，就可以找到评分最高的若干个物品推荐给用户。而基于项目(item-based)的协同过滤和基于用户的协同过滤类似，只不过这时我们转向找到物品和物品之间的相似度，只有找到了目标用户对某些物品的评分，那么我们就可以对相似度高的类似物品进行预测，将评分最高的若干个相似物品推荐给用户。

### 基于模型的协同过滤

本文的重点之一就是基于模型的**协同过滤**。基于模型的协同过滤作为目前最主流的协同过滤类型。其主要思想是，有m个物品，m个用户的数据，只有部分用户和部分数据之间是有评分数据的，其它部分评分是空白，此时我们要用已有的部分稀疏数据来预测那些空白的物品和数据之间的评分关系，找到最高评分的物品推荐给用户。

针对于这个问题，用机器学习的思想来建模解决，主流的方法可以分为：用关联算法，聚类算法，分类算法，回归算法，矩阵分解，神经网络,图模型以及隐语义模型来解决。

#### 关联算法
关联推荐是对最近邻使用者的记录进行关联规则(association rules)挖掘。关联规则分析 (Association Rules，又称 Basket Analysis) 用于从大量数据中挖掘出有价值的数据项之间的相关关系。关联规则解决的常见问题如：“如果一个消费者购买了产品A，那么他有多大机会购买产品B?”以及“如果他购买了产品C和D，那么他还将购买什么产品？”

- 常见的关联推荐算法
    - Apriori 算法
    - FP Tree 算法
    - PrefixSpan 算法

接下来逐一进行讲解。

##### Apriori 算法

首先我们需要解释关于Apriori[<sup>1</sup>](#Apriori)的一些相关的概念：

**定义 1** ： 两个不相交的非空集合 $X,Y$，如果有 $X \rightarrow Y$ ，就说 $X \rightarrow Y$ 是一条关联规则。如吃咸菜的人偏爱喝粥（ $咸菜 \rightarrow 粥$ ）就是一条关联规则。 关联规则的强度(可信度)用支持度(support)和自信度(confidence)来描述。

**定义 2** 项集 $X,Y$ 种同时发生的概率称为关联规则 $X \rightarrow Y$ 的支持度(support)：
$$
Support(X \rightarrow Y) = P(X \cup Y) = P(XY)
$$最小支持度是用户或专家定义的用来衡量支持度的一个阈值，表示关联规则具有统计学意义的最低重要性。具有“统计学意义”的显著特征就是这个事件发生的概率/频率不能太低（否则就不具有推广到其他个体的价值）。

由于现实生活中，常用古典概型估计概率，因此，上式也可写为：$$
Support(X \rightarrow Y) = X,Y同时发生的事件个数 / 总事件数
$$

**定义 3** 项集 $X$ 发生的前提下，项集 $Y$ 发生的概率称为关联规则 $X \rightarrow Y$ 的自信度(confidence 条件概率)：

$$
Confidence(X \rightarrow Y) = P(Y \mid X)
 $$最小置信度是用户或专家定义的用来衡量置信度的一个阈值，表示关联规则的最低可靠性。同理，在古典概型中，上式也可以写为：$$
 Confidence(X \rightarrow Y) = {X,Y同时发生的事件个数}/{X发生的事件个数}
 $$

 所以接下来我们**要做的工作**就是：
 1. 生成频繁项集：
这一阶段找出所有满足最小支持度的项集（具有统计学意义的组合），找出的这些项集称为频繁项集。自信度与支持度的计算涉及到多个列表的循环，极大影响频繁项集的生成时间。

2. 生成关联规则：
在上一步产生的频繁项集的基础上生成满足最小自信度的规则，产生的规则称为强规则。

在这里我们有**两个重要的定理**：
1. 如果一个集合是频繁项集，则它的所有子集都是频繁项集。假设一个集合 {A,B} 是频繁项集，则它的子集 {A}, {B} 都是频繁项集。
2. 如果一个集合不是频繁项集，则它的所有超集都不是频繁项集。假设集合 {A} 不是频繁项集，则它的任何超集如 {A,B}，{A,B,C} 必定也不是频繁项集。

根据定理1和定理2易知：若 $X \rightarrow Y$ 是强规则，则 $X,Y,XY$ 都必须是频繁项集。

**算法如下图**：
<img height="500px" src="img/Apriori .jpg"/>

**代码实现如下**：

```python
import numpy as np
from pprint import pprint

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


# 输出代表的意思是，前n-1项可以推出第n项，例如：A3+F4-->H4 confidence=0.879... support=0.0784...
model = Apriori(0.06, 0.75)
pprint(model.find_rules('./Data/apriori.txt'))
```

    [(('A3', 'F4', 'H4'), (0.8795180722891566, 0.07849462365591398)),
     (('C3', 'F4', 'H4'), (0.875, 0.07526881720430108)),
     (('B2', 'F4', 'H4'), (0.7945205479452054, 0.06236559139784946)),
     (('C2', 'E3', 'D2'), (0.7543859649122807, 0.09247311827956989)),
     (('D2', 'F3', 'H4', 'A2'), (0.7532467532467533, 0.06236559139784946))]


    





## 基于深度学习的推荐方法




## References

[1] [Fast Algorithms for Mining Association Rules](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=011922E95979A9A163656A1CC432BE46?doi=10.1.1.100.2474&rep=rep1&type=pdf) <div id="Apriori"></div>





