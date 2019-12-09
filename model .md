
# 算法实现

### 协同过滤

#### Apriori


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
    

#### FP-GRowth


```python
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        # needs to be updated
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        """inc(对count变量增加给定值)
        """
        self.count += numOccur

    def disp(self, ind=1):
        """disp(用于将树以文本形式显示)

        """
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
            #    ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict


# this version does not use recursion
def updateHeader(nodeToTest, targetNode):
    """updateHeader(更新头指针，建立相同元素之间的关系，例如： 左边的r指向右边的r值，就是后出现的相同元素 指向 已经出现的元素)

    从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    性能：如果链表很长可能会遇到迭代调用的次数限制。

    Args:
        nodeToTest  满足minSup {所有的元素+(value, treeNode)}
        targetNode  Tree对象的子节点
    """
    # 建立相同元素之间的关系，例如： 左边的r指向右边的r值
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(items, inTree, headerTable, count):
    """updateTree(更新FP-tree，第二次遍历)

    # 针对每一行的数据
    # 最大的key,  添加
    Args:
        items       满足minSup 排序后的元素key的数组（大到小的排序）
        inTree      空的Tree对象
        headerTable 满足minSup {所有的元素+(value, treeNode)}
        count       原数据集中每一组Kay出现的次数
    """
    # 取出 元素 出现次数最高的
    # 如果该元素在 inTree.children 这个字典中，就进行累加
    # 如果该元素不存在 就 inTree.children 字典中新增key，value为初始化的 treeNode 对象
    if items[0] in inTree.children:
        # 更新 最大元素，对应的 treeNode 对象的count进行叠加
        inTree.children[items[0]].inc(count)
    else:
        # 如果不存在子节点，我们为该inTree添加子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 如果满足minSup的dist字典的value值第二位为null， 我们就设置该元素为 本节点对应的tree节点
        # 如果元素第二位不为null，我们就更新header节点
        if headerTable[items[0]][1] is None:
            # headerTable只记录第一次节点出现的位置
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 本质上是修改headerTable的key对应的Tree，的nodeLink值
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 递归的调用，在items[0]的基础上，添加item0[1]做子节点， count只要循环的进行累计加和而已，统计出节点的最后的统计值。
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def createTree(dataSet, minSup=1):
    """createTree(生成FP-tree)

    Args:
        dataSet  dist{行：出现次数}的样本数据
        minSup   最小的支持度
    Returns:
        retTree  FP-tree
        headerTable 满足minSup {所有的元素+(value, treeNode)}
    """
    # 支持度>=minSup的dist{所有元素：出现的次数}
    headerTable = {}
    # 循环 dist{行：出现次数}的样本数据
    for trans in dataSet:
        # 对所有的行进行循环，得到行里面的所有元素
        # 统计每一行中，每个元素出现的总次数
        for item in trans:
            # 例如： {'ababa': 3}  count(a)=3+3+3=9   count(b)=3+3=6
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 删除 headerTable中，元素次数<最小支持度的元素
    for k in list(headerTable.keys()):  # python3中.keys()返回的是迭代器不是list,不能在遍历时对其改变。
        if headerTable[k] < minSup:
            del(headerTable[k])

    # 满足minSup: set(各元素集合)
    freqItemSet = set(headerTable.keys())
    # 如果不存在，直接返回None
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        # 格式化： dist{元素key: [元素次数, None]}
        headerTable[k] = [headerTable[k], None]

    # create tree
    retTree = treeNode('Null Set', 1, None)
    # 循环 dist{行：出现次数}的样本数据
    for tranSet, count in dataSet.items():
        # print('tranSet, count=', tranSet, count)
        # localD = dist{元素key: 元素总出现次数}
        localD = {}
        for item in tranSet:
            # 判断是否在满足minSup的集合中
            if item in freqItemSet:
                # print('headerTable[item][0]=', headerTable[item][0], headerTable[item])
                localD[item] = headerTable[item][0]
        # print('localD=', localD)
        # 对每一行的key 进行排序，然后开始往树添加枝丫，直到丰满
        # 第二次，如果在同一个排名下出现，那么就对该枝丫的值进行追加，继续递归调用！
        if len(localD) > 0:
            # p=key,value; 所以是通过value值的大小，进行从大到小进行排序
            # orderedItems 表示取出元组的key值，也就是字母本身，但是字母本身是大到小的顺序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print 'orderedItems=', orderedItems, 'headerTable', headerTable, '\n\n\n'
            # 填充树，通过有序的orderedItems的第一位，进行顺序填充 第一层的子节点。
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def ascendTree(leafNode, prefixPath):
    """ascendTree(如果存在父节点，就记录当前节点的name值)

    Args:
        leafNode   查询的节点对于的nodeTree
        prefixPath 要查询的节点值
    """
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    """findPrefixPath 基础数据集

    Args:
        basePat  要查询的节点值
        treeNode 查询的节点所在的当前nodeTree
    Returns:
        condPats 对非basePat的倒叙值作为key,赋值为count数
    """
    condPats = {}
    # 对 treeNode的link进行循环
    while treeNode is not None:
        prefixPath = []
        # 寻找改节点的父节点，相当于找到了该节点的频繁项集
        ascendTree(treeNode, prefixPath)
        # 排除自身这个元素，判断是否存在父元素（所以要>1, 说明存在父元素）
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            # prefixPath[1:] 变frozenset后，字母就变无序了
            # condPats[frozenset(prefixPath)] = treeNode.count
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
        # print(treeNode)
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """mineTree(创建条件FP树)

    Args:
        inTree       myFPtree
        headerTable  满足minSup {所有的元素+(value, treeNode)}
        minSup       最小支持项集
        preFix       preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        freqItemList 用来存储频繁子项的列表
    """
    # 通过value进行从小到大的排序， 得到频繁项集的key
    # 最小支持项集的key的list集合
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    print('-----', sorted(headerTable.items(), key=lambda p: p[1][0]))
    print('bigL=', bigL)
    # 循环遍历 最频繁项集的key，从小到大的递归寻找对应的频繁项集
    for basePat in bigL:
        # preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('newFreqSet=', newFreqSet, preFix)

        freqItemList.append(newFreqSet)
        print('freqItemList=', freqItemList)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases=', basePat, condPattBases)

        # 构建FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        print('myHead=', myHead)
        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if myHead is not None:
            myCondTree.disp(1)
            print('\n\n\n')
            # 递归 myHead 找出频繁项集
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        print('\n\n\n')
```


```python

# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# # 将树以文本形式显示
# # print(rootNode.disp())

# load样本数据
simpDat = loadSimpDat()
# print(simpDat, '\n')
# frozen set 格式化 并 重新装载 样本数据，对所有的行进行统计求和，格式: {行：出现次数}
initSet = createInitSet(simpDat)
print(initSet)

# 创建FP树
# 输入：dist{行：出现次数}的样本数据  和  最小的支持度
# 输出：最终的PF-tree，通过循环获取第一层的节点，然后每一层的节点进行递归的获取每一行的字节点，也就是分支。
# 然后所谓的指针，就是后来的指向已存在的
myFPtree, myHeaderTab = createTree(initSet, 3)
myFPtree.disp()

# 抽取条件模式基
# 查询树节点的，频繁子项
print('x --->', findPrefixPath('x', myHeaderTab['x'][1]))
print('z --->', findPrefixPath('z', myHeaderTab['z'][1]))
print('r --->', findPrefixPath('r', myHeaderTab['r'][1]))

# 创建条件模式基
freqItemList = []
mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList)
print("freqItemList: \n", freqItemList)

```

    {frozenset({'p', 'j', 'r', 'h', 'z'}): 1, frozenset({'t', 'y', 'w', 'x', 'v', 's', 'z', 'u'}): 1, frozenset({'z'}): 1, frozenset({'o', 'n', 'x', 'r', 's'}): 1, frozenset({'t', 'y', 'q', 'p', 'x', 'r', 'z'}): 1, frozenset({'t', 'y', 'q', 'm', 'e', 'x', 's', 'z'}): 1}
       Null Set   1
         z   5
           r   1
           x   3
             t   3
               y   3
                 s   2
                 r   1
         x   1
           r   1
             s   1
    x ---> {frozenset({'z'}): 3}
    z ---> {}
    r ---> {frozenset({'z'}): 1, frozenset({'x'}): 1, frozenset({'z', 't', 'y', 'x'}): 1}
    ----- [('r', [3, <__main__.treeNode object at 0x0000026CEA22B080>]), ('t', [3, <__main__.treeNode object at 0x0000026CEA22B5F8>]), ('y', [3, <__main__.treeNode object at 0x0000026CEA22BF60>]), ('s', [3, <__main__.treeNode object at 0x0000026CEA22B358>]), ('x', [4, <__main__.treeNode object at 0x0000026CEA22B630>]), ('z', [5, <__main__.treeNode object at 0x0000026CEA22B208>])]
    bigL= ['r', 't', 'y', 's', 'x', 'z']
    newFreqSet= {'r'} set()
    freqItemList= [{'r'}]
    condPattBases= r {frozenset({'z'}): 1, frozenset({'x'}): 1, frozenset({'z', 't', 'y', 'x'}): 1}
    myHead= None
    
    
    
    
    newFreqSet= {'t'} set()
    freqItemList= [{'r'}, {'t'}]
    condPattBases= t {frozenset({'z', 'x'}): 3}
    myHead= {'z': [3, <__main__.treeNode object at 0x0000026CEA231668>], 'x': [3, <__main__.treeNode object at 0x0000026CEA231630>]}
       Null Set   1
         z   3
           x   3
    
    
    
    
    ----- [('z', [3, <__main__.treeNode object at 0x0000026CEA231668>]), ('x', [3, <__main__.treeNode object at 0x0000026CEA231630>])]
    bigL= ['z', 'x']
    newFreqSet= {'t', 'z'} {'t'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}]
    condPattBases= z {}
    myHead= None
    
    
    
    
    newFreqSet= {'t', 'x'} {'t'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}]
    condPattBases= x {frozenset({'z'}): 3}
    myHead= {'z': [3, <__main__.treeNode object at 0x0000026CEA231A90>]}
       Null Set   1
         z   3
    
    
    
    
    ----- [('z', [3, <__main__.treeNode object at 0x0000026CEA231A90>])]
    bigL= ['z']
    newFreqSet= {'t', 'z', 'x'} {'t', 'x'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}]
    condPattBases= z {}
    myHead= None
    
    
    
    
    
    
    
    
    
    
    
    
    newFreqSet= {'y'} set()
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}]
    condPattBases= y {frozenset({'t', 'z', 'x'}): 3}
    myHead= {'t': [3, <__main__.treeNode object at 0x0000026CEA231D68>], 'z': [3, <__main__.treeNode object at 0x0000026CEA231E10>], 'x': [3, <__main__.treeNode object at 0x0000026CEA231E48>]}
       Null Set   1
         t   3
           z   3
             x   3
    
    
    
    
    ----- [('t', [3, <__main__.treeNode object at 0x0000026CEA231D68>]), ('z', [3, <__main__.treeNode object at 0x0000026CEA231E10>]), ('x', [3, <__main__.treeNode object at 0x0000026CEA231E48>])]
    bigL= ['t', 'z', 'x']
    newFreqSet= {'t', 'y'} {'y'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}]
    condPattBases= t {}
    myHead= None
    
    
    
    
    newFreqSet= {'z', 'y'} {'y'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}]
    condPattBases= z {frozenset({'t'}): 3}
    myHead= {'t': [3, <__main__.treeNode object at 0x0000026CEA2231D0>]}
       Null Set   1
         t   3
    
    
    
    
    ----- [('t', [3, <__main__.treeNode object at 0x0000026CEA2231D0>])]
    bigL= ['t']
    newFreqSet= {'z', 't', 'y'} {'z', 'y'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}]
    condPattBases= t {}
    myHead= None
    
    
    
    
    
    
    
    
    newFreqSet= {'y', 'x'} {'y'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}]
    condPattBases= x {frozenset({'t', 'z'}): 3}
    myHead= {'t': [3, <__main__.treeNode object at 0x0000026CEA223320>], 'z': [3, <__main__.treeNode object at 0x0000026CEA2233C8>]}
       Null Set   1
         t   3
           z   3
    
    
    
    
    ----- [('t', [3, <__main__.treeNode object at 0x0000026CEA223320>]), ('z', [3, <__main__.treeNode object at 0x0000026CEA2233C8>])]
    bigL= ['t', 'z']
    newFreqSet= {'t', 'y', 'x'} {'y', 'x'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}]
    condPattBases= t {}
    myHead= None
    
    
    
    
    newFreqSet= {'z', 'y', 'x'} {'y', 'x'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}]
    condPattBases= z {frozenset({'t'}): 3}
    myHead= {'t': [3, <__main__.treeNode object at 0x0000026CEA2236D8>]}
       Null Set   1
         t   3
    
    
    
    
    ----- [('t', [3, <__main__.treeNode object at 0x0000026CEA2236D8>])]
    bigL= ['t']
    newFreqSet= {'z', 't', 'y', 'x'} {'z', 'y', 'x'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}]
    condPattBases= t {}
    myHead= None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    newFreqSet= {'s'} set()
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}]
    condPattBases= s {frozenset({'z', 't', 'y', 'x'}): 2, frozenset({'r', 'x'}): 1}
    myHead= {'x': [3, <__main__.treeNode object at 0x0000026CEA223828>]}
       Null Set   1
         x   3
    
    
    
    
    ----- [('x', [3, <__main__.treeNode object at 0x0000026CEA223828>])]
    bigL= ['x']
    newFreqSet= {'s', 'x'} {'s'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}, {'s', 'x'}]
    condPattBases= x {}
    myHead= None
    
    
    
    
    
    
    
    
    newFreqSet= {'x'} set()
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}, {'s', 'x'}, {'x'}]
    condPattBases= x {frozenset({'z'}): 3}
    myHead= {'z': [3, <__main__.treeNode object at 0x0000026CEA223AC8>]}
       Null Set   1
         z   3
    
    
    
    
    ----- [('z', [3, <__main__.treeNode object at 0x0000026CEA223AC8>])]
    bigL= ['z']
    newFreqSet= {'z', 'x'} {'x'}
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}, {'s', 'x'}, {'x'}, {'z', 'x'}]
    condPattBases= z {}
    myHead= None
    
    
    
    
    
    
    
    
    newFreqSet= {'z'} set()
    freqItemList= [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}, {'s', 'x'}, {'x'}, {'z', 'x'}, {'z'}]
    condPattBases= z {}
    myHead= None
    
    
    
    
    freqItemList: 
     [{'r'}, {'t'}, {'t', 'z'}, {'t', 'x'}, {'t', 'z', 'x'}, {'y'}, {'t', 'y'}, {'z', 'y'}, {'z', 't', 'y'}, {'y', 'x'}, {'t', 'y', 'x'}, {'z', 'y', 'x'}, {'z', 't', 'y', 'x'}, {'s'}, {'s', 'x'}, {'x'}, {'z', 'x'}, {'z'}]
    
