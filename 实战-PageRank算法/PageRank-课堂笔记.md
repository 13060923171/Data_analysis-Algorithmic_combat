## PageRank的简化模型

举个例子：，假设有4个网页

![image-20210811114820190](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811114820.png)

在开始之前有两个重要概念需要了解一下：

1. 出链指的是链接出去的链接。
2. 入链指的是链接进来的链接。
3. 比如是图中有两个是入链，3个出链

一个网页的影响力 = 所有入链集合的页面的加权影响力之和，用公式表示：

![image-20210811115154260](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811115154.png)

在上面的例子中可以看到，A有三个出链分别链接到了B,C,D上。在访问A的时候，就有跳到B,C或者D的可能性为1/3

B有两个出链，链接到A和D上，跳转的概率为1/2

![image-20210811115358532](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811115358.png)

![image-20210811115430676](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811115430.png)

![image-20210811115643000](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811115643.png)

从这里可以看出，A页面相比其他页面的权重更大，也就是PR值更高，而B,C,D页面的PR值相同

虽然是这样说，但是我们要面临两个问题：

1. 等级泄露，如果一个网页没有出链，就会像一个黑洞一样，吸收了其他网页的影响力而不释放，最终会导致其他网页的PR值为0

   ![image-20210811115908998](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811115909.png)

2. 等级沉没，如果一个网页只有出链，没有入链，计算的过程迭代下来，会导致这个网页的PR值为0

   ![image-20210811120012710](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811120012.png)

## 针对这种问题的出现，PageRank的随机预览模型就出现了

计算公式如下：

![image-20210811120202895](https://gitee.com/mqsnq30/gitee-table/raw/master/img/20210811120202.png)

## PageRank在社交影响力评估中的应用

网页之间会形成一个网络，是我们的互联网，论文之间也存在着相互引用的关系，可以说我们所处的环境计算各种网络的集合。

只要有网络的地方，就存在出链和入链，就会有PR权重的计算，也就可以运用PageRank算法