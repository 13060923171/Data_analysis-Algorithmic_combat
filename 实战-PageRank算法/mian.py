import networkx as nx
#无向图指的是不用节点之间的边的方向
A = nx.Graph()
#创建有向图,有向图指的是节点之间的边是有方向的
G = nx.DiGraph()
edges = [('A','B'),('A','C'),('A','D'),('B','A'),('B','D'),('C','A'),('D','B'),('D','C')]
for edge in edges:
    G.add_edge(edge[0],edge[1])
pagerank_list = nx.pagerank(G,alpha=1)
print('pagerank值是：',pagerank_list)


#关于节点的增加，删除和查询
#增加节点
B = G.add_node('A')
#也可以使用来添加节点集合
B1 = G.add_nodes_from(['B','C','D','E'])
#删除节点
C = G.remove_node('A')
#也可以集中删除节点
C1 = G.remove_nodes_from(['B','C','D','E'])

#查询所有节点
D = G.nodes()
#查询节点个数
D1 = G.number_of_nodes()

#添加指定的’从A到B‘的边
E = G.add_edge("A","B")
