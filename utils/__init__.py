# import networkx as nx
#
# # 指定 Graph 文件路径
# graph_file = "/home/xyNLP/data/kl/DRGN-main/data/cpnet/conceptnet.en.pruned.graph"
#
# # 读取 Graph 文件
# graph = nx.read_gpickle(graph_file)
#
# num_nodes = graph.number_of_nodes()
# num_edges = graph.number_of_edges()
#
# print("num_nodes:")
# print(num_nodes)
# print("num_edges:")
# print(num_edges)
# # 遍历节点并打印节点信息
# print("Nodes:")
# for node in graph.nodes:
#     print(node)
#
# # 遍历边并打印边信息
# print("Edges:")
# for u, v, data in graph.edges(data=True):
#     print("u:{}".format(u))
#     print("v:{}".format(v))
#     print("data:{}".format(data))
#
# # 关闭图谱文件
# graph.close()