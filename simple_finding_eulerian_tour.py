
# coding: utf-8

# In[1]:

def next_node(edge, current):
    return edge[0] if current == edge[1] else edge[1]

def remove_edge(raw_list, discard):
    return [item for item in raw_list if item != discard]

def find_eulerian_tour(graph):
    search = [[[], graph[0][0], graph]]
    while search:
        path, node, unexplore = search.pop()
        path += [node]

        if not unexplore:
            return path

        for edge in unexplore:
            if node in edge:
                search += [[path, next_node(edge, node), remove_edge(unexplore, edge)]]

if __name__ == '__main__':
    graph = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 3)]
    print find_eulerian_tour(graph)

