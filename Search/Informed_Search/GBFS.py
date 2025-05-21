import matplotlib.pyplot as plt
import numpy as np
import heapq
import re
import sys, os

# Import parser to read the graph file
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "data_reader"))
from parser import parse_graph_file

# Import Network class from custom search directory
aco_routing_dir = os.path.join(current_dir, "..", "Custom_Search", "aco_routing")
sys.path.append(aco_routing_dir)
from network import Network

class Node:
    def __init__(self, start, heuristic):
        self.start = start
        self.heuristic = heuristic

    def __lt__(self,  other):
        return self.heuristic < other.heuristic

def find_heuristic(pos, current, goal):
    goal_x, goal_y = int(pos[goal][0]), int(pos[goal][1])
    current_x, current_y = int(pos[current][0]), int(pos[current][1])

    return math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)

def find_next_node(graph, current, visited_list, pos, goal):
    heuristic_value = []

    if graph[current] == []:
        return "Dead end"
    else:
        # print("Current node:", current)
        for i in graph[current]:
            if i not in visited_list:
                heuristic = find_heuristic(pos, i, goal)
                heapq.heappush(heuristic_value, Node(i, heuristic))
        
        if not heuristic_value:
            return "Loop"
        # print("Heuristic value:", heuristic_value)
        return heapq.heappop(heuristic_value)

def GBFS_search(graph, start, goal, heuristic):
    # path dictionary to track the explored paths
    path = {start:None}

    visited = set() # to keep track of visited nodes

    # Priority queue to hold nodes to explore, sorted by heuristic value
    priority_queue = []
    first = Node(start, 0)
    heapq.heappush(priority_queue, first)

    while priority_queue:
        current_node = heapq.heappop(priority_queue).start
        
        visited.add(current_node)

        # if the goal is reached, reconstruct the path
        if goal in visited:
            return reconstruct_path(path, start, goal)

        # find next node
        next_node = find_next_node(graph, current_node, visited, pos, goal)
        if next_node == "Dead end":
            print("Dead end at node ", current_node)
            # Go back to the previous node
            # and check for other paths
            heapq.heappush(priority_queue, Node(path[current_node], find_heuristic(pos, path[current_node], goal)))
        elif next_node == "Loop":
            count = 0
            print("Loop at node ", current_node)
            # Go back to the previous node
            # and check for other paths
            heapq.heappush(priority_queue, Node(path[current_node], find_heuristic(pos, path[current_node], goal)))
            if count == 5:
                print("Loop detected at node ", current_node)
                return reconstruct_path(path, start, current_node)
            count += 1
        else:
            heapq.heappush(priority_queue, next_node)

            if next_node.start not in path:
                path[next_node.start] = current_node
        
    return None

def reconstruct_path(path, start, goal):
    current = goal
    result_path = []

    while current is not None:
        result_path.append(current)
        current = path[current]
        
    result_path.reverse()
    return result_path


def visualise(paths, pos, edges):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_xticks(range(11))
    ax1.set_yticks(range(11))

    ax2.set_xticks(range(11))
    ax2.set_yticks(range(11))

    ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax2.grid(True, which='both', linestyle='-', linewidth=0.5)

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # plot the edge
    for edge in edges.keys():
        p1, p2 = edge

        x_values = [pos[p1][0], pos[p2][0]]
        y_values = [pos[p1][1], pos[p2][1]]

        ax1.plot(x_values, y_values, marker='o', color='blue')
        ax2.plot(x_values, y_values, marker='o', color='blue')

    # Annotate the points
    for point, coord in pos.items():
        ax1.text(coord[0], coord[1], point, fontsize=12, color='black', ha='right')
        ax2.text(coord[0], coord[1], point, fontsize=12, color='black', ha='right')
    
    xpoints = []
    ypoints = []
    for i in range(len(paths[0])):
        xpoints.append(pos[paths[0][i]][0])
        ypoints.append(pos[paths[0][i]][1])

    ax1.plot(xpoints, ypoints, marker='o', color='red')

    xpoints = []
    ypoints = []
    for i in range(len(paths[1])):
        xpoints.append(pos[paths[1][i]][0])
        ypoints.append(pos[paths[1][i]][1])

    ax2.plot(xpoints, ypoints, marker='o', color='lightgreen')

    plt.show()

def main():
    # Check if file path is provided as command line argument
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        # Default file for testing
        file_path = os.path.join("Data", "Modified_TSP", "test_4.txt")
    
    # Parse the file
    nodes, edges, origin, destinations = parse_graph_file(file_path)
    
    # Create Data Structure    
    G = Network()
    G.graph = {node: [] for node in nodes}
    
    # Add edges 
    for (start, end), weight in edges.items():
        G.add_edge(start, end, cost=float(weight))
    # print(G.graph)
    
    result_paths = []
    path_weights = []

    for dest in destinations:
        # print("Starting search from ", origin, " to ", dest)
        weight = 0
        result_path = GBFS_search(G.graph, origin, dest, edges)
        if result_path[-1] != dest:
            print("Path not found")
        else:
            print(result_path)

        for i in range(len(result_path)-1):
            weight += edges[(result_path[i], result_path[i+1])]
        
        # print(f"Path weight: {weight}\n")
        
        path_weights.append(weight)
        result_paths.append(result_path)
    
    if not result_paths:
        print("No paths found")
    else:
        # Pick the shortest path
        min_weight = min(path_weights)
        min_index = path_weights.index(min_weight)
        
        # Print the results
        print(f"{file_path} GBFS")
        print(f"{destinations} {len(nodes)}")
        print(f"{result_paths[min_index]}")
        print(f"{min_weight}")
    
    # Optionally visualize
    # visualise(result_paths, nodes, edges)

# Example usage:
if __name__ == "__main__":
    main()
