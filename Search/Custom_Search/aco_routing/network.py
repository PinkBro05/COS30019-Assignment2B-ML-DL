class Network:
    def __init__(self):
        self.graph = {}  # Stores adjacency list: node -> list of neighbors
        self.edges = {}  # Stores edge attributes: (u,v) -> attribute dict
        self.pos = {} # Stores node positions for visualization

    def add_edge(self, u, v, **attr):
        """Add an edge between u and v with optional attributes."""
        # Ensure nodes exist in the graph
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
            
        # Add v to u's neighbors if not already present
        if v not in self.graph[u]:
            self.graph[u].append(v)
            
        # Calculate Euclidean distance if positions are available
        if u in self.pos and v in self.pos:
            # Extract coordinates
            pos_u = self.pos[u]
            pos_v = self.pos[v]
            
            # Calculate Euclidean distance
            distance = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
            
            # Add distance as an attribute
            attr['distance'] = round(distance, 2)
            
        # Store edge attributes
        self.edges[(u, v)] = attr

    def number_of_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self.graph)

    def number_of_edges(self):
        """Return the number of edges in the graph."""
        return len(self.edges)

    def nodes(self):
        """Return an iterator over the graph nodes."""
        return self.graph.keys()

    def neighbors(self, node):
        """Return a list of the node's neighbors."""
        return self.graph.get(node, [])

    def get_edge_data(self, u, v):
        """Return the attribute dictionary associated with edge (u,v)."""
        return self.edges.get((u, v), {})

    def has_edge(self, u, v):
        """Return True if the edge (u,v) is in the graph."""
        return (u, v) in self.edges
    
    def get_edge_attributes(self, attribute_name):
        """Returns a dictionary mapping edge tuples to attribute values."""
        return {edge: data.get(attribute_name, None) 
                for edge, data in self.edges.items() 
                if attribute_name in data}
                
    def get_edges(self, data=False):
        """Returns a list of edges, optionally with data."""
        if data:
            return [(u, v, self.edges[u, v]) for u, v in self.edges]
        else:
            return list(self.edges.keys())
            
    def __getitem__(self, node):
        """Allow dictionary-like access to node attributes."""
        if node in self.graph:
            return NodeDict(self, node)
        raise KeyError(f"Node {node} not in graph")

class NodeDict:
    def __init__(self, network, node):
        self.network = network
        self.node = node
        
    def __getitem__(self, neighbor):
        """Allow access to edge attributes like: G[node1][node2]['attribute']"""
        if neighbor in self.network.neighbors(self.node):
            return self.network.edges.get((self.node, neighbor), {})
        raise KeyError(f"No edge from {self.node} to {neighbor}")