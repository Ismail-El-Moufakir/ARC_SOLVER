from object import Object, Grid
from Dsl import place
import json

class Graph:
    def __init__(self,in_Grid,out_Grid):
        self.in_Grid = Grid(len(in_Grid), len(in_Grid[0]))
        self.in_Grid.extract_Objects(in_Grid)
        self.in_Grid.Shape_Mtx()
        self.out_Grid = Grid(len(out_Grid), len(out_Grid[0]))
        self.out_Grid.extract_Objects(out_Grid)
        self.out_Grid.Shape_Mtx()
        self.nodes = self.in_Grid.Layers 
        self.edges = []
        self.grid_size = (len(in_Grid), len(in_Grid[0]))
        
    def compute_edges(self):
            #compute edges that represent relative positions between objects
            for i in range(len(self.nodes)):
                for j in range(i + 1, len(self.nodes)):
                    obj1 = self.nodes[i]
                    obj2 = self.nodes[j]
                    # Calculate relative positions
                    dx = obj2.Position[0] - obj1.Position[0]
                    dy = obj2.Position[1] - obj1.Position[1]
                    
                    # If objects are aligned horizontally or vertically
                    if dx == 0:  # Vertical alignment
                        self.edges.append((obj1.id, obj2.id, "V", abs(dy)))
                    elif dy == 0:  # Horizontal alignment
                        self.edges.append((obj1.id, obj2.id, "H", abs(dx)))
    
    def print_node_details(self):
        print("\n=== Node Details ===")
        for node in self.nodes:
            print(f"\nNode ID: {node.id}")
            print(f"Position: {node.Position}")
            print(f"Color: {node.color}")
            print("Shape Matrix:")
            print(node.Shape_Mtx)
            print("Shape Coordinates:", node.Shape_Coords)
            print("-" * 30)

Grid_dict = {}
with open(r"..\data\training\6fa7a44f.json", 'r') as file:
        Grid_dict = json.load(file)       
in_grid,out_grid = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]  
graph = Graph(in_grid, out_grid)
graph.compute_edges()

print("\n=== Graph Summary ===")
print("Nodes:", len(graph.nodes))
print("Edges:", len(graph.edges))

print("\n=== Edge Details ===")
for edge in graph.edges:
    print(f"Edge from {edge[0]} to {edge[1]} is {edge[2]} with distance {edge[3]}")

# Print detailed node information
graph.print_node_details()           
