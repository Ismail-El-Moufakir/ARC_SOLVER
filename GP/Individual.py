import sys
from collections import deque
import graphviz
import numpy as np
import random
sys.path.insert(0,r"../object_representation")
from Dsl import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#____________________________________UTILS______________________________________
def print_objects(objects):
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        print(f"Object Position: {obj.Position}, Color: {obj.color}, Shape: {obj.Shape_Mtx}")
FUNCTIONS = {
    "Add_Object": {"arity": 2, "input_types": ['Object', 'List[Object]'], 'output_types': 'List[Object]'},
    "concat": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'Object'},
    "Combine": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'Object'},
    "arrange": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List["Object"]'},
    "updateColor": {"arity": 2, "input_types": ['Object', 'int'], 'output_types': 'Object'},
    "getColor": {"arity": 1, "input_types": ['Object'], 'output_types': 'int'},
    "filter_by_color": {"arity": 2, "input_types": ['List[Object]', 'int'], 'output_types': 'List[Object]'},
    "filter_by_size": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'Object'},
    "get_First_Object": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'Object'},
    "Top_Left_Coord": {"arity": 1, "input_types": ['Object'], 'output_types': 'Tuple[int, int]'},
    "Top_Right_Coord": {"arity": 1, "input_types": ['Object'], 'output_types': 'Tuple[int, int]'},
    "Bot_Left_Coord": {"arity": 1, "input_types": ['Object'], 'output_types': 'Tuple[int, int]'},
    "Bot_Right_Coord": {"arity": 1, "input_types": ['Object'], 'output_types': 'Tuple[int, int]'},
    "No_Background": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
    "mirror": {"arity": 2, "input_types": ['Object', 'str'], 'output_types': 'Object'},
    "place": {"arity": 2, "input_types": ['Object', 'Tuple[int, int]'], 'output_types': 'Object'},
    "exclude_object": {"arity": 2, "input_types": ['List[Object]', 'Object'], 'output_types': 'List[Object]'},
    "Insert_Line": {"arity": 2, "input_types": ['Object', 'str'], 'output_types': 'Object'},
    "order_position_Right": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Left": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Top": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Bottom": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},

    
}

TERMINALS = ["Object", "h", "v", "max", "min", "left", "right", "top", "bottom", "asc", "desc"] + [i for i in range(9)] 
 # Assuming colors are represented by integers 0-8
FUNC_PRIMITIVES  = [concat, Combine, arrange, updateColor, getColor, filter_by_color, filter_by_size, get_First_Object,
                    No_Background, mirror, place,order_position_Right, order_position_Left, order_position_Top, order_position_Bottom,
                    exclude_object, Insert_Line]

COORD_PRIMITIVES = [Top_Left_Coord, Top_Right_Coord, Bot_Left_Coord, Bot_Right_Coord]
DICT_FUNC_WITH_ARGS = {
    updateColor: range(9),
    filter_by_color: range(9),
    filter_by_size: ["max", "min"],
    exclude_object: ["Object"],
    order_position_Left: ["asc", "desc"],
    order_position_Right: ["asc", "desc"],
    order_position_Top: ["asc", "desc"],
    order_position_Bottom: ["asc", "desc"],
    Insert_Line: ["top", "bottom", "left", "right"],
    place: ["coordinates"],
    mirror: ["h", "v"],
    arrange: range(9),
    Insert_Between: ["Object"],
    updateColor: range(9),

}
#_____________________________________________UTILS___________________________________________

class Individual:
    def __init__(self, root = None):
        self.Root = root
        self.Id = 0

    def Show_Tree(self):
        queue = deque()
        dot = graphviz.Digraph(comment='The Tree Structure')

        if self.Root is None:
            print("The tree is empty.")
            return dot

        def add_node_to_graph(node):
            if isinstance(node.Value, str):
                label = f'Terminal: {node.Value}'
            elif node.is_function():
                label = f'Function: {node.Value.__name__}'
            elif isinstance(node.Value, int):
                label = f'color: {node.Value}'
            else:
                label = f'Unknown: {node.Value}'
            dot.node(str(id(node)), label)

        queue.append(self.Root)
        add_node_to_graph(self.Root)

        while queue:
            current = queue.popleft()

            for i, child in enumerate(current.children):
                if child:
                    add_node_to_graph(child)
                    dot.edge(str(id(current)), str(id(child)), label=f'Child {i+1}')
                    queue.append(child)
    
        return dot
    def execute(self, in_Task):
        return self._execute_node(self.Root, in_Task)
    def _execute_node(self, node, in_Task, depth=0):
        # ── terminal-string nodes ───────────────────────────────
        if isinstance(node.Value, str):
            handlers = {
                "Object": lambda _: extract_object(in_Task),
                "h":       lambda _: "h",
                "v":       lambda _: "v",
                "max":     lambda _: "max",
                "min":     lambda _: "min",
                "left":    lambda _: "left",
                "right":   lambda _: "right",
                "top":     lambda _: "top",
                "bottom":  lambda _: "bottom",
                "asc":     lambda _: "asc",
                "desc":    lambda _: "desc",
            }
            if node.Value in handlers:
                result = handlers[node.Value](in_Task)
                print(f"[depth {depth}] '{node.Value}' → {result}")
                return result
            raise ValueError(f"Unknown terminal string '{node.Value}'")

        # ── callable nodes ──────────────────────────────────────
        if node.is_function():
            # Evaluate all children
            args = [self._execute_node(child, in_Task, depth + 1) for child in node.children]
            
            if len(args) == 0:
                raise ValueError(f"Function node at depth {depth} has no arguments: {node.Value.__name__}")
            elif len(args) == 1:
                result = node.Value(args[0])
                print(f"[depth {depth}] {node.Value.__name__}({args[0]}) → {result}")
                return result
            elif len(args) == 2:
                result = node.Value(args[0], args[1])
                print(f"[depth {depth}] {node.Value.__name__}({args[0]}, {args[1]}) → {result}")
                return result
            else:
                raise ValueError(f"Function node at depth {depth} has invalid number of arguments ({len(args)}): {node.Value.__name__}")

        
    def Random_Instance(self, max_depth: int = 5):
        """
        Generates a random instance of the program tree with a maximum depth.
        The root is always an Add_Object node.
        """

        def get_leaf_nodes(node):
            if not node.children:
                return [node]
            leaves = []
            for child in node.children:
                if node.Value not in TERMINALS:
                    leaves.extend(get_leaf_nodes(child))
            return leaves

   
        self.Root = Node(Add_Object)
        self._Add_Node(self.Root, max_depth)

        leaves = get_leaf_nodes(self.Root)
        print(f"Leaf nodes found: {len(leaves)}")

    
        for leaf in leaves:
                if leaf.Value not in TERMINALS:
                    # two case leaf that needs object of leaf that need a object function
                    input_types = FUNCTIONS.get(leaf.Value.__name__, {}).get("input_types", [])
                    arity = FUNCTIONS.get(leaf.Value.__name__, {}).get("arity", 0)
                    if input_types[0] == 'List[Object]':
                        leaf.children.append(Node("Object"))
                        if arity > 1:
                            if input_types[1] == 'Object':
                                child = Node(random.choice([func for func in FUNC_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == 'Object']))
                                leaf.children.append(child)
                                leaves.append(child)
                            else:
                                #complete with terminal arguments
                                leaf.children.append(Node(DICT_FUNC_WITH_ARGS[leaf.Value][0]))
                    elif input_types[0] == 'Object':
                         leaf.children.append(Node(random.choice([func for func in FUNC_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == 'Object'])))
                         leaves.append(leaf.children[0])
                         if len(input_types) > 1:
                                if input_types[1] == 'Object':
                                    leaf.children.append(Node(random.choice([func for func in FUNC_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == 'Object'])))
                                    leaves.append(leaf.children[1])
                                elif input_types[1] == 'Tuple[int, int]':
                                        child = Node(random.choice(COORD_PRIMITIVES))
                                        leaf.children.append(child)
                                        leaves.append(child)

                                else:
                                    #complete with terminal arguments
                                        leaf.children.append(Node(DICT_FUNC_WITH_ARGS[leaf.Value][0]))
                         

                   

        return self.Root

        
        

    def _match_Args(self,function:Callable):
            return DICT_FUNC_WITH_ARGS[function]
    def _Add_Node(self, node, max_depth):
        
        # Check if max_depth is reached or node is a terminal (arity 0)
        if max_depth <= 0:
            # Only set to "object" if the node's type matches the parent node's required argument type
            if hasattr(node.Value, '__name__') and node.Value.__name__ == 'Object':
                node.Value = "Object"  # Set to terminal value only if type matches
            return

        # Get the arity and input types of the current node
        if hasattr(node.Value, '__name__'):
            node_arity = FUNCTIONS.get(node.Value.__name__, {}).get("arity", 0)
            input_types = FUNCTIONS.get(node.Value.__name__, {}).get("input_types", [])

            # Ensure input_types matches the arity
            if len(input_types) != node_arity:
                raise ValueError(f"Input types length {len(input_types)} does not match arity {node_arity} for {node.Value.__name__}")

            # Handle children based on arity and input types
            for arg_idx, arg_type in enumerate(input_types):
                possible_functions = []
                if node.Value == Add_Object:
                    possible_functions  = FUNC_PRIMITIVES
                elif arg_type == 'List[Object]'or arg_type == 'Object':
                    possible_functions = [func for func in FUNC_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == arg_type]
                    print(f"Possible functions for {arg_type}: {[func.__name__ for func in possible_functions]}")
                    np.random.shuffle(possible_functions)  
                elif arg_type == 'Tuple[int, int]':
                    possible_functions = [func for func in COORD_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == arg_type]
                    print(f"Possible functions for {arg_type}: {[func.__name__ for func in possible_functions]}")
                    np.random.shuffle(possible_functions)
                else:
                    possible_args = self._match_Args(node.Value)
                    if not possible_args:
                        raise ValueError(f"No matching arguments found for {node.Value.__name__} at index {arg_idx}")
                    elif possible_args == "Object":
                        node.childeren.append(random.choice([func for func in FUNC_PRIMITIVES if FUNCTIONS.get(func.__name__, {}).get("output_types") == "Object"]))
                    elif possible_args == "coordinates":
                        node.children.append(Node(random.choice(COORD_PRIMITIVES)))
                    else:
                        node.children.append(Node(random.choice(possible_args)))
                        continue

                if not possible_functions:
                    raise ValueError(f"No suitable functions found for type {arg_type} at index {arg_idx}")
                node.children.append(Node(random.choice(possible_functions)))

            # Recursive call to add children nodes
            for child in node.children:
                self._Add_Node(child, max_depth - 1)




                
            
       

       






            


class Node:
    def __init__(self, value, left=None, right=None):
        self.Value = value
        self.children = []
    def is_function(self) -> bool:
        return callable(self.Value)
    

# Build the 6fa7a44f program tree 
"""leaf_1 = Node("objects")
Concat_Node_1 = Node(concat, leaf_1, None)
mirror_Node_1 = Node(mirror, Concat_Node_1, Node("h"))
leaf_2 = Node("objects")
Concat_Node_2 = Node(concat, leaf_2, None)
mirror_Node_2 = Node(mirror, Concat_Node_2, Node("h"))
Bot_Coord_Node = Node(Bot_Left_Coord, mirror_Node_1, None)
place_Node = Node(place,mirror_Node_2, Bot_Coord_Node)
leaf_3 = Node("objects")
Concat_Node_3 = Node(concat, leaf_3, None)
O = Add_Object_Node = Node(Add_Object, place_Node, Concat_Node_3)"""
#---------------------------------------------------
#
#Build the 1f642eb9 program tree 
'''No_Background_Node_1 = Node(No_Background, Node("objects"), None)
concatenate_Node = Node(concat, No_Background_Node_1, None)
Insert_Line_Node_1 = Node(Insert_Line, concatenate_Node, Node("top"))
Insert_Line_Node_2 = Node(Insert_Line, Insert_Line_Node_1, Node("bottom"))
Insert_Line_Node_3 = Node(Insert_Line, Insert_Line_Node_2, Node("left"))
Insert_Line_Node_4 = Node(Insert_Line, Insert_Line_Node_3, Node("right"))
No_Background_Node_2 = Node(No_Background, Node("objects"), None)
No_Background_Node_3 = Node(No_Background, Node("objects"), None)
filter_by_size_Node = Node(filter_by_size, No_Background_Node_2, Node("max"))
exlude_Object_Node = Node(exclude_object, No_Background_Node_3, filter_by_size_Node)
O = Add_Object_Node = Node(Add_Object, exlude_Object_Node,Insert_Line_Node_4 )
'''

