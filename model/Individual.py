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
    "Add_Object": {"arity": 2, "input_types": ['List[Object]','Object'], 'output_types': 'List[Object]'},
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
    "mirror": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'Object'},
    "place": {"arity": 2, "input_types": ['Object', 'Tuple[int, int]'], 'output_types': 'Object'},
    "exclude_object": {"arity": 2, "input_types": ['List[Object]', 'Object'], 'output_types': 'List[Object]'},
    "Insert_Line_Right": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
    "Insert_Line_Left": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
    "Insert_Line_Top": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
    "Insert_Line_Bottom": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
    "order_position_Right": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Left": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Top": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Bottom": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},

    
}

TERMINALS = ["Object", "h", "v", "max", "min", "left", "right", "top", "bottom", "asc", "desc","coordinates"] + [i for i in range(9)] 
 # Assuming colors are represented by integers 0-8
FUNC_PRIMITIVES  = [concat, Combine, arrange, updateColor, getColor, filter_by_color, filter_by_size, get_First_Object,
                    No_Background, mirror, place,order_position_Right, order_position_Left, order_position_Top, order_position_Bottom,
                    exclude_object, Insert_Line_Right, Insert_Line_Left, Insert_Line_Top, Insert_Line_Bottom]
FILTER_PRIMITIVES = [filter_by_color, filter_by_size, order_position_Right, order_position_Left, order_position_Top, order_position_Bottom,No_Background,get_First_Object]
TRANSFORM_PRIMITIVES = [mirror, place, Insert_Line_Right, Insert_Line_Left, Insert_Line_Top, Insert_Line_Bottom,place]
ENSEMBLE_PRIMITIVES = [concat, Combine, arrange]
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
    def _get_depth(self) -> int:
        # Handle an empty tree first
        if getattr(self, "Root", None) is None:
            return 0

        queue = deque([(self.Root, 0)])   # (node, depth)
        max_depth = 0

        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            # Safely iterate over children if they exist
            for child in getattr(node, "children", ()):
                if child is not None:
                    queue.append((child, depth + 1))

        return max_depth
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
                #print(f"[depth {depth}] '{node.Value}' → {result}")
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
               # print(f"[depth {depth}] {node.Value.__name__}({args[0]}) → {result}")
                return result
            elif len(args) == 2:
                result = node.Value(args[0], args[1])
                #print(f"[depth {depth}] {node.Value.__name__}({args[0]}, {args[1]}) → {result}")
                return result
            else:
                raise ValueError(f"Function node at depth {depth} has invalid number of arguments ({len(args)}): {node.Value.__name__}")

    def get_leaf_nodes(self,node):
            if not node.children:
                return [node]
            leaves = []
            for child in node.children:
                if node.Value not in TERMINALS:
                    leaves.extend(get_leaf_nodes(child))
            return leaves
    
    def Random_Instance(self, max_depth: int = 5, in_Task=None):
        """
        Generates a random instance of the program tree with a maximum depth.
        The root is always an Add_Object node.
        """

       
   
        self.Root = Node(Add_Object)
        self._Add_Node(self.Root, max_depth,in_Task)
        return self.Root

        
        

    def _match_Args(self,function:Callable):
            return DICT_FUNC_WITH_ARGS[function]
    def _Add_Node(self, node, depth, in_Task):
        """
        Recursively expands *node* until *max_depth* reaches 0,
        adapting the primitive set to the (in_Task, out_Task) pair.

        Parameters
        ----------
        node : Node
            Current node in the tree.
        max_depth : int
            Remaining depth budget.
        in_Task / out_Task : user-defined task descriptors
            Provide context for choosing terminals (e.g. extract_object).
        """

        # ── 1. Stop condition ──────────────────────────────────────────────
        if depth <= 0:
            # Turn every open leaf below *node* into a terminal of the right type
            for leaf in self.get_leaf_nodes(node):
                if callable(leaf.Value) and leaf.Value not in TERMINALS:
                    meta        = FUNCTIONS[leaf.Value.__name__]
                    input_types = meta["input_types"]
                    arity       = meta["arity"]

                    for i in range(arity):
                        t = input_types[i]

                        if t == "List[Object]":
                            leaf.children.append(Node("Object"))

                        elif t == "Object":
                            objs = extract_object(in_Task)
                            leaf.children.append(Node(random.choice(objs) if objs else "Object"))

                        elif t == "Tuple[int, int]":
                            leaf.children.append(Node((0, 0)))

                        else:                      # generic-fallback terminal
                            leaf.children.append(Node(random.choice(self._match_Args(leaf.Value))))
            return   # nothing more to do at depth limit
        # ------------------------------------------------------------------

        # ── 2. Expand *this* node if it is a function ──────────────────────
        if not callable(node.Value):     # it’s already a terminal
            return

        meta        = FUNCTIONS[node.Value.__name__]
        input_types = meta["input_types"]
        arity       = meta["arity"]

        possible_func = ENSEMBLE_PRIMITIVES + TRANSFORM_PRIMITIVES + COORD_PRIMITIVES 
        factor = random.random()
        if factor < 0.5:  # 20% chance to use a filter primitive
            possible_func = FUNC_PRIMITIVES
       
        # ── 3. Fill every missing argument slot ────────────────────────────
        while len(node.children) < arity:
            idx        = len(node.children)
            expected_t = input_types[idx]

            # First argument of a filter gets only type-compatible producers
            if idx == 0 and callable(node.Value):
                compatible = [
                    f for f in possible_func
                    if FUNCTIONS[f.__name__]["output_types"] == expected_t
                ]
                child_val = random.choice(compatible) if compatible else "Object"

            else:
                if expected_t == "Object":
                    child_val = random.choice(TRANSFORM_PRIMITIVES + ENSEMBLE_PRIMITIVES)
                elif expected_t == "Tuple[int, int]":
                    child_val = (0, 0)
                else:
                    # last-resort: any terminal known to fit *expected_t*
                    child_val = random.choice(DICT_FUNC_WITH_ARGS[node.Value])

            node.children.append(Node(child_val))

        # ── 4. Recurse on children (depth-1) ───────────────────────────────
        for child in node.children:
            self._Add_Node(child, depth - 1, in_Task)



                
            
       

       






            


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
Grid_dict = {}
with open(r"../data/training/6fa7a44f.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]
Ind = Individual()
Ind.Random_Instance(max_depth=3, in_Task=in_Task)
Ind.Show_Tree().render('tree', format='pdf', cleanup=True)