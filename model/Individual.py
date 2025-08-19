import sys
from collections import deque
import graphviz
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, r"../object_representation")
from Dsl import *

# ____________________________________UTILS______________________________________

def print_objects(objects):
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        print(f"Object Position: {obj.Position}, Color: {obj.color}, Shape: {obj.Shape_Mtx}")


# ---------------------------- DSL SPECIFICATION --------------------------------

FUNCTIONS = {
    # Set arity to -1 to indicate **variadic** (1 list + n objects)
    "Add_Object": {"arity": -1, "input_types": ['List[Object]', 'Object'], 'output_types': 'List[Object]'},
    "concat": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'Object'},
    "Combine": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'Object'},
    "arrange": {"arity": 1, "input_types": ['List[Object]'], 'output_types': 'List[Object]'},
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
    "Insert_Line_Right": {"arity": 1, "input_types": ['Object'], 'output_types': 'List[Object]'},
    "Insert_Line_Left": {"arity": 1, "input_types": ['Object'], 'output_types': 'List[Object]'},
    "Insert_Line_Top": {"arity": 1, "input_types": ['Object'], 'output_types': 'List[Object]'},
    "Insert_Line_Bottom": {"arity": 1, "input_types": ['Object'], 'output_types': 'List[Object]'},
    "order_position_Right": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Left": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Top": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
    "order_position_Bottom": {"arity": 2, "input_types": ['List[Object]', 'str'], 'output_types': 'List[Object]'},
}

TERMINALS = [
    "Object",
    "h",
    "v",
    "max",
    "min",
    "left",
    "right",
    "top",
    "bottom",
    "asc",
    "desc",
    "coordinates",
] + [i for i in range(9)]  # colors 0–8

# Primitive groupings (unchanged variable names)
FUNC_PRIMITIVES = [
    concat,
    Combine,
    arrange,
    updateColor,
    getColor,
    filter_by_color,
    filter_by_size,
    get_First_Object,
    No_Background,
    mirror,
    place,
    order_position_Right,
    order_position_Left,
    order_position_Top,
    order_position_Bottom,
    exclude_object,
    Insert_Line_Right,
    Insert_Line_Left,
    Insert_Line_Top,
    Insert_Line_Bottom,
]
FILTER_PRIMITIVES = [
    filter_by_color,
    filter_by_size,
    order_position_Right,
    order_position_Left,
    order_position_Top,
    order_position_Bottom,
    No_Background,
    get_First_Object,
]
TRANSFORM_PRIMITIVES = [
    mirror,
    place,
    Insert_Line_Right,
    Insert_Line_Left,
    Insert_Line_Top,
    Insert_Line_Bottom,
]
ENSEMBLE_PRIMITIVES = [concat, Combine, arrange]
COORD_PRIMITIVES = [Top_Left_Coord, Top_Right_Coord, Bot_Left_Coord, Bot_Right_Coord]

# Map of functions that require constant/terminal arguments
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
    # Insert_Between removed – not defined in snippet
}

# ___________________________________________UTILS___________________________________________


class Node:
    def __init__(self, value):
        self.Value = value
        self.children = []

    def is_function(self) -> bool:
        return callable(self.Value)


class Individual:
    def __init__(self, root=None):
        self.Root = root
        self.Id = 0
        self.score = 0

    # -------------------------- Tree-helper methods -----------------------------

    def _get_depth(self) -> int:
        if self.Root is None:
            return 0
        queue = deque([(self.Root, 0)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            for child in getattr(node, "children", ()):  # safe iteration
                if child is not None:
                    queue.append((child, depth + 1))
        return max_depth

    def Show_Tree(self):
        """Return a Graphviz Digraph of the current tree."""
        dot = graphviz.Digraph(comment='The Tree Structure')
        if self.Root is None:
            print("The tree is empty.")
            return dot

        def add_node(n):
            if isinstance(n.Value, str):
                label = f'Terminal: {n.Value}'
            elif n.is_function():
                label = f'Function: {n.Value.__name__}'
            elif isinstance(n.Value, int):
                label = f'color: {n.Value}'
            else:
                label = f'Unknown: {n.Value}'
            dot.node(str(id(n)), label)

        queue = deque([self.Root])
        add_node(self.Root)
        while queue:
            current = queue.popleft()
            for i, child in enumerate(current.children):
                if child:
                    add_node(child)
                    dot.edge(str(id(current)), str(id(child)), label=f'Child {i + 1}')
                    queue.append(child)
        return dot

    # -------------------------- Execution --------------------------------------

    def execute(self, in_Task):
        return self._execute_node(self.Root, in_Task)

    def _execute_node(self, node, in_Task, depth=0):
        # ----- Terminal strings -------------------------------------------------
        if isinstance(node.Value, str):
            handlers = {
                "Object": lambda _: extract_object(in_Task),
                "h": lambda _: "h",
                "v": lambda _: "v",
                "max": lambda _: "max",
                "min": lambda _: "min",
                "left": lambda _: "left",
                "right": lambda _: "right",
                "top": lambda _: "top",
                "bottom": lambda _: "bottom",
                "asc": lambda _: "asc",
                "desc": lambda _: "desc",
                "coordinates": lambda _: (0, 0),
            }
            if node.Value in handlers:
                return handlers[node.Value](in_Task)
            raise ValueError(f"Unknown terminal string '{node.Value}'")

        # ----- Function nodes ---------------------------------------------------
        if node.is_function():
            args = [self._execute_node(child, in_Task, depth + 1) for child in node.children]

            # Custom handling for variadic Add_Object
            if node.Value == Add_Object:
                if len(args) < 2:
                    raise ValueError("Add_Object requires at least a list and one object")
                result = args[0]
                for obj in args[1:]:
                    result = Add_Object(result, obj)
                return result

            # Default: unary or binary functions as before
            if len(args) == 1:
                return node.Value(args[0])
            elif len(args) == 2:
                return node.Value(args[0], args[1])
            else:
                raise ValueError(
                    f"Function {node.Value.__name__} received {len(args)} arguments; only unary/binary supported other than Add_Object."
                )

    # -------------------------- Tree generation --------------------------------

    def get_leaf_nodes(self, node):
        if not node.children:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.get_leaf_nodes(child))
        return leaves

    def Random_Instance(self, max_depth: int = 5, in_Task=None):
        """Generate a random tree whose root is a variadic Add_Object."""
        self.Root = Node(Add_Object)
        self._Add_Node(self.Root, max_depth, in_Task, is_root=True)
        return self.Root

    def _match_Args(self, function: callable):
        return DICT_FUNC_WITH_ARGS[function]

    def _Add_Node(self, node, depth, in_Task, is_root=False):
        """Recursively expand *node* until *depth* reaches 0."""
        # -- 1. Stop condition ---------------------------------------------------
        if depth <= 0:
            for leaf in self.get_leaf_nodes(node):
                if callable(leaf.Value) and leaf.Value not in TERMINALS:
                    meta = FUNCTIONS[leaf.Value.__name__]
                    input_types = meta["input_types"]
                    arity = meta["arity"] if meta["arity"] != -1 else 2  # at least two
                    for i in range(arity):
                        t = input_types[0] if i == 0 else input_types[-1]
                        if t == "List[Object]":
                            leaf.children.append(Node("Object"))
                        elif t == "Object":
                            objs = extract_object(in_Task)
                            leaf.children.append(Node(random.choice(objs) if objs else "Object"))
                        elif t == "Tuple[int, int]":
                            leaf.children.append(Node((0, 0)))
                        else:
                            leaf.children.append(Node(random.choice(self._match_Args(leaf.Value))))
            return

        # -- 2. Expand if function ----------------------------------------------
        if not callable(node.Value):  # already terminal
            return

        meta = FUNCTIONS[node.Value.__name__]
        input_types = meta["input_types"]
        arity_spec = meta["arity"]

        # Determine desired arity
        if arity_spec == -1:  # variadic Add_Object
            desired_arity = random.randint(1, 4) if is_root else 2  # only root gets extras
        else:
            desired_arity = arity_spec

        # Choose function pool based on context
        possible_func = ENSEMBLE_PRIMITIVES + TRANSFORM_PRIMITIVES + COORD_PRIMITIVES
        if random.random() < 0.5:
            possible_func = FUNC_PRIMITIVES

        # -- 3. Fill argument slots ---------------------------------------------
        while len(node.children) < desired_arity:
            idx = len(node.children)
            # Variadic Add_Object: first arg list, others objects
            if arity_spec == -1:
                expected_t = input_types[0] if idx == 0 else 'Object'
            else:
                expected_t = input_types[idx]

            # Special handling for first argument of filters (unchanged)
            if idx == 0 and callable(node.Value):
                compatible = [
                    f for f in possible_func if FUNCTIONS[f.__name__]["output_types"] == expected_t
                ]
                child_val = random.choice(compatible) if compatible else "Object"
            else:
                if expected_t == "Object":
                    child_val = random.choice(TRANSFORM_PRIMITIVES + ENSEMBLE_PRIMITIVES)
                elif expected_t == "Tuple[int, int]":
                    child_val = (0, 0)
                else:
                    child_val = random.choice(DICT_FUNC_WITH_ARGS.get(node.Value, ["Object"]))

            node.children.append(Node(child_val))

        # -- 4. Recurse ----------------------------------------------------------
        for child in node.children:
            self._Add_Node(child, depth - 1, in_Task)


# ---------------------------- Example usage ------------------------------------

if __name__ == "__main__":
    Grid_dict = {}
    with open(r"../data/training/1f642eb9.json", "r") as file:
        Grid_dict = json.load(file)

    in_Task = Grid_dict["train"][0]["input"]
    Ind = Individual()
    Ind.Random_Instance(max_depth=3, in_Task=in_Task)
    Ind.Show_Tree().render("tree", format="pdf", cleanup=True)
