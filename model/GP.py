import sys
from collections import deque
from typing import Callable

import numpy as np
import random

sys.path.insert(0,r"../object_representation")
from Dsl import *
import copy 
from Individual import *
#____________________PARAMETERS____________________
POPPULATION_SIZE = 500
OFFSPRING_SIZE = 1000
GENERATIOON_SIZE = 10
CROSSOVER_RATE = 0.7

ELIT_RATE = 0.3
ELIT_SIZE = 0.05
#______________________________________UTILS________________________________________
Ops_FACTOR = 0.2
def get_leaf_nodes(node):
            if not node.children:
                return [node]
            leaves = []
            for child in node.children:
                if node.Value not in TERMINALS:
                    leaves.extend(get_leaf_nodes(child))
            return leaves


class Genetic_Prog:
    def __init__(self,population_size = 5, Generation_Count = 5,max_depth = 3):
        self.population_size = population_size
        self.max_depth = max_depth
        self.Population = []
        for i in range(self.population_size):
            ind = Individual()
            ind.Random_Instance(max_depth = self.max_depth)
            ind.Id = i
            self.Population.append(ind)
    def _match_Args(self,function:Callable):
            if function in  DICT_FUNC_WITH_ARGS.keys():
                #get the args of the function
                args = DICT_FUNC_WITH_ARGS[function]
                if len(args) == 0:
                    return ["Object"]
                else:
                    return args
            return ["Object"]

    def SubTree_Mutate(self, Parent: Individual):

        def _build_subtree(node: Node, depth: int):
            # get function-node meta-data
            input_Types = FUNCTIONS[node.Value.__name__]["input_types"]
            output_Type = FUNCTIONS[node.Value.__name__]["output_types"]

            # choose the replacement primitive
            if random.random() > Ops_FACTOR:
                Ops_PRIMITIVES = ENSEMBLE_PRIMITIVES + TRANSFORM_PRIMITIVES
                pool = Ops_PRIMITIVES
            else:
                pool = FUNC_PRIMITIVES

            compatibles = [
                f for f in pool
                if FUNCTIONS[f.__name__]["input_types"][0] == input_Types[0]
                and FUNCTIONS[f.__name__]["output_types"] == output_Type
            ]

            if len(compatibles) == 0:        # nothing fits → keep old subtree
                return

            node.Value = random.choice(compatibles)
            node.children = []
            print("compatible ops are:", [f.__name__ for f in compatibles])

            # complete the subtree
            q: deque[tuple["Node", int]] = deque([(node, depth)])
            random_depth = random.randint(min(depth + 1, self.max_depth), self.max_depth)

            while q:
                current, c_depth = q.popleft()

                if current.Value in TERMINALS:
                    continue

                input_Types = FUNCTIONS[current.Value.__name__]["input_types"]
                arity = FUNCTIONS[current.Value.__name__]["arity"]

                if c_depth < random_depth:
                    # first (mandatory) child: function whose output matches current’s first input
                    func_choices = [
                        f for f in FUNC_PRIMITIVES
                        if input_Types[0] == FUNCTIONS[f.__name__]["output_types"]
                    ]
                    first_child = Node(random.choice(func_choices))
                    current.children.append(first_child)
                    q.append((first_child, c_depth + 1))

                    # remaining args, if any
                    if arity > 1:
                        possible_args = self._match_Args(current.Value)      # ← fixed
                        if possible_args and possible_args[0] == "coordinates":
                            coord_child = Node(random.choice(COORD_PRIMITIVES))
                            current.children.append(coord_child)
                            q.append((coord_child, c_depth + 1))
                        else:
                            arg_child = Node(random.choice(possible_args))
                            current.children.append(arg_child)
                            q.append((arg_child, c_depth + 1))
        def Complete_tree(Ind: Individual):
            #check if the tree is good and complete with args if it's necessary      
            Leafs = get_leaf_nodes(Ind.Root)
            print("Leafs are:", [leaf.Value for leaf in Leafs])
            for leaf in Leafs:
                if not callable(leaf.Value):
                    continue
                else:
                    input_Types = FUNCTIONS[leaf.Value.__name__]["input_types"]
                    arity = FUNCTIONS[leaf.Value.__name__]["arity"]
                    if input_Types[0] == "List[Object]":
                       leaf.children.append(Node("Object"))
                       if arity > 1:
                        args = self._match_Args(leaf.Value)
                        leaf.children.append(Node(random.choice(args)))
                    elif input_Types[0] == "Object":
                        px_Obj = Object((0.0),(0.0),1,0)
                        leaf.children.append(Node(px_Obj))
                        if arity > 1:
                            args = self._match_Args(leaf.Value)
                            if args and args[0] == "coordinates":
                                leaf.children.append(Node((0, 0)))
                            else:
                             leaf.children.append(Node(random.choice(args)))

                        

                    
                    

        # choose a random depth at which to start the mutation
        random_depth = random.randint(1, self.max_depth)
        Child = copy.deepcopy(Parent)

        queue: deque[tuple["Node", int]] = deque([(Child.Root, 0)])
        while queue:
            current, depth = queue.popleft()

            if depth >= random_depth:
                if random.random() >= 0.5 and callable(current.Value):
                    print(f"Mutating node {current.Value.__name__} at depth {depth}")
                    _build_subtree(current, depth)
                    Complete_tree(Child)
                    return Child
                    

            queue.extend((child, depth + 1) for child in current.children)

        return Child
    def Node_Mutation(self, parent: Individual):
        '''Performs a mutation on a random node in the parent individual.'''
        child = copy.deepcopy(parent)
        #random depth to start mutatuion process
        target_depth = random.randint(1, self.max_depth)

        #queue for BFS with Node and its depth
        queue = deque([(child.Root, 0)])

        while queue:
            node, depth = queue.popleft()

            # skip nodes that are shallower than the target depth
            if depth < target_depth:
                
                queue.extend((c, depth + 1) for c in node.children)
                continue

            # start random mutationn process
            if random.random() < 0.3 and callable(node.Value):

                out_type = FUNCTIONS.get(node.Value.__name__, {}).get("output_types")
                compatibles = [
                    f for f in ENSEMBLE_PRIMITIVES + TRANSFORM_PRIMITIVES
                    if f is not node.Value
                    and FUNCTIONS.get(f.__name__, {}).get("output_types") == out_type
                ]
                np.random.shuffle(compatibles)          # random candidate order

                node_in_types = FUNCTIONS.get(node.Value.__name__, {}).get("input_types")

                for cand in compatibles:
                    cand_in_types = FUNCTIONS.get(cand.__name__, {}).get("input_types")

                    if cand_in_types == node_in_types:
                        old_fun = node.Value
                        node.Value = cand

                        # re-wire children if arity > 1
                        arity = FUNCTIONS.get(cand.__name__, {}).get("arity", 0)
                        if arity > 1:
                            new_args = self._match_Args(cand)
                            node.children = [node.children[0]] + [Node(random.choice(new_args))]

                        #print(f"Mutation applied: replaced {old_fun.__name__} "
                         #   f"with {cand.__name__} at depth {depth}")
                        return child        # one mutation per call

            # keep scanning deeper nodes
            queue.extend((c, depth + 1) for c in node.children)

        # no suitable mutation found
        return child
    def Crossover(self, Parent_1: Individual, Parent_2: Individual):
        Child = Individual()
        # Select a random sub tree and a random cut node that are compatible
        def get_random_node(node):
            q = deque(node.children)
            while q:
                c = q.popleft()
                factor = random.random()
                if factor < 0.5 and c.Value not in TERMINALS:
                    return c
                else:
                    q.extend(c.children)
            return None  # In case no suitable node is found

        random_node_1 = get_random_node(Parent_1.Root)
        if random_node_1 is None:
            return None
       # print(f"cut node selected from Parent 1: {random_node_1.Value.__name__}")

        # Find compatible subtree to crossover
        def get_compatible_node(Parent_2, random_node_1):
            q2 = deque(Parent_2.Root.children)
            #print(f"Searching for compatible node in Parent 2 for {random_node_1.Value}")
            if callable(random_node_1.Value):
                input_type = FUNCTIONS.get(random_node_1.Value.__name__, {}).get("input_types")
                while q2:
                    node = q2.popleft()
                    if  callable(node.Value):
                        try:
                            if node.Value.__name__ !=  random_node_1.Value.__name__:
                                output_type = FUNCTIONS.get(node.Value.__name__, {}).get("output_types")
                                if input_type and input_type[0] == output_type:
                                    return node
                                q2.extend(node.children)
                        except AttributeError:
                            print(f"Error with node {node.Value}: {node.Value.__name__ if hasattr(node.Value, '__name__') else 'No __name__ attribute'}")
                return None
            return None
        

        random_node_2 = get_compatible_node(Parent_2, random_node_1)
        #print(f"donar node selected from Parent 2: {random_node_2.Value.__name__ if random_node_2 else 'None'}")

        if random_node_2 is None:
            return None
        else:
            # Deepcopy Parent_1 into Child
            Child = copy.deepcopy(Parent_1)

            # Find matching node in Child (equivalent to random_node_1 in Parent_1)
            def find_corresponding_node(orig_root, clone_root, target_node):
                q1 = deque([orig_root])
                q2 = deque([clone_root])
                while q1:
                    orig = q1.popleft()
                    clone = q2.popleft()
                    if orig is target_node:
                        return clone
                    q1.extend(orig.children)
                    q2.extend(clone.children)
                return None

            target_in_child = find_corresponding_node(Parent_1.Root, Child.Root, random_node_1)
            if target_in_child:
                target_in_child.Value = copy.deepcopy(random_node_2.Value)
                target_in_child.children = copy.deepcopy(random_node_2.children)


            return Child
    def Fitness(self,
    ind,                       # ← Individual instance
    in_task,                   # ← input task
    out_task,                  # ← expected output task
    *,
    # ---------- pixel-wise scoring -------------------------------------------
    empty_val: int      = 0,
    tp_reward: float    = 2,
    fp_penalty: float   = 1,
    fn_penalty: float   = 1,
    shape_weight: float = 10,    # penalty multiplier for shape mismatch
    # ---------- structural rewards / penalties -------------------------------
    novelty_weight: float = 20, # reward factor for novelty (positive bonus)
    depth_weight: float   = 0.2,  # penalty per level beyond max_depth
    max_depth: int       = 20,  # depth allowed before penalties kick in
    nested_weight: float = 2, # penalty per internal (non-leaf) node
    # ---------- housekeeping --------------------------------------------------
    invalid_score: int  = -10000
) -> float:
        """
        Fitness with pixel accuracy, shape penalty, novelty reward,
        depth penalty and nested-node penalty.
        """
        # ---------------------------------------------------------------- 1) run
        try:
            O = ind.execute(in_task)
        except Exception as e:
            print(f"[fitness] Execution error: {e}")
            return invalid_score

        # ---------------------------------------------------------------- 2) grid
        try:
            if getattr(O, "Layers", None) is None:
                print("[fitness] Output has no Layers")
                return invalid_score
            pred_grid = np.asarray(O.construct_grid())
        except Exception as e:
            print(f"[fitness] Grid construction error: {e}")
            return invalid_score

        target_grid = np.asarray(out_task)
        print(f"[fitness] sucessfuly individual {ind.Id} executed and constructed grid")
        # ---------------------------------------------------------------- 3) shape
        d_rows = abs(pred_grid.shape[0] - target_grid.shape[0])
        d_cols = abs(pred_grid.shape[1] - target_grid.shape[1])
        shape_penalty = -shape_weight * (d_rows + d_cols)

        # overlapping window
        n_rows = min(pred_grid.shape[0], target_grid.shape[0])
        n_cols = min(pred_grid.shape[1], target_grid.shape[1])
        sub_pred   = pred_grid[:n_rows, :n_cols]
        sub_target = target_grid[:n_rows, :n_cols]

        # ---------------------------------------------------------------- 4) pixel
        tp = np.sum((sub_pred == sub_target) & (sub_target != empty_val))
        fp = np.sum((sub_pred != sub_target) & (sub_target == empty_val))
        fn = np.sum((sub_pred != sub_target) & (sub_target != empty_val))
        pixel_score = tp_reward * tp - fp_penalty * fp - fn_penalty * fn

        # ---------------------------------------------------------------- 5) depth
        depth = getattr(ind, "_get_depth", lambda: 0)()
        depth_penalty = -depth_weight * max(depth - max_depth, 0)

        # ---------------------------------------------------------------- 6) nested-node
        def _count_nested_nodes(node) -> int:
            """Return how many nodes in the subtree *node* have at least one child."""
            if not hasattr(node, "children") or not node.children:
                return 0
            return 1 + sum(_count_nested_nodes(ch) for ch in node.children)

        try:
            nested_nodes = ind.count_nested_nodes()       # fast path (user-provided)
        except AttributeError:
            root = getattr(ind, "root", None)
            nested_nodes = _count_nested_nodes(root) if root else 0

        nested_penalty = -nested_weight * nested_nodes

        # ---------------------------------------------------------------- 7) novelty
        novelty = getattr(ind, "novelty", 0)   # user can attach any novelty metric
        novelty_reward = novelty_weight * novelty

        # ---------------------------------------------------------------- 8) final
        return (
            pixel_score            # accuracy
            + shape_penalty        # global shape
            + depth_penalty        # shallower is better
            + nested_penalty       # simpler is better
            + novelty_reward       # encourage exploration
        )
    def evaluate_population(self, in_Task, out_Task):
        """
        Evaluate the fitness of each individual in the population.
        """
        self.scores = []
        i= 0
        for ind in self.Population:
            score = self.Fitness(ind, in_Task, out_Task,max_depth=self.max_depth)
            ind.Id = i
            self.scores.append(score)
            i+= 1
            #print(f"Individual {ind.Id} score: {score}")       
    def Tournament_selection(self, k: int = 2, tournament_size: int = 3):
        """
        Perform tournament selection to choose k individuals.

        Parameters
        ----------
        k : int
            Number of individuals to select.
        tournament_size : int
            Number of individuals in each tournament.

        Returns
        -------
        list
            A list of k selected individuals.
        """
        selected = []
        pop = list(self.Population)
        
        for _ in range(k):
            tournament = random.sample(pop, min(tournament_size, len(pop)))
            winner = max(tournament, key=lambda ind: self.scores[ind.Id])
            selected.append(winner)

        return selected   
    def new_population(self,
                    in_Task, out_Task,
                    OFFSPRING_SIZE: int | None = None,
                    ELITISM: float | int = 0.10):
        """
        (μ + λ) replacement **with explicit elitism**

        • μ  == current population size         (POPPULATION_SIZE)
        • λ  == OFFSPRING_SIZE (defaults to μ)
        • ELITISM
            – float in (0,1] → fraction of μ
            – int  ≥1        → absolute number
            – at least one elite is always preserved
        """
        μ = POPPULATION_SIZE
        λ = OFFSPRING_SIZE or μ

        # ---------------------------------------------------------------- A) rank parents & copy the elites
        # make sure every parent has a fitness score first
        for ind in self.Population:
            if ind.Id not in self.scores:
                self.scores[ind.Id] = self.Fitness(ind, in_Task, out_Task)

        e = (max(1, round(μ * ELITISM))            # fraction
            if isinstance(ELITISM, float) else
            max(1, min(int(ELITISM), μ)))         # absolute count

        ranked_parents = sorted(self.Population,
                                key=lambda ind: self.scores[ind.Id],
                                reverse=True)

        elites = [copy.deepcopy(ind) for ind in ranked_parents[:e]]
        for ind in elites:                         # optional marker
            ind.is_elite = True

        # ---------------------------------------------------------------- B) create λ offspring
        offspring: list[Individual] = []
        while len(offspring) < λ:
            k = random.randint(2, max(2, μ // 10))            # tournament size
            p1, p2 = random.sample(self.Tournament_selection(k=k), 2)

            children = self.Crossover(p1, p2)
            if not isinstance(children, (list, tuple)):
                children = (children,)

            for child in children:
                if child is None or len(offspring) >= λ:
                    continue
                child = (self.Node_Mutation(child) if random.random() < 0.5
                        else self.SubTree_Mutate(child))
                offspring.append(child)

        # ---------------------------------------------------------------- C) evaluate offspring
        for ind in offspring:
            self.scores[ind.Id] = self.Fitness(ind, in_Task, out_Task)

        # ---------------------------------------------------------------- D) build the next generation
        #  – elites are *always* kept
        #  – remaining μ-e slots go to the best of (non-elite parents ∪ offspring)
        candidates = ranked_parents[e:] + offspring
        candidates.sort(key=lambda ind: self.scores[ind.Id], reverse=True)

        survivors = elites + candidates[:μ - e]               # exact size μ

        # neat, dense Ids again
        for new_id, ind in enumerate(survivors):
            ind.Id = new_id

        self.Population = survivors

        # (optional) diagnostics
        best10 = [self.scores[ind.Id] for ind in survivors[:10]]
        print(f"Top 10 scores after elitist (μ+λ): {best10}")

        return survivors
    def train(self,in_task,out_task):
         Best_score_over_gen = []
         for generaton in range(GENERATIOON_SIZE):
            self.evaluate_population(in_task, out_task)
            print(f"Generation {generaton + 1}/{GENERATIOON_SIZE} best score: {max(self.scores)} ------------------------------------------------------------")
            max_score = np.max(self.scores)
            Best_score_over_gen.append(max_score)
            self.new_population(in_task, out_task, OFFSPRING_SIZE=OFFSPRING_SIZE)
         return Best_score_over_gen


'''#example usage
Grid_dict = {}
with open(r"..\data\training\6fa7a44f.json", 'r') as file:
        Grid_dict = json.load(file)       

Genetic_Prog = Genetic_Prog(population_size=POPPULATION_SIZE, max_depth=3)
Parent_1 = Genetic_Prog.Population[0]

#-------------------------------
# first task
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]
Mean_Score_over_generation = Genetic_Prog.train(in_Task, out_Task)
#second task
in_Task,out_Task = Grid_dict["train"][1]["input"], Grid_dict["train"][1]["output"]
Mean_Score_over_generation += Genetic_Prog.train(in_Task, out_Task)
#third task
in_Task,out_Task = Grid_dict["train"][2]["input"], Grid_dict["train"][2]["output"]
Mean_Score_over_generation += Genetic_Prog.train(in_Task, out_Task)
print("Scores over generations:")
#Plotting the mean score over generations
plt.plot(range(1,len(Mean_Score_over_generation)+1), Mean_Score_over_generation, marker='o')
plt.title('Max Score Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.show()
Best_Ind= max(Genetic_Prog.Population, key=lambda ind: Genetic_Prog.scores[ind.Id])
Best_Ind.Show_Tree().render('Best_Ind', format='pdf', cleanup=True,view = True)
#show output of best individual
in_task = Grid_dict["test"][0]["input"]
O = Best_Ind.execute(in_Task)
viewer.display_grid(O.construct_grid())
print(f"output task {out_Task}")
print(f"output of best individual {O.construct_grid()}")
#----------------------------------
'''


            

            


        

