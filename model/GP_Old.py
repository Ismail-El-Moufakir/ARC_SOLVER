import sys
from collections import deque
import graphviz
import numpy as np
import random
import itertools
sys.path.insert(0,r"../object_representation")
from Dsl import *
import copy 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Individual import *
#___________________UTILS____________________
MODE = [ "CREATE" ,"REFINE"]
def  decide_Mode(in_task,out_Task):
    """
    Decide the mode of the task based on the input and output tasks.
    """
    if in_Task.Shape ==  out_Task.Shape:
        return MODE[0]
    else:
        return MODE[1]
#____________________PARAMETERS____________________
POPPULATION_SIZE = 2000
GENERATIOON_SIZE = 20
CROSSOVER_RATE = 0.7
ELIT_SIZE = 0.1

#_____________________________________Genetic Operations______________________________________
class Genetic_Prog():
    def __init__(self,population_size=3,max_depth = 5):
        self.max_depth = max_depth
        self.scores = []
        self.Population = [Individual() for _ in range(population_size)]
        

        for i in range(len(self.Population)):
            self.Population[i].id = i
            self.Population[i].Random_Instance(max_depth=max_depth)
    def _match_Args(self,function:Callable):
            if function in  DICT_FUNC_WITH_ARGS.keys():
                #get the args of the function
                args = DICT_FUNC_WITH_ARGS[function]
                if len(args) == 0:
                    return ["Object"]
                else:
                    return args
            return ["Object"]
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
            input_type = FUNCTIONS.get(random_node_1.Value.__name__, {}).get("input_types")
            while q2:
                node = q2.popleft()
                if node.Value not in TERMINALS:
                    try:
                        if node.Value.__name__ !=  random_node_1.Value.__name__:
                            output_type = FUNCTIONS.get(node.Value.__name__, {}).get("output_types")
                            if input_type and input_type[0] == output_type:
                                return node
                            q2.extend(node.children)
                    except AttributeError:
                        print(f"Error with node {node.Value}: {node.Value.__name__ if hasattr(node.Value, '__name__') else 'No __name__ attribute'}")
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
    def Mutation(self, parent: "Individual"):
        """
        Perform one type-safe mutation on *parent*.
        The BFS traversal is started only after we reach a randomly
        selected depth in the range [1, self.max_depth].
        Returns the (possibly unchanged) offspring.
        """
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
            if random.random() < 0.3 and node.Value not in TERMINALS:

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
    def Fitness(self, ind: Individual, in_Task, out_Task,
            *,
            w_shape        = 4.0,     # IoU reward
            w_pixel        = 2,     # pixel-wise reward
            w_far_penalty  = 1.0,     # penalty for being “way off”
            w_depth        = 0.3,     # anti-bloat (depth)
            w_nested       = 0.4,     # penalty for nested nodes
            w_novelty      = 0.4,     # exploration boost
            shape_tol      = 0.1,    # IoU error tolerated before penalty
            novelty_archive=None) -> float:
        """
        Compute fitness for an individual.
        Larger = better.
        Adds a structural penalty proportional to the number of *nested* (non-leaf) nodes.
        """

        # ------------------------------------------------------------------ 1) run individual
        try:
            O = ind.execute(in_Task)
        except Exception as e:
            print(f"[run-error] {e}")
            return -400                       # hard fail

        if getattr(O, "Layers", None) is None:
            print("[run-error] output Layers is None")
            return -25                        # soft fail

        try:
            cand_grid = O.construct_grid()
        except Exception as e:
            print(f"[grid-error] {e}")
            return -50                        # soft fail
        print(f"[run-success] individual {ind.Id} executed successfully.")
        # ------------------------------------------------------------------ 2) resize grids
        tgt_grid  = np.array(out_Task).astype(bool)
        cand_grid = np.array(cand_grid).astype(bool)
        
        H = max(tgt_grid.shape[0],  cand_grid.shape[0])
        W = max(tgt_grid.shape[1],  cand_grid.shape[1])

        pad_tgt, pad_cand = (np.zeros((H, W), dtype=bool) for _ in range(2))
        pad_tgt[:tgt_grid.shape[0],  :tgt_grid.shape[1]]  = tgt_grid
        pad_cand[:cand_grid.shape[0], :cand_grid.shape[1]] = cand_grid

        # ------------------------------------------------------------------ 3) shape reward (IoU)
        inter = np.logical_and(pad_tgt, pad_cand).sum()
        union = np.logical_or (pad_tgt, pad_cand).sum() or 1
        iou   = inter / union

        shape_error  = 1.0 - iou
        shape_reward = w_shape * (1.0 - shape_error)
        far_penalty  = w_far_penalty * shape_error * (shape_error > shape_tol)

        # ------------------------------------------------------------------ 4) pixel-by-pixel reward
        pixel_match_ratio = np.mean(pad_tgt == pad_cand)
        pixel_reward      = w_pixel * pixel_match_ratio

        # ------------------------------------------------------------------ 5) depth penalty
        depth_penalty = w_depth * max(ind._get_depth() - self.max_depth, 0)

        # ------------------------------------------------------------------ 6) NEW: nested-node penalty
        def _count_nested_nodes(node) -> int:
            """Return how many nodes in the subtree *node* have at least one child."""
            if not hasattr(node, "children") or not node.children:
                return 0                      # leaf
            # count this node (+1) and recurse on children
            return 1 + sum(_count_nested_nodes(ch) for ch in node.children)

        # Attempt fastest route: ask the individual if it can tell us directly
        try:
            nested_nodes = ind.count_nested_nodes()
        except AttributeError:
            # Fall back to scanning from the root
            root = getattr(ind, "root", None)
            nested_nodes = _count_nested_nodes(root) if root else 0

        nested_penalty = w_nested * nested_nodes

        # ------------------------------------------------------------------ 7) optional novelty bonus
        novelty_bonus = 0.0
        if novelty_archive:
            k = min(5, len(novelty_archive))
            if k:
                flat_cand   = pad_cand.ravel()
                distances   = sorted(
                    np.sum(flat_cand ^ prev.ravel()) / flat_cand.size
                    for prev in novelty_archive
                )[:k]
                novelty_bonus = w_novelty * (sum(distances) / k)

        # ------------------------------------------------------------------ 8) combine terms
        fitness_score = (
            shape_reward         # overlap quality
            + pixel_reward         # exact pixel matches
            + novelty_bonus        # exploration
            - far_penalty          # large-shape mismatch
            - depth_penalty        # over-deep programs
            - nested_penalty       # NEW structural complexity penalty
        )
        return fitness_score
    def evaluate_population(self, in_Task, out_Task):
        """
        Evaluate the fitness of each individual in the population.
        """
        self.scores = []
        i= 0
        for ind in self.Population:
            score = self.Fitness(ind, in_Task, out_Task)
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
            winner = max(tournament, key=lambda ind: self.scores[ind.id])
            selected.append(winner)

        return selected
        
            
    
    
    
    def new_population(self, in_Task, out_Task):
            # ────────────────────────────────
        # 1. Elitism  (always copy top N)
        # ────────────────────────────────
        elite_n = max(1, int(ELIT_SIZE * POPPULATION_SIZE))   # at least one elite
    
        ranked   = sorted(self.Population,
                            key=lambda ind: self.scores[ind.id],
                            reverse=True)
    
        new_pop  = [copy.deepcopy(ind) for ind in ranked[:elite_n]]
        # re order the IDs of the elite individuals
        for i in range(len(new_pop)):
            new_pop[i].id = i
        max_id = len(new_pop) - 1
        #print score of the best 10
        print(f"Best {elite_n} individuals (scores): {[self.scores[ind.id] for ind in new_pop]}")

        # ────────────────────────────────
        # 2. Fill the rest of the pop
        # ────────────────────────────────
        while len(new_pop) < POPPULATION_SIZE:
            max_id += 1
            # Tournament size k in [2 … POP_SIZE//10]  (but never < 2)
            k = random.randint(2, max(2, POPPULATION_SIZE // 10))

            if random.random() < CROSSOVER_RATE:
                # ----- Crossover -----
                # Select two distinct parents via tournament
                parents = self.Tournament_selection(k=k)
                p1, p2  = random.sample(parents, 2)

                # Your Crossover may return 1-or-2 offspring → normalise to tuple
                offspring = self.Crossover(p1, p2)
                if not isinstance(offspring, (list, tuple)):
                    offspring = (offspring,)

                for child in offspring:
                    if child is not None and len(new_pop) < POPPULATION_SIZE:
                        new_pop.append(child)

            else:
                # ----- Mutation -----
                parent     = random.choice(self.Tournament_selection(k=k))
                mutant     = self.Mutation(parent)

                if mutant is not None and len(new_pop) < POPPULATION_SIZE:
                    new_pop.append(mutant)

        # Finally, swap the reference
        self.Population = new_pop

        self.Population = new_pop
    def train(self,in_task,out_task):
         Best_score_over_gen = []
         for generaton in range(GENERATIOON_SIZE):
            self.evaluate_population(in_task, out_task)
            print(f"Generation {generaton + 1}/{GENERATIOON_SIZE} best score: {max(self.scores)} ------------------------------------------------------------")
            max_score = np.max(self.scores)
            Best_score_over_gen.append(max_score)
            self.new_population(in_task, out_task)
         return Best_score_over_gen

            



#example usage
Grid_dict = {}
with open(r"..\data\training\6fa7a44f.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]
Genetic_Prog = Genetic_Prog(population_size=POPPULATION_SIZE, max_depth=3)
Parent_1 = Genetic_Prog.Population[0]

#-------------------------------
Mean_Score_over_generation = Genetic_Prog.train(in_Task, out_Task)
print("Scores over generations:")
#Plotting the mean score over generations
plt.plot(range(1, GENERATIOON_SIZE + 1), Mean_Score_over_generation, marker='o')
plt.title('Max Score Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.show()
Best_Ind= max(Genetic_Prog.Population, key=lambda ind: Genetic_Prog.scores[ind.id])
Best_Ind.Show_Tree().render('Best_Ind', format='pdf', cleanup=True,view = True)
#show output of best individual
in_task = Grid_dict["test"][0]["input"]
O = Best_Ind.execute(in_Task)
viewer.display_grid(O.construct_grid())
print(f"output task {out_Task}")
print(f"output of best individual {O.construct_grid()}")
#----------------------------------

