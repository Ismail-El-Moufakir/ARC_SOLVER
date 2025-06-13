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
#____________________PARAMETERS____________________
POPPULATION_SIZE = 3
GENERATIOON_SIZE = 20
MULATION_RATE = 0.
CROSSOVER_RATE = 0.7

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
            return DICT_FUNC_WITH_ARGS[function]
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
        print(f"cut node selected from Parent 1: {random_node_1.Value.__name__}")

        # Find compatible subtree to crossover
        def get_compatible_node(Parent_2, random_node_1):
            q2 = deque(Parent_2.Root.children)
            input_type = FUNCTIONS.get(random_node_1.Value.__name__, {}).get("input_types")
            while q2:
                node = q2.popleft()
                if node.Value not in TERMINALS:
                    output_type = FUNCTIONS.get(node.Value.__name__, {}).get("output_types")
                    if input_type and input_type[0] == output_type:
                        return node
                    q2.extend(node.children)
            return None

        random_node_2 = get_compatible_node(Parent_2, random_node_1)
        print(f"donar node selected from Parent 2: {random_node_2.Value.__name__ if random_node_2 else 'None'}")

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
                    f for f in FUNC_PRIMITIVES
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

                        print(f"Mutation applied: replaced {old_fun.__name__} "
                            f"with {cand.__name__} at depth {depth}")
                        return child        # one mutation per call

            # keep scanning deeper nodes
            queue.extend((c, depth + 1) for c in node.children)

        # no suitable mutation found
        return child
    def Fitness(self,ind: Individual,in_Task,out_Task):
         try:
             O = ind.execute(in_Task)
         except Exception as e:
             print(f"Error executing program: {e}")
             return -50   #penalty for invalid program
         #program executed suceessfully
         try:
            if O.Layers is None:
                        print("Output is None")
                        return -50
            out_Grid = O.construct_grid()
         except Exception as e:
             print(f"Error constructing grid: {e}")
             return -50   #penalty for invalid output
         score = 0
             #comapre predcted output and out_task shape
         r_Out, c_Out = np.array(out_Grid).shape
         r_Task, c_Task = np.array(out_Task).shape
         if r_Out == r_Task and c_Out == c_Task:
            for i in range(r_Out):
                for j in range(c_Out):
                        if out_Grid[i][j] == out_Task[i][j] and out_Task[i][j] != 0:
                            score += 1
            return score
         else:
                 #if output shape is not same as task shape, return a penalty score
                 print("Output shape does not match task shape")
                 return -25
    def evaluate_population(self, in_Task, out_Task):
        """
        Evaluate the fitness of each individual in the population.
        """
        self.scores = []
        for ind in self.Population:
            score = self.Fitness(ind, in_Task, out_Task)
            self.scores.append(score)
            print(f"Individual {ind.Id} score: {score}")
        return self.scores
    def stochastic_universal_sampling(self, k: int = 2):
        """
        Select k individuals from the population using Stochastic Universal Sampling (SUS).

        Parameters
        ----------
        k : int
            Number of individuals to select.

        Returns
        -------
        list
            The k selected individuals (with replacement possible if their
            cumulative fitness spans several pointer positions).
        """
        # -------- 1. Build fitness distribution --------
        fitnesses = [self.scores[ind.id] for ind in self.Population]
        total_fit  = sum(fitnesses)

        # Guard-rail: if every fitness is zero (or they all cancel out),
        # fall back to uniform sampling.
        if total_fit == 0:
            return random.sample(self.Population, k)

        probs   = [f / total_fit for f in fitnesses]         # normalised fitness
        cumsum  = list(itertools.accumulate(probs))           # CDF

        # -------- 2. Create equally-spaced pointers --------
        step     = 1.0 / k
        start    = random.uniform(0.0, step)                  # first pointer in [0, step)
        pointers = [start + i * step for i in range(k)]

        # -------- 3. Walk the CDF to pick parents --------
        selected = []
        idx = 0
        for p in pointers:
            while p > cumsum[idx]:
                idx += 1
            selected.append(self.Population[idx])

        return selected
    
           
    
    
    
    def new_population(self, in_Task, out_Task):
        """
        Create a new population using crossover and mutation.
        """
        new_pop = []
        while len(new_pop) < POPPULATION_SIZE:
            if random.random() < CROSSOVER_RATE:
                selected_parents = self.stochastic_universal_sampling()
                # Select two parents for crossover
                parent_1,parent_2 = random.sample(selected_parents, 2)
                child = self.Crossover(parent_1, parent_2)
                if child is not None:
                    new_pop.append(child)
            else:
                # Select one parent for mutation
                selected_parents = self.stochastic_universal_sampling()
                parent = random.choice(selected_parents)
                child = self.Mutation(parent)
                new_pop.append(child)
        self.Population = new_pop

        self.Population = new_pop
    def train(self,in_task,out_task):
         Best_Scores_over_generations = []
         for generaton in range(GENERATIOON_SIZE):
            self.evaluate_population(in_task, out_task)
            print(f"Generation {generaton + 1}/{GENERATIOON_SIZE} best score: {max(self.scores)} ------------------------------------------------------------")
            Best_Score = max(self.scores)
            Best_Scores_over_generations.append(Best_Score)
            self.new_population(in_task, out_task)
         return Best_Scores_over_generations

            



#example usage
Grid_dict = {}
with open(r"..\data\training\1f642eb9.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]
Genetic_Prog = Genetic_Prog(population_size=POPPULATION_SIZE, max_depth=2)
Best_Score_over_generation = Genetic_Prog.train(in_Task, out_Task)
in_Task,out_Task = Grid_dict["train"][1]["input"], Grid_dict["train"][1]["output"]
Best_Score_over_generation = Genetic_Prog.train(in_Task, out_Task)
in_Task,out_Task = Grid_dict["train"][2]["input"], Grid_dict["train"][2]["output"]
Best_Score_over_generation = Genetic_Prog.train(in_Task, out_Task)

#-------------------------------

print("Scores over generations:")
for i, score in enumerate(Best_Score_over_generation):
    print(f"Generation {i + 1}: {score}")
Best_Ind= max(Genetic_Prog.Population, key=lambda ind: Genetic_Prog.scores[ind.id])
Best_Ind.Show_Tree().render('Best_Ind', format='pdf', cleanup=True,view = True)
#show output of best individual
in_task = Grid_dict["test"][0]["input"]
O = Best_Ind.execute(in_Task)
viewer.display_grid(O.construct_grid())
print(f"output task {out_Task}")
print(f"output of best individual {O.construct_grid()}")
#----------------------------------

'''O = parent_1.execute(in_Task)
out_Grid = O.construct_grid()
viewer.display_grid(out_Grid)'''


'''parent_2 = Genetic_Prog.Population[1]
 
child  =  Genetic_Prog.Mutation(parent_1)
child.Show_Tree().render('child', format='pdf', cleanup=True,view = True)
'''

