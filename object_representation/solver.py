from object import Object,Grid
from typing import List, Tuple, Dict
from Dsl import *
import json
import viewer
def solve_6fa7a44f(in_Task):
    objects = extract_object(in_Task)
    objec_Concatinated = concat(objects)
    objec_Mirrored = mirror(objec_Concatinated, 'h')
    Bot_Cordinates_Of_Object = Bot_Left_Coord(objec_Mirrored)
    objec_Placed = place(objec_Mirrored,Bot_Cordinates_Of_Object)
    O = Add_Object(objec_Concatinated, objec_Placed)
    return O
def solve_681b3aeb(in_Task):
    objects = extract_object(in_Task)
    no_Background = No_Background(objects)
    concatinated_Object = Combine(no_Background)
    O =  Add_Object(concatinated_Object)
    return O
def solve_1f642eb9(in_Task):
  
    objects = extract_object(in_Task)
    no_back = No_Background(objects)
    concatinated_Object = concat(no_back)
    Inserted_obj = Insert_Line(concatinated_Object, 'top')
    Inserted_obj = Insert_Line(Inserted_obj, 'bottom')
    Inserted_obj = Insert_Line(Inserted_obj, 'left')
    Inserted_obj = Insert_Line(Inserted_obj, 'right')
    container = filter_by_size(no_back,"max")
    no_back= exclude_object(no_back, container)
    O = Add_Object(Inserted_obj, *no_back)
    return O
def solve_cdecee7f(in_Task):
    objects = extract_object(in_Task)
    no_Background = No_Background(objects)
    Objets_Ordered = order_by_position(no_Background)
    Objets_Ordered  = arrange(Objets_Ordered)
    O = Add_Object(Objets_Ordered)
        
    
    return O
def solve_a699fb00(in_Task):
    objects = extract_object(in_Task)
    no_Background = No_Background(objects)
    
    commun_obj = get_First_Object(no_Background)  
    color = getColor(commun_obj)
    commun_obj = updateColor(commun_obj, 6)
    for obj in no_Background:
        print(f"Object at {obj.Position} with color {obj.color} and shape matrix:\n{obj.Shape_Mtx}")
    no_Background = filter_by_color(no_Background, color)
    
    Inserted_obj = Insert_Between(no_Background, commun_obj)
    O = Add_Object(Inserted_obj, commun_obj)

        
    
    return O

Grid_dict = {}
with open(r"..\data\training\cdecee7f.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["test"][0]["input"], Grid_dict["train"][0]["output"] 
in_Grid = Grid()
O = solve_cdecee7f(in_Task)

viewer.display_grid(O.construct_grid())
for obj in O.Layers:
    print(f"Object at {obj.Position} with color {obj.color} and shape matrix:\n{obj.Shape_Mtx}")



