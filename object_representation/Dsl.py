from object import Object, Grid
import json
import viewer as viewer
import copy  
from typing import List, Tuple, Dict, Callable
import numpy as np
import math

# ─────────────────────────────────────────── helper utilities ──────────
def _nonzero_cells(mtx: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return (row, col, colour) for every coloured (non-zero) cell."""
    rows, cols = mtx.nonzero()
    return [(r, c, int(mtx[r, c])) for r, c in zip(rows, cols)]

def _update_bounds(min_r, min_c, max_r, max_c, coords):
    """Expand bounding box to encompass coords."""
    rs, cs = zip(*coords)
    return (min(min_r, min(rs)), min(min_c, min(cs)),
            max(max_r, max(rs)), max(max_c, max(cs)))

# ─────────────────────────────────────────── main routine ───────────────

#-----Object Representation Functions-----#

# colors
def updateColor(Obj:Object,Color):
     '''
     color an object with a specfic color
     '''
     new_Obj = Object(
          Position= Obj.Position,
          Shape_Coords= Obj.Shape_Coords,
          Color= Color,
          id= Obj.id
     )
     new_Obj.Shape_Mtx = np.zeros(np.shape(Obj.Shape_Mtx))
     for i in range(len(Obj.Shape_Mtx)):
         for j in range(len(Obj.Shape_Mtx[0])):
             if Obj.Shape_Mtx[i][j]!=0:
                 new_Obj.Shape_Mtx[i][j] = Color
     return new_Obj
def getColor(Obj:Object) -> int:
        '''
        get the color of an object
        '''
        return Obj.color
#filters 
def filter_by_color(set:List[Object],color:int) -> List[Object]:
     '''
     filter object by color

     '''
     if not set:
            raise ValueError("The set is empty.")
     for obj in set:
        print(f"Object at {obj.Position} with color {obj.color} and shape matrix:\n{obj.Shape_Mtx}")
     return [obj for obj in set if obj.color == color] 
def filter_by_size(Set:List[Object],arg:str) -> Object:
     if not Set:
            raise ValueError("The set is empty.")
     if arg not in ['max', 'min']:
            raise ValueError("Argument must be 'max' or 'min'.")
     if arg == 'max':
            return max(Set, key=lambda obj: np.sum(obj.Shape_Mtx != 0))
     elif arg == 'min':
            return min(Set, key=lambda obj: np.sum(obj.Shape_Mtx != 0))
def order_by_position(Set: List[Object], order: str = 'asc',Direction:str = "left") -> List[Object]:
     if not Set:
            raise ValueError("The set is empty.")
     if order not in ['asc', 'desc']:
            raise ValueError("Order must be 'asc' or 'desc'.")
     if Direction not in ['left', 'right', 'top', 'bottom']:
            raise ValueError("Direction must be 'left', 'right', 'top', or 'bottom'.")
     if Direction == 'left':
            return sorted(Set, key=lambda obj: obj.Position[1], reverse=(order == 'desc'))
     elif Direction == 'right':
            return sorted(Set, key=lambda obj: obj.Position[1] + obj.Shape_Mtx.shape[1], reverse=(order == 'desc'))
     elif Direction == 'top':
            return sorted(Set, key=lambda obj: obj.Position[0], reverse=(order == 'desc'))
     elif Direction == 'bottom':
            return sorted(Set, key=lambda obj: obj.Position[0] + obj.Shape_Mtx.shape[0], reverse=(order == 'desc'))
def order_position_Left(Set: List[Object], order: str = 'asc') -> List[Object]:
    '''
    Order objects by their left position.
    '''
    if not Set:
        raise ValueError("The set is empty.")
    if order not in ['asc', 'desc']:
        raise ValueError("Order must be 'asc' or 'desc'.")
    return sorted(Set, key=lambda obj: obj.Position[1], reverse=(order == 'desc'))
def order_position_Right(Set: List[Object], order: str = 'asc') -> List[Object]:
    '''
    Order objects by their right position.
    '''
    if not Set:
        raise ValueError("The set is empty.")
    if order not in ['asc', 'desc']:
        raise ValueError("Order must be 'asc' or 'desc'.")
    return sorted(Set, key=lambda obj: obj.Position[1] + obj.Shape_Mtx.shape[1], reverse=(order == 'desc'))
def order_position_Top(Set: List[Object], order: str = 'asc') -> List[Object]:
    '''
    Order objects by their top position.
    '''
    if not Set:
        raise ValueError("The set is empty.")
    if order not in ['asc', 'desc']:
        raise ValueError("Order must be 'asc' or 'desc'.")
    return sorted(Set, key=lambda obj: obj.Position[0], reverse=(order == 'desc'))
def order_position_Bottom(Set: List[Object], order: str = 'asc') -> List[Object]:
    '''
    Order objects by their bottom position.
    '''
    if not Set:
        raise ValueError("The set is empty.")
    if order not in ['asc', 'desc']:
        raise ValueError("Order must be 'asc' or 'desc'.")
    return sorted(Set, key=lambda obj: obj.Position[0] + obj.Shape_Mtx.shape[0], reverse=(order == 'desc'))
  



def get_First_Object(Set: List[Object]) -> Object:
    '''
    Get the first object in a set.
    '''
    if not Set:
        raise ValueError("The set is empty.")
    return Set[0]     

def Top_Left_Coord(obj: Object):
    '''
    Get the top-left coordinates of an object.
    '''
    return obj.Position
def Top_Right_Coord(obj: Object):
    '''
    Get the top-right coordinates of an object.
    '''
    return (obj.Position[0], obj.Position[1] + obj.Shape_Mtx.shape[1])
def Bot_Left_Coord(obj: Object):
    '''
    Get the bottom-left coordinates of an object.
    '''
    return (obj.Position[0] + obj.Shape_Mtx.shape[0], obj.Position[1])
def Bot_Right_Coord(obj: Object):
    '''
    Get the bottom-right coordinates of an object.
    '''
    return (obj.Position[0] + obj.Shape_Mtx.shape[0],
            obj.Position[1] + obj.Shape_Mtx.shape[1])
def No_Background(Set: list[Object]):
     '''
     Remove background from a set of objects.
     '''
     new_Set = []
     for obj in Set:
          if obj.color != 0:  # Assuming 0 is the background color
               new_Set.append(obj)
     return new_Set   
def exclude_object(objects: List["Object"], target: "Object") -> List["Object"]:
    """
    Return every object in `objects` except `target`.
    Two objects are considered identical when BOTH
    their positions and their Shape_Mtx dimensions match.
    """
    if not objects:
        raise ValueError("The set is empty.")
    if not target:
        raise ValueError("No target object provided.")
    return [
        o for o in objects
        if not (
            o.Position == target.Position and
            getattr(o.Shape_Mtx, "shape", None) == getattr(target.Shape_Mtx, "shape", None)
        )
    ]
          
#transformation
def mirror(obj:Object, axis:str):
     '''
     Mirror an object along a specified axis ('h' for horizontal, 'v' for vertical).
     --> obj
     '''
     new_Shape_Mtx = []
     if axis not in ['h', 'v']:
          raise ValueError("Axis must be 'h' for horizontal or 'v' for vertical.")
     if axis == 'h':
          new_Shape_Mtx = np.flipud(obj.Shape_Mtx)
     elif axis == 'v':
          new_Shape_Mtx = np.fliplr(obj.Shape_Mtx)
     new_obj = Object(
          Position=obj.Position,
          Shape_Coords=obj.Shape_Coords,
          Color=obj.color,
          id=obj.id
     )
     new_obj.Shape_Mtx = new_Shape_Mtx
     return new_obj
def place(obj:Object, Position):
     '''
     Place an object at a specified position.
     --> obj, Position
     '''
     new_obj = Object(
          Position=Position,
          Shape_Coords=obj.Shape_Coords,
          Color=obj.color,
          id=obj.id
     )
     #update the coordinates of the shape matrix
     new_Shape_Coords = []
     for i in range(len(obj.Shape_Mtx)):
          for j in range(len(obj.Shape_Mtx[0])):
               if obj.Shape_Mtx[i][j] != 0:
                    new_Shape_Coords = [(i+Position[0], j+Position[1]) for i, j in obj.Shape_Coords]
     
     new_obj.Shape_Coords = new_Shape_Coords
     new_obj.Shape_Mtx = obj.Shape_Mtx.copy()
     return new_obj
def Insert_Between(objs: List["Object"],
                   spacer: "Object",
                   spacing: int = 0) -> List["Object"]:
    """
    Return a new list in which a *clone* of `spacer` is physically placed
    between every consecutive pair of objects in `objs`.

    Parameters
    ----------
    objs : list[Object]
        Must be ordered in the reading direction (left-to-right).  The
        originals are left untouched.
    spacer : Object
        The object to insert.  Each insertion is a *deep copy* so later
        edits do not interfere with one another.
    spacing : int, default 0
        Extra empty pixels to leave between objects horizontally.

    Returns
    -------
    list[Object]
        `[obj₀, spacer, obj₁, spacer, …, objₙ]`, with each spacer’s
        `Position` chosen to sit exactly between its neighbours.
    """
    if not objs:
        raise ValueError("The set is empty.")
    if spacer is None:
        raise ValueError("No spacer object provided.")

    # ------------------------------------------------------------------
    # Make deep copies so nothing in the returned list shares memory
    # with the caller’s objects.
    # ------------------------------------------------------------------
    originals = [copy.deepcopy(o) for o in objs]
    template  = copy.deepcopy(spacer)

    result: List["Object"] = []

    for i, left_obj in enumerate(originals):
        # Always append the (cloned) original first
        result.append(left_obj)

        # If there is a "right neighbour", insert a new spacer
        if i < len(originals) - 1:
            right_obj = originals[i + 1]
            new_spacer = copy.deepcopy(template)

            # ----------------- Horizontal placement ------------------
            left_edge   = left_obj.Position[1]
            left_width  = left_obj.Shape_Mtx.shape[1]
            spacer_col  = left_edge + left_width + spacing

            # ----------------- Vertical placement -------------------
            # vertical centres of the neighbours
            l_cy = left_obj.Position[0] + left_obj.Shape_Mtx.shape[0] / 2
            r_cy = right_obj.Position[0] + right_obj.Shape_Mtx.shape[0] / 2
            mid_cy = (l_cy + r_cy) / 2

            spacer_height = new_spacer.Shape_Mtx.shape[0]
            spacer_row = int(round(mid_cy - spacer_height / 2))
            spacer_row = max(spacer_row, 0)             # clamp to top

            # ----------------- Apply position -----------------------
            new_spacer.Position = (spacer_row, spacer_col)

            result.append(new_spacer)

    return result





#ensembling 
def concat(Set: list[Object]):
    '''
    Concatenate a set of objects into a single object.
    '''
    if not Set:
        raise ValueError("The set is empty.")

    dic_coord_color = {}
    for obj in Set:
         #dict for coord assigning colors
         for coor in obj.Shape_Coords:
              dic_coord_color[coor] = obj.color
    new_set_Coord = set(dic_coord_color.keys())
    #Reconstruct Mtx shape
    Shape_Mtx = []
    Max_i, Max_j = max(new_set_Coord,key = lambda x : x[0])[0],max(new_set_Coord,key = lambda x : x[1])[1] 
    Min_i, Min_j = min(new_set_Coord,key = lambda x : x[0])[0],min(new_set_Coord,key = lambda x : x[1])[1] 
    r,c = abs(Max_i - Min_i)+1,abs(Max_j - Min_j)+1
    Mtx = np.zeros((r,c))
    for coord in new_set_Coord:
          Mtx[coord[0] - Min_i][coord[1] - Min_j] = dic_coord_color[coord]
     #return new object
    new_obj = Object(
          Position=(Min_i, Min_j),
          Shape_Coords=list(new_set_Coord),
          Color=-1,  # -1 indicates a concatenated object
          id=0  # Assuming a new ID for the concatenated object
     )
  
    Mtx = Mtx[np.any(Mtx != 0, axis=1)]  # Remove empty rows
    Mtx = Mtx[:, np.any(Mtx != 0, axis=0)]  # Remove empty columns
    #update Position
    diff_r = r - Mtx.shape[0]
    diff_c = c - Mtx.shape[1]
    new_obj.Position = (Min_i + (diff_r // 2)-1, Min_j + (diff_c // 2)-1)

    new_obj.Shape_Mtx = Mtx
    return new_obj
def Combine(objs: List["Object"]) -> "Object":
    """
    Combine several pattern objects into ONE by *re-positioning* them
    (just integer translations) so that their coloured cells never
    collide.  Example: a reverse-U frame and an I-bar can be merged
    by sliding the I into the U’s cavity.

    • Each object's `Shape_Mtx` is its authoritative pattern.
    • Its initial `Position` is only the *starting* hint; during
      packing we are free to slide it.
    • Search strategy: brute-force integer offsets within a window
      large enough to cover all patterns, picking the placement that
      yields the **smallest area** so far.
    • Complexity is fine for ARC-style grids (≤ 30×30), but you can
      tighten `SEARCH_PAD` for larger scenes.
    """
    if not objs:
        raise ValueError("Combine requires at least one object.")


    # ---------------------------------------------------------------
    # 1. Pre-extract every object's coloured cells
    # ---------------------------------------------------------------
    obj_cells = []
    obj_shapes = []
    for obj in objs:
        cells = _nonzero_cells(obj.Shape_Mtx)
        if not cells:
            continue
        obj_cells.append(cells)             # list[(r, c, colour)]
        obj_shapes.append(obj.Shape_Mtx.shape)

    # ---------------------------------------------------------------
    # 2. Greedy placement: anchor the first object, then fit others
    # ---------------------------------------------------------------
    placed: Dict[Tuple[int, int], int] = {}     # world (r, c) ➜ colour
    min_r = min_c = +10**9
    max_r = max_c = -10**9

    # ── anchor #0 at its current Position ──────────────────────────
    base_obj = objs[0]
    base_off_r, base_off_c = base_obj.Position
    for r, c, clr in obj_cells[0]:
        world = (r + base_off_r, c + base_off_c)
        placed[world] = clr
    min_r, min_c, max_r, max_c = _update_bounds(
        +10**9, +10**9, -10**9, -10**9, placed.keys()
    )

    # ── pack the remaining objects one-by-one ──────────────────────
    SEARCH_PAD = 30  # how far (in cells) we are willing to slide
    for cells, (h, w) in zip(obj_cells[1:], obj_shapes[1:]):
        best_offset = None
        best_area = None

        # search window large enough to slide pattern around base
        for dr in range(-SEARCH_PAD, SEARCH_PAD + 1):
            for dc in range(-SEARCH_PAD, SEARCH_PAD + 1):
                conflict = False
                new_coords = []

                for r, c, clr in cells:
                    wr, wc = r + dr + base_off_r, c + dc + base_off_c
                    if (wr, wc) in placed and placed[(wr, wc)] != clr:
                        conflict = True
                        break
                    new_coords.append((wr, wc))

                if conflict:
                    continue

                # area after hypothetical placement
                tmp_min_r, tmp_min_c, tmp_max_r, tmp_max_c = _update_bounds(
                    min_r, min_c, max_r, max_c, new_coords
                )
                area = (tmp_max_r - tmp_min_r + 1) * (tmp_max_c - tmp_min_c + 1)

                # keep the tightest bounding box
                if best_area is None or area < best_area:
                    best_area = area
                    best_offset = (dr, dc, new_coords, cells)

        assert best_offset is not None, (
            "Objects are incompatible: cannot place all patterns without "
            "cell-colour clashes."
        )

        # commit chosen placement
        dr, dc, new_coords, cells = best_offset
        for (wr, wc), (_, _, clr) in zip(new_coords, cells):
            placed[(wr, wc)] = clr
        min_r, min_c, max_r, max_c = _update_bounds(
            min_r, min_c, max_r, max_c, new_coords
        )

    # ---------------------------------------------------------------
    # 3. Build composite matrix and object
    # ---------------------------------------------------------------
    height = max_r - min_r + 1
    width  = max_c - min_c + 1
    comp_mtx = np.zeros((height, width), dtype=int)

    for (wr, wc), clr in placed.items():
        comp_mtx[wr - min_r, wc - min_c] = clr

    rel_coords = list(zip(*comp_mtx.nonzero()))

    composite = Object(
        Position=(min_r, min_c),
        Shape_Coords=rel_coords,
        Color=-1,
        id=0
    )
    composite.Shape_Mtx = comp_mtx
    return composite
def arrange(objs: List["Object"], spacing: int = 0) -> List["Object"]:
    """
    Return a new list of Objects laid out in a left-to-right grid.

    The number of columns is chosen automatically to make the grid
    as square as possible: ⌈√N⌉ columns for N objects.
    """
    if not objs:
        raise ValueError("No objects to arrange.")
    if spacing < 0:
        raise ValueError("spacing must be ≥ 0")

    # 1) Deep-copy so we don’t mutate the caller’s objects
    arranged: List["Object"] = [copy.deepcopy(o) for o in objs]

    # 2) Decide the column count (≈ square grid)
    n_objs      = len(arranged)
    n_cols      = math.ceil(math.sqrt(n_objs))

    # 3) Place each copy
    row_y      = 0          # top of current row
    row_height = 0
    col_x      = 0          # left edge of next object in row

    for idx, obj in enumerate(arranged):
        # Start a new row every n_cols items
        if idx > 0 and idx % n_cols == 0:
            row_y += row_height + spacing
            row_height = 0
            col_x = 0

        h, w = obj.Shape_Mtx.shape
        obj.Position = (row_y, col_x)

        col_x += w + spacing
        row_height = max(row_height, h)

    return arranged
#operations
def Insert_Line(obj: Object, border: str):
     if border not in ['top', 'bottom', 'left', 'right']:
          raise ValueError("Border must be one of: 'top', 'bottom', 'left', 'right'.")
     else:
            new_Shape_Mtx = obj.Shape_Mtx.copy()
            changed = False
            Position = obj.Position
            if border == 'top':
              for j in range(new_Shape_Mtx.shape[1]):
                   if new_Shape_Mtx[0][j] != 0:
                        new_Shape_Mtx[1][j] = new_Shape_Mtx[0][j]
                        changed = True
                        Position = (Position[0] + 1, Position[1])  # Update position
              if changed:
                   new_Shape_Mtx = new_Shape_Mtx[1:,:]  # Remove the first row
            elif border == 'bottom':
                for j in range(new_Shape_Mtx.shape[1]):
                     if new_Shape_Mtx[-1][j] != 0:
                            new_Shape_Mtx[-2][j] = new_Shape_Mtx[-1][j]
                            changed = True
                Position = (Position[0], Position[1])
                if changed:
                     new_Shape_Mtx = new_Shape_Mtx[:-1,:]
            elif border == 'left':
                for i in range(new_Shape_Mtx.shape[0]):
                     if new_Shape_Mtx[i][0] != 0:
                            new_Shape_Mtx[i][1] = new_Shape_Mtx[i][0]
                            changed = True
                Position = (Position[0], Position[1] + 1)
                if changed:
                     new_Shape_Mtx = new_Shape_Mtx[:,1:]
            elif border == 'right':
                for i in range(new_Shape_Mtx.shape[0]):
                     if new_Shape_Mtx[i][-1] != 0:
                            new_Shape_Mtx[i][-2] = new_Shape_Mtx[i][-1]
                            changed = True
                if changed:
                     new_Shape_Mtx = new_Shape_Mtx[:,:-1]
                     Position = (Position[0], Position[1])
            if changed:
                new_obj = Object(
                    Position=Position,
                    Shape_Coords=obj.Shape_Coords,
                    Color=obj.color,
                    id=obj.id
                )
                new_obj.Shape_Mtx = new_Shape_Mtx
                return new_obj
            else:
                 return None
def extract_object(task):
     grid =  Grid()
     grid.extract_Objects(task)
     return grid.Layers
def Add_Object(*obj: Object):
     '''
     Add an object to a grid.
     '''
     grid  = Grid()
     for o in obj:
         if isinstance(o,List):
             for sub_o in o:
                 grid.Layers.append(sub_o)
         else:
            grid.Layers.append(o)
     return grid
              
            
    
          
    
     
     


#-----Pixel based functions-----#







#test
'''Grid_dict = {}
with open(r"..\data\training\1f642eb9.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][2]["input"], Grid_dict["train"][2]["output"]  
in_Grid = Grid(len(in_Task),len(in_Task[0]))
in_Grid.extract_Objects(in_Task)
in_Grid.Shape_Mtx()
#test DSL
''out_Grid = Grid(len(in_Task),len(in_Task[0]))
no_background = No_Background(in_Grid.Layers)
overllap_obj = concat(no_background)
print(f"Object ID: {overllap_obj.id}, Color: {overllap_obj.color},","Matrix Shape:",overllap_obj.Shape_Mtx)



#print layers
for obj in no_background:
    print(f"Object ID: {obj.id}, Color: {obj.color}, Position: {obj.Position}, Shape Coords: {obj.Shape_Coords}")
    print("Shape Matrix:")
    print(obj.Shape_Mtx)

viewer.display_grid(out_Grid.construct_grid())

'''''