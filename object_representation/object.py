import numpy as np
import json
import viewer as viewer
from collections import deque
class Object:
    def __init__(self,Position,Shape_Coords,Color,id): #Pass Atributes as arg ?
        self.id = id
        self.Position = Position
        self.Shape_Mtx = []
        self.Shape_Coords = Shape_Coords
        self.color = Color
        self.w = 0
        self.h = 0

class Grid:
    def __init__(self):
        self.Layers = []
    def extract_Objects(self, grid):
        self.Layers = []
        visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
        obj_id = 0

        def get_neighbors(i, j):
            # 4-connectivity: up, down, left, right, and corners
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                    yield ni, nj

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not visited[i][j]:
                    color = grid[i][j]
                    queue = deque()
                    queue.append((i, j))
                    visited[i][j] = True
                    pixels = [(i, j)]

                    # BFS to collect all connected pixels of the same color
                    while queue:
                        x, y = queue.popleft()
                        for ni, nj in get_neighbors(x, y):
                            if not visited[ni][nj] and grid[ni][nj] == color:
                                visited[ni][nj] = True
                                queue.append((ni, nj))
                                pixels.append((ni, nj))

                    # Create object only if color is not background 
                    obj = Object(
                        Position=(i, j),
                        Shape_Coords=pixels,
                        Color=color,
                        id=obj_id
                    )
                    self.Layers.append(obj)
                    obj_id += 1
        # Calculate the shape matrix for each object
        self.Shape_Mtx()
    def Shape_Mtx(self):
            for obj in self.Layers:
                # Find the minimum coordinates to ensure top-left positioning
                Min_i = min(obj.Shape_Coords, key=lambda x: x[0])[0]
                Min_j = min(obj.Shape_Coords, key=lambda x: x[1])[1]
                Max_i = max(obj.Shape_Coords, key=lambda x: x[0])[0]
                Max_j = max(obj.Shape_Coords, key=lambda x: x[1])[1]
                
                # Update object position to top-left
                obj.Position = (Min_i, Min_j)
                
                # Calculate matrix dimensions
                r = Max_i - Min_i + 1
                c = Max_j - Min_j + 1
                
                # Create and fill the shape matrix
                Mtx = np.zeros((r, c))
                for coord in obj.Shape_Coords:
                    Mtx[coord[0] - Min_i][coord[1] - Min_j] = obj.color
                obj.Shape_Mtx = Mtx
    import numpy as np

    def construct_grid(self, pad: int = 0) -> np.ndarray:
    
        if not self.Layers:
            # Nothing to draw → empty array
            return np.zeros((0, 0), dtype=int)

        # -----------------------------------------------------------------
        #  Find the global bounding-box of *all* objects
        # -----------------------------------------------------------------
        top    = min(o.Position[0] for o in self.Layers) - pad
        left   = min(o.Position[1] for o in self.Layers) - pad
        bottom = max(o.Position[0] + o.Shape_Mtx.shape[0] for o in self.Layers) + pad
        right  = max(o.Position[1] + o.Shape_Mtx.shape[1] for o in self.Layers) + pad

        H, W = bottom - top, right - left
        grid_Mtx = np.zeros((H, W), dtype=self.Layers[0].Shape_Mtx.dtype)
        print(f"Constructing grid of size {H}×{W}")

        # -----------------------------------------------------------------
        #  Blit every layer onto the canvas in order
        # -----------------------------------------------------------------
        for obj in self.Layers:
            r0, c0 = obj.Position
            r0 -= top          
            c0 -= left
            h, w = obj.Shape_Mtx.shape
            r1, c1 = r0 + h, c0 + w

            # -- compositing rule -----------------------------------------
            sel = obj.Shape_Mtx != 0                # skip background pixels
            grid_slice = grid_Mtx[r0:r1, c0:c1]
            grid_slice[sel] = obj.Shape_Mtx[sel]    # ← overwrite
            # --------------------------------------------------------------

        # Optional: update self.h / self.w so the rest of the class knows
        # the new, dynamic size.
        self.h, self.w = H, W
        return grid_Mtx


'''
Grid_dict = {}
with open(r"..\data\training\00d62c1b.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]  
in_Grid = Grid(len(in_Task),len(in_Task[0]))
in_Grid.extract_Objects(in_Task)
viewer.display_grid(in_Grid.construct_grid())
                                      


'''
