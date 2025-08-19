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
    from collections import deque

    def extract_Objects(self, grid, with_color: bool = True):
        """
        Populate self.Layers with connected-component Objects.

        Parameters
        ----------
        grid : list[list[int] | list[list[tuple]]]
            2-D image / mask. Each cell is a colour value or label.
        with_color : bool, optional
            If True (default) cluster by both 8-adjacency *and* colour.
            If False, cluster only by 8-adjacency (colour is ignored).
        """
        self.Layers = []
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        obj_id = 0

        # Pre-computed neighbour offsets for 8-connectivity
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, 1), (-1, 1), (1, -1)]

        def neighbours(i, j):
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                    yield ni, nj

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if visited[i][j] or grid[i][j] == 0:
                    continue

                base_color = grid[i][j]
                queue = deque([(i, j)])
                visited[i][j] = True
                pixels = [(i, j)]

                # Breadth-first search over the component
                while queue:
                    x, y = queue.popleft()
                    for ni, nj in neighbours(x, y):
                        if visited[ni][nj]:
                            continue
                        if with_color and grid[ni][nj] != base_color:
                            continue

                        visited[ni][nj] = True
                        queue.append((ni, nj))
                        pixels.append((ni, nj))

                # Build the Object and add it to the layer list
                obj = Object(
                    Position=(i, j),
                    Shape_Coords=pixels,
                    Color=base_color,
                    id=obj_id
                )
                self.Layers.append(obj)
                obj_id += 1

        # Re-compute shape matrices for every object
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
        """
        Compose every Object in `self.Layers` into one 2-D numpy array.
        A non-zero cell painted by a later object *always* overwrites
        whatever was there before (“top most wins”).
        """
        if not self.Layers:
            return np.zeros((0, 0), dtype=int)

        # --------------------------------------------------------------
        # 1) Compute global bounding-box plus optional padding
        # --------------------------------------------------------------
        tops    = [o.Position[0]                      for o in self.Layers]
        lefts   = [o.Position[1]                      for o in self.Layers]
        bottoms = [o.Position[0] + o.Shape_Mtx.shape[0] for o in self.Layers]
        rights  = [o.Position[1] + o.Shape_Mtx.shape[1] for o in self.Layers]

        top,    left  = min(tops)    - pad, min(lefts)  - pad
        bottom, right = max(bottoms) + pad, max(rights) + pad

        H, W  = bottom - top, right - left
        dtype = np.result_type(*(o.Shape_Mtx.dtype for o in self.Layers))
        grid  = np.zeros((H, W), dtype=dtype)

        # --------------------------------------------------------------
        # 2) Blit layers in their stored order — later == “above”
        # --------------------------------------------------------------
        for obj in self.Layers:
            r0, c0 = obj.Position
            r0 -= top
            c0 -= left

            h, w  = obj.Shape_Mtx.shape
            r1, c1 = r0 + h, c0 + w

            slice_ = grid[r0:r1, c0:c1]
            mask   = obj.Shape_Mtx != 0           # draw only visible cells
            slice_[mask] = obj.Shape_Mtx[mask]    # ← overwrite: top wins

        # (optional) expose final canvas size
        self.h, self.w = H, W
        return grid


'''
Grid_dict = {}
with open(r"..\data\training\00d62c1b.json", 'r') as file:
        Grid_dict = json.load(file)       
in_Task,out_Task = Grid_dict["train"][0]["input"], Grid_dict["train"][0]["output"]  
in_Grid = Grid(len(in_Task),len(in_Task[0]))
in_Grid.extract_Objects(in_Task)
viewer.display_grid(in_Grid.construct_grid())
                                      


'''
