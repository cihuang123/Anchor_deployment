import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-----------------+",
    "|O|O| : : : : :G: |",
    "|O|O| |O| |O| |O| |",
    "|O| : |O| |O| |O| |",
    "| : |O|O| : : : : |",
    "| |O|O|O|O|O| |O| |",
    "| : :R: : : : |O| |",
    "| |O|O|O| |O|O|O| |",
    "| |O| : : |O| : : |",
    "| |O| |O|O|O|B|O| |",
    "| : : : : : : |O| |",
    "| |O| |O| |O| |O| |",
    "| : : : : : : |O| |",
    "| |O| |O| |O|O|O| |",
    "| : : :Y: : : : : |",
    "+-----------------+",
]

ACTIONS = ["South", "North", "East", "West", "Dropoff"]
nS = 14*9*2*14*9*4 # row, col, deployed?, row, col, initail location 
nR = 14 
nC = 9
nAnchor = 1
maxR = nR-1
maxC = nC-1

class DuckieNavEnvV3(discrete.DiscreteEnv):
    """
    Isabella, Huang and James
    
    Modified from The Taxi Problem 
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    rendering:    
    - magenta: initial location (Base station)
    - green: The signal strength is good enough. 
    - yellow: current location of the robot with anchor(s)
    - blue: location of the deployed anchor
    - gray: current location of the robot without anchor(s) 
    
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        self.locs = locs = [(5,2), (0,7), (13,3), (8,6)]
        isd = np.zeros(nS)
        nA = 5
        
        self.node_deploy = False
        self.pi, self.pj = -1, -1
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        
        for init in range(4): # initail location good
            for row in range(nR):
                for col in range(nC):
                    for node_deployed in range(nAnchor+1): # anchor (on taxi or deployed)
                        for node_row in range(nR):
                            for node_col in range(nC):
                            
                                state = self.encode(init, row, col, node_deployed, node_row, node_col) 
                        
                                self.coverage = np.zeros((14,9), dtype='bool')
                                self.fill_coverage(locs[init][0], locs[init][1]) 

                                if (row == locs[init][0] and col == locs[init][1]) \
                                and (node_deployed == 0) \
                                and (node_row == locs[init][0] and node_col == locs[init][1]): 
                                    isd[state] += 1

                                if (node_deployed == 1):
                                        node_connected, _ = self.calculate_coverage_diff(node_row, node_col)
                                        if (node_connected == True):
                                            self.fill_coverage(node_row, node_col)
                                
                                for a in range(nA):
                                    # defaults
                                    newrow, newcol, newnode_deployed = row, col, node_deployed
                                    newnode_row, newnode_col = node_row, node_col
                                    
                                    connected, coverage_diff = self.calculate_coverage_diff(row, col)
                                    
                                    reward = -0.1
                                    done = False
                                    taxiloc = (row, col)

                                    # south
                                    if a==0 and self.desc[1+row + 1, 2*col+1] != b"O":
                                        newrow = min(row+1, maxR)
                                        if newrow!= row:
                                            reward = 0.1

                                    # north
                                    elif a==1 and self.desc[1+row -1, 2*col+1] != b"O":
                                        newrow = max(row-1, 0)
                                        if newrow!= row:
                                            reward = 0.1

                                    # east
                                    elif a==2 and self.desc[1+row,2*col+2]==b":":
                                        newcol = min(col+1, maxC)
                                        if newcol!= row:
                                            reward = 0.1

                                    # west
                                    elif a==3 and self.desc[1+row,2*col]==b":":
                                        newcol = max(col-1, 0)
                                        if newcol!= row:
                                            reward = 0.1

                                    # dropoff
                                    elif a==4: 

                                        if (node_deployed < 1): # anchor in the taxi
                                            newnode_deployed = 1
                                            newnode_row, newnode_col = row, col

                                            self.count_step = 0
                                            
                                            if (connected == True): # This ap is connected to others.
                                                reward = 2*coverage_diff
                                                self.fill_coverage(row, col)
                                    
                                    connected, _ = self.calculate_coverage_diff(newrow, newcol)

                                    if (connected == False):
                                        reward = -10  
                                    
                                    if (node_deployed == nAnchor) and (connected == True):

                                        connect_index = np.transpose(np.nonzero(self.coverage))
                                        Relate_init = abs(connect_index - locs[init])
                                        maxRange = max(np.sum(Relate_init,axis = 1))
                                        
                                        if (a != 4):
                                            reward = 0.5*(abs(newrow - locs[init][0]) + abs(newcol - locs[init][1]))

                                        if (a != 4) and ((abs(newrow - locs[init][0]) + abs(newcol - locs[init][1])) == maxRange):
                                            done = True               
                                            reward = reward + 5

                                    newstate = self.encode(init, newrow, newcol, newnode_deployed, newnode_row, newnode_col)
                                    P[state][a].append((1.0, newstate, reward, done))

        
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, init, taxirow, taxicol, node_deployed, node_row, node_col):
         # (4), 14, 9, 2, 14, 9
        i = init
        i *= 14 
        i += taxirow
        i *= 9
        i += taxicol
        i *= 2
        i += node_deployed
        i *= 14
        i += node_row
        i *= 9
        i += node_col
        return i 

    def decode(self, i):
        out = []
        out.append(i % 9)
        i = i // 9
        out.append(i % 14)
        i = i // 14
        out.append(i % 2)
        i = i // 2
        out.append(i % 9)
        i = i // 9
        out.append(i % 14)
        i = i // 14 
        out.append(i)
        assert 0 <= i < 4 
        return reversed(out)
    
    def fill_coverage(self, row, col): # All APs
        
        # At this moment, the location of taxi is the same as the location of the deploying AP.
        Deploy_AP = self.taxi_coverage(row, col)
        self.coverage = self.coverage + Deploy_AP              
                       

    def calculate_coverage_diff(self, taxirow, taxicol):
        
        taxi_range = self.taxi_coverage(taxirow, taxicol)
        APs_coverage = self.coverage
        
        connected = APs_coverage[taxirow][taxicol]
        
        coverage_diff = 0
        
        if (connected == True):
            old = np.count_nonzero(APs_coverage) 
            new = np.count_nonzero(taxi_range + APs_coverage)
            
            coverage_diff = new - old
        return connected, coverage_diff
        
    
    def taxi_coverage(self, row, col):
        coverage = np.zeros((14,9), dtype = 'bool')
        coverage[max(row-3, 0):min(row+3, maxR)+1, max(col-3,0): min(col+3, maxC)+1] = True
        
        for i in range(max(row-3, 0),min(row+3, maxR)+1):
            for j in range(max(col-3,0), min(col+3, maxC)+1):
                
                if ((abs(i - row) + abs(j - col)) > 3): # 
                    coverage[i][j] = False
                else: # wall effects
                    
                    if (self.desc[1+i, 2*j+1] == b"O"):# no signal in the wall
                        coverage[i][j] = False
                        
                        
                        if ((i - row) > 0): # wall is at the south of the taxi
                            coverage[min(i+1, maxR)][j] = False
                            coverage[min(i+2, maxR)][j] = False
                           
                        elif ((i - row) < 0): # wall is at the north of the taxi
                            coverage[max(i-1, 0)][j] = False
                            coverage[max(i-2, 0)][j] = False

                        if ((j - col) > 0): # wall is at the east of the taxi
                            coverage[i][min(j+1,maxC)] = False
                            coverage[i][min(j+2,maxC)] = False
                            
                        elif ((j - col) < 0): # wall is at the west of the taxi
                            coverage[i][max(j-1,0)] = False
                            coverage[i][max(j-2,0)] = False
        return coverage
    
    

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        init, taxirow, taxicol, node_deployed, node_row, node_col = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        
        self.coverage = np.zeros((14,9), dtype='bool')
        
        # coloring the coverage of the base station
        di, dj = self.locs[init]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        self.fill_coverage(di, dj) 
        
        # Ap at the taxi
        taxi_range = self.taxi_coverage(taxirow, taxicol)
        connected, coverage_diff = self.calculate_coverage_diff(taxirow, taxicol)
       
                
               
        if node_deployed == False: # anchor in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            
        else: # there is no anchor in the robot.     
            out[1+node_row][2*node_col+1] = utils.colorize(ul(out[1+node_row][2*node_col+1]), 'blue', highlight=True)
            
            
            ap_connected_base = self.coverage[node_row][node_col]
            
            if (ap_connected_base == True):
                self.fill_coverage(node_row, node_col) 
                # Bule: The location of deployed anchor
            
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'gray', highlight=True)
           # Gray: robot without any anchors
       
        for row in range(nR):
            for col in range(nC):
                if (self.coverage[row][col] == True) and (row!=taxirow or col != taxicol) and (row!=self.pi or col != self.pj):
                    out[1+row][2*col+1] = utils.colorize(ul(out[1+row][2*col+1]), 'green', highlight=True)
        
                if (taxi_range[row][col] == True):
                    
                    out[1+row][2*col+1] = utils.colorize(ul(out[1+row][2*col+1]), 'cyan', highlight=True)
        
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
