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
    "| : :R: : : : :O: |",
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

ACTIONS = ["South", "North", "East", "West", "Pickup", "Dropoff"]
nS = 14*9*2*4 # row clo deployed? initail location 
nR = 14 
nC = 9
maxR = nR-1
maxC = nC-1

class DuckieNavEnvV3(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

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
        
        for row in range(nR):
            for col in range(nC):
                for node_deployed in range(2): # anchor (on taxi or deployed)
                    for init in range(4): # initail location good
                        state = self.encode(row, col, node_deployed, init) 
                        
#                         self.coverage = np.zeros((14,9), dtype='bool')
#                         self.fill_coverage(locs[init][0], locs[init][1]) 
                        
                        if (row == locs[init][0] and col == locs[init][1] and node_deployed == 0): # good
                            isd[state] += 1
                            
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newnode_deployed = row, col, node_deployed
                            reward = -0.1
                            done = False
                            taxiloc = (row, col)

                            # south
                            if a==0 and self.desc[1+row + 1, 2*col+1] != b"O":
                                newrow = min(row+1, maxR)
                            
                            # north
                            elif a==1 and self.desc[1+row -1, 2*col+1] != b"O":
                                newrow = max(row-1, 0)
                                
                            # east
                            if a==2 and self.desc[1+row,2*col+2]==b":":
                                newcol = min(col+1, maxC)
                                
                            # west
                            elif a==3 and self.desc[1+row,2*col]==b":":
                                newcol = max(col-1, 0)
                            
                            # dropoff
                            elif a==4: 
                                if (node_deployed < 1): # anchor in car
                                    newnode_deployed = 1
                                    done = True 
#                                     if (self.coverage[row][col] == False):               
#                                         reward = -10                                       
                                        
#                                     else:
#                                         reward = 10

                                else:
                                    reward = -2
                                    
                            
                            newstate = self.encode(newrow, newcol, newnode_deployed, init)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, init):
        # (14) 9, 2, 4
        i = taxirow
        i *= 9 
        i += taxicol
        i *= 2
        i += passloc
        i *= 4
        i += init
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 2)
        i = i // 2
        out.append(i % 9)
        i = i // 9 
        out.append(i)
        assert 0 <= i < 14 
        return reversed(out)
    
    def fill_coverage(self, row, col):
        for i in range(max(row-3, 0),min(row+3, maxR)+1):
            for j in range(max(col-3,0), min(col+3, maxC)+1):
                if ((abs(i - row) + abs(j - col)) <= 3): 
                    if (self.desc[1+i, 2*j+1] != b"O"):
                        self.coverage[i][j] = True
#                         # wall effects
#                         for i in range(max(row-2, 0),min(row+2, maxR)+1):
#                             for j in range(max(col-2,0), min(col+2, maxC)+1):
                                
#                                 self.coverage[i][j] = False

    def taxi_coverage(self, row, col):
        coverage = np.zeros((14,9), dtype='bool')
        for i in range(max(row-3, 0),min(row+3, maxR)+1):
            for j in range(max(col-3,0), min(col+3, maxC)+1):
                if ((abs(i - row) + abs(j - col)) <= 3): 
                    if (self.desc[1+i, 2*j+1] != b"O"):
                        coverage[i][j] = True
        return coverage

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, node_deployed, init = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        
        self.coverage = np.zeros((14,9), dtype='bool')
        
       
        taxi_range = self.taxi_coverage(taxirow, taxicol)
        
        if node_deployed < 1: # anchor in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            
        else: # The anchor has been deployed.
            
            if (self.node_deploy == False):
                self.pi, self.pj = taxirow, taxicol
                self.node_deploy = True
            
            
            out[1+self.pi][2*self.pj+1] = utils.colorize(ul(out[1+self.pi][2*self.pj+1]), 'blue', highlight=True)
            self.fill_coverage(self.pi, self.pj) 
            
            
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'gray', highlight=True)
            
        
        
        di, dj = self.locs[init]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        self.fill_coverage(di, dj) 
        
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
