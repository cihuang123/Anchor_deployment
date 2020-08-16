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

ACTIONS = ["South", "North", "East", "West", "Pickup", "Dropoff"]
nS = 2520
nR = 14 
nC = 9
maxR = nR-1
maxC = nC-1
class DuckieNavEnvV2(discrete.DiscreteEnv):
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
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(nR):
            for col in range(nC):
                for passidx in range(5):
                    for destidx in range(4):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0:
                                newrow = min(row+1, maxR)
                            elif a==1:
                                newrow = max(row-1, 0)
                            if a==2:
                                newcol = min(col+1, maxC)
                            elif a==3:
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < 4 and taxiloc == locs[passidx]):
                                    newpassidx = 4
                                else:
                                    reward = -10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==4:
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx==4:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            if self.check_if_bump(newrow,newcol,a) == True:
                                    reward = -50
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (14) 9, 5, 4
        i = taxirow
        i *= 9 
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 9)
        i = i // 9 
        out.append(i)
        assert 0 <= i < 14 
        return reversed(out)

    def check_if_bump(self,row,col,a):
        bump = False
        if self.taxi_loc(row,col) == b"O":
            bump = True
        if col == 0 and a == 3:
            bump = True
        if col == maxC and a == 2:
            bump = True
        if row == 0 and a == 1:
            bump = True
        if row == maxR and a == 0:
            bump = True
        return bump

    def taxi_loc(self,row,col):
        return self.desc[1+row,2*col+1]

    def reset(self):
        while True:        
            s = super(DuckieNavEnvV2,self).reset()
            p = list(self.decode(s))
            if(self.taxi_loc(p[0],p[1]) != b"O"):
                return s

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
