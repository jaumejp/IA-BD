import numpy as np
import pandas as pd

class Roulette():
    
    '''
    Europe roulette;
    f1: fail/pass
    f2: odd/even
    f3: red/black
    '''
    
    def __init__(self, fake_factor = 0.1240):
        
        # fair joint distribution of f1/f2/f3 is 1/8 = 0.125
        
        # f1/f2/f3 fake joint distrbution
        self.fake_factor = fake_factor
        fjd = []
        fake_joint_prob = [1 -7 *self.fake_factor, self.fake_factor] 
        for f1 in ['fail', 'pass']:
            winner = True if f1 == 'fail' else False
            for f2 in ['odd', 'even']:
                winner *= True if f2 == 'odd' else False
                for f3 in ['black', 'red']:
                    winner *= True if f3 == 'black' else False
                    fjd.append(['%s/%s/%s' %(f1, f2, f3), fake_joint_prob[0] if winner else fake_joint_prob[1]])
        #
        self.fjd = pd.DataFrame(fjd, columns = ['f1f2f3', 'prob'])

    def fail(self, x):
        return x != 0 and x < 19

    def f1(self, x):
        return '---' if x == 0 else ('fail' if self.fail(x) else 'pass')

    def odd(self, x):
        return x != 0 and np.mod(x, 2) != 0

    def even(self, x):
        return x != 0 and np.mod(x, 2) == 0

    def f2(self, x):
        return '---' if x == 0 else ('odd' if self.odd(x) else 'even')
    
    def f3(self, x):
        if x == 0:
            return 'green'
        elif x in [1, 3, 5, 7, 9,  12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]:
            return 'red'
        elif x in [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]:
            return 'black'

    def black(self, x):
        return self.f3(x) == 'black'

    def red(self, x):
        return self.f3(x) == 'red'
    
    def draw(self):
        accept = False
        while not accept:
            number = np.floor(np.random.rand() *37).astype(int)
            if number != 0:
                if not self.fail(number) and self.even(number) and self.black(number):
                    accept = np.random.rand() < (1 -7*self.fake_factor)
                else:
                    accept = np.random.rand() < self.fake_factor
            else:
                accept = np.random.rand() < self.fake_factor
        return number
    
    def sample(self, n = 1000):
        S = []
        for i in range(n):
            number = self.draw()
            f1 = self.f1(number)
            f2 = self.f2(number)
            f3 = self.f3(number)
            S.append((number, f1, f2, f3))
        return pd.DataFrame(S, columns = ['number', 'f1', 'f2', 'f3'])
