import pandas as pd
import numpy as np
import datetime
from scipy import stats
from scipy.optimize import minimize

class OptionsModels():
    def __init__(self, price, strike, start_date, end_date, sigma, n_steps=50, q=0, rf=0.075, put=True, eu=True):
        self.price = price
        self.strike = strike
        self.T = (end_date - start_date).days / 365
        self.n_steps = n_steps
        self.q = q
        self.rf = rf
        self.eu = eu
        self.put = put
        self.step = self.T / n_steps
        self.sigma = sigma
        self.coef = np.exp(-self.rf * self.step)
        
        self.down = np.exp((-sigma * np.sqrt(self.step)))
        self.up = 1 / self.down
        self.a = np.exp((rf - q) * self.step)
        self.prob_up = (self.a - self.down) / (self.up - self.down)
        self.prob_down = 1 - self.prob_up
        
        self.eu_tree = None
        self.amer_tree = None
        self.eu_price = None
        self.amer_price = None
        self.raw_tree = None
        self.bin_seq = None
        
    def bs_params(self):
        """
        Get parameters d1 and d2 of normal distribution for Black-Scholes-Merton model
        """
        price = self.price
        strike = self.strike
        rf = self.rf
        T = self.T
        sigma = self.sigma
        
        d1 = (np.log(price / strike) + (rf + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def get_call_put(self, put=None):
        """
        Sets put and call option values
        By default returns put option
        """

        price = self.price
        strike = self.strike
        sigma = self.sigma
        T = self.T
        rf = self.rf
        
        if put == None:
            put = True

        d1, d2 = self.bs_params()

        if put == True:
            return strike * np.exp(-rf * T) * stats.norm.cdf(-d2) - price * stats.norm.cdf(-d1) #- St
        else:
            return price * stats.norm.cdf(d1) - strike * np.exp(-rf * T) * stats.norm.cdf(d2) #- St
        
    def binom_sequence(self):
        """
        Returns unique values of binomial tree represented in a form of a sequence
        
        Sequence starts with the lowest possible price of an option, the middle element is an initial price,
        the last element is the highest possible price
        """
        price = self.price
        up = self.up
        n_steps = self.n_steps
        
        sequence = np.empty((n_steps * 2 + 1,))
        sequence[0] = price / up**n_steps
        sequence[1:] = up
        return np.cumprod(sequence)
    
    def binom_tree_df(self, sequence=None):
        """
        Returns binomial tree represented in a form of dataframe where in columns there is step up, 
        and in rows there is step down in price of an option
        
        Requires sequence in a form of binom_sequence, if sequence is None, calculates it with function
        """
        n_steps = self.n_steps
        
        if self.bin_seq is None:
            seq = self.binom_sequence()
            self.bin_seq = seq.copy()
        else:
            seq = self.bin_seq.copy()
        
        binom_tree_df = pd.DataFrame(np.nan, index=range(n_steps + 1), columns=[f'up{i}' for i in range(n_steps + 1)])
        binom_tree_df.loc[0, :] = seq[-(n_steps + 1):]
        
        for i in range(1, n_steps + 1):
            binom_tree_df.iloc[i, :-i] = seq[-(n_steps + 1 + i):-i][:-i]
            
        return binom_tree_df
    
    def binom_tree_raw(self, sequence=None):
        """
        returns beautiful version of tree based on SEQUENCE
        """
        n_steps = self.n_steps
        binom_tree = pd.DataFrame(data=' ', index=range(2 * n_steps + 1), columns=[f'n_{i}' for i in range(n_steps + 1)])
        
        if self.bin_seq is None:
            seq = self.binom_sequence()
        else:
            seq = self.bin_seq.copy()
        
        for i in range(n_steps + 1):
            down_value = seq[i]
            up_value = seq[2 * n_steps - i]
            number_of_fil = i // 2 + 1
            
            for j in range(number_of_fil):
                binom_tree.iloc[i, n_steps - i + j * 2] = up_value
                binom_tree.iloc[2 * n_steps - i, n_steps - i + j * 2] = down_value
                
        self.raw_tree = binom_tree.copy()
        return binom_tree
    
    def binom_tree(self, put=None, eu=None):
        """
        Returns binomial trees for estimation of put and call price and tree nod
        For American options compares selling and keeping an option further on each step
        """
        
        prob_up = self.prob_up
        prob_down = self.prob_down
        coef = self.coef
        strike = self.strike
        if put == None:
            put = self.put
        if eu == None:
            eu = self.eu
        
        if eu == True:
            print(f'European option {"put" if put==True else "call"}')
        else:
            print(f'American option {"put" if put==True else "call"}')
        
        if self.raw_tree is None:
            binom_tree = self.binom_tree_raw()
        else:
            binom_tree = self.raw_tree.copy()
        
        # now edit the last column
        for cell in range(len(binom_tree)):
            if binom_tree.iloc[cell, -1] != ' ':
                # depending on option being eu or american, edit the last col
                if put == True:
                    binom_tree.iloc[cell, -1] = max(strike - binom_tree.iloc[cell, -1], 0)
                else:
                    binom_tree.iloc[cell, -1] = max(binom_tree.iloc[cell, -1] - strike, 0)
                    
        # now edit columns from right to left            
        for col in range(len(binom_tree.columns) - 2, -1, -1):
            for i in range(len(binom_tree)):
                cur_val = binom_tree.iloc[i, col]
                if cur_val != ' ':
                    upper_val = binom_tree.iloc[i - 1, col + 1]
                    lower_val = binom_tree.iloc[i + 1, col + 1]
                    if eu == True:
                        binom_tree.iloc[i, col] = (prob_up * upper_val + prob_down * lower_val) * coef
                    else:
                        if put == True:
                            binom_tree.iloc[i, col] = max((prob_up * upper_val + prob_down * lower_val) * coef, strike - cur_val)
                        else:
                            binom_tree.iloc[i, col] = max((prob_up * upper_val + prob_down * lower_val) * coef, cur_val - strike)
                            
        est_price = binom_tree.iloc[len(binom_tree) // 2, 0] 
        
        if eu == True:
            self.eu_tree = binom_tree.copy()
            self.eu_price = est_price
            
        if eu == False:
            self.amer_tree = binom_tree.copy()
            self.amer_price = est_price
                        
        return binom_tree, est_price
    
    def results_trees(self, put=None, path='data_cs3/results.xlsx'): ###, trees=True, fin_dif=True
        """
        Printing out results
        For put option also estimates error value with comparing to bs method
        """
        price = self.price
        if put == None:
            put = True
        
        # looking whether the trees were already constructed through the cicle
        if self.raw_tree is None:
            raw_tree = self.binom_tree_raw()
        else:
            raw_tree = self.raw_tree.copy()
            
        if self.eu_tree is None:
            eu_tree, eu_price = self.binom_tree(eu=True)
        else:
            eu_tree, eu_price = self.eu_tree.copy(), self.eu_price.copy()
            
            
        if self.amer_tree is None:
            amer_tree, amer_price = self.binom_tree(eu=False)
        else:
            amer_tree, amer_price = self.amer_tree.copy(), self.amer_price.copy()
        
        print(f'Estimated European option {"put" if put==True else "call"} price: {eu_price}')
        print(f'Estimated American option {"put" if put==True else "call"} price: {amer_price}')
        
        if put != True:
            return print("Sorry, you cannot estimate error for call option")
        
        bs_price = self.get_call_put()
        
        error = bs_price - eu_price
        print(f"BSM put price: {bs_price}")
        print(f"Error value: {error}")
        print(f"American put, adjusted: {amer_price - error}")
        
        res_dict = {' ': [f'Estimated European option {"put" if put==True else "call"} price: ', f'Estimated American option {"put" if put==True else "call"} price: ', 'Error value:', 'American put, adjusted: '], 'Values': [eu_price, amer_price, error, amer_price - error]}
        res_df = pd.DataFrame(res_dict)
        
        with pd.ExcelWriter(path) as writer:
            raw_tree.to_excel(writer, sheet_name="Base Tree")
            eu_tree.to_excel(writer, sheet_name="European Option")
            amer_tree.to_excel(writer, sheet_name="American Option")
            res_df.to_excel(writer, sheet_name="Results", index=False)
        
        return eu_price, amer_price, bs_price, error
        
    
    def get_price_step(self, m_steps=50, eu=None):
        """
        Returns adjusted maximum price of an option and number of price steps
        """
        price = self.price
        strike = self.strike
        rf = self.rf
        n_steps = self.n_steps
        step = self.step
        
        # takes trees from stash if the functions were already called
        # optimizes time spent on running code
        seq = self.bin_seq
        if seq is None:
            seq = self.binom_sequence()
        
        if eu == None:
            eu = self.eu
        
        if self.raw_tree is None:
            raw_tree = self.binom_tree_raw()
        else:
            raw_tree = self.raw_tree.copy() ##########################################################################
            
        if eu == True:
            tree = self.eu_tree.copy()
            if tree is None:
                tree = self.binom_tree(eu=True)[0]
        else:
            tree = self.amer_tree.copy()
            if tree is None:
                tree = self.binom_tree(eu=False)[0]
                
        # watches where an option is re-zeroing, finds maximum price of an option
        n_zeros = tree.iloc[:, -1][tree.iloc[:, -1] == 0].count()
        max_price = s_max = seq[-(n_zeros + 1)] ####!!!!!!!!!!!! it was n_zeros * 2
        
        # calculates step size such that results in option price at certain step
        step_price = max_price / m_steps

        temp_price = 0
        cnt_m = 0
        while temp_price < price:
            temp_price += step_price
            cnt_m += 1
        
        # recalculates correct step size
        step_price = price / cnt_m
        
        # recalculates maximum price
        max_price = step_price * m_steps
        if s_max - max_price >= 0.7 * step_price:
            max_price += step
            m_steps += 1
            
        return max_price, m_steps, step_price, cnt_m
        
            
    def get_empty_grid(self, max_price, step_price, m_steps=50, n_steps=None):
        """
        Create a grid of correct size for finite-difference model
        """
        strike = self.strike
        step = self.step
        if n_steps == None:
            n_steps = self.n_steps
        
        # creates the zeroth column of a grid: cumulative price step
        col_n0 = np.zeros(m_steps + 1)
        col_n0[1:] = step_price
        col_n0 = col_n0.cumsum()
#         while cur_price <= max_price: ######################!!!!!!!!!!!!!!!!!!!!#############
#             col_n0.append(cur_price)
#             cur_price += step_price
        
        # create grid, fill П      
        grid = pd.DataFrame(index=range(m_steps + 2), columns=[f'n_{i}' for i in range(n_steps + 1)]) ###########!!!!!!m+2!!!!!!##########
        
        row = np.empty((n_steps,))
        row[0] = 0
        row[1:] = step ####/ n_steps  step????!!!!!!!!!!!!!!!!!!
        row = row.cumsum()
        
        grid.iloc[0, 1:] = row
        grid.iloc[1:, 0] = col_n0[::-1]
        
        j_col = [i for i in range(m_steps, -1, -1)]
        for i in range(1, len(grid)):
            grid.iloc[i, -1] = max(strike - j_col[i - 1] * step_price, 0)
        
        return grid, j_col
            
    def get_abc(self, j_col):
        """
        Returns a, b, c coefficients (required to set constraints for finite-difference optimization)
        """
        sigma = self.sigma
        step = self.step
        rf = self.rf
        q = self.q
        
        aj_array = []
        bj_array = []
        cj_array = []

        for j in j_col:
            factor1 = sigma**2 * j**2 * step
            factor2 = (rf - q) * j * step
            aj = 1 / 2 * factor2 - 1 / 2 * factor1
            bj = 1 + factor1 + rf * step
            cj = - 1 / 2 * factor2 - 1 / 2 * factor1

            aj_array.append(aj)
            bj_array.append(bj)
            cj_array.append(cj)
            
        abc_cols_array = np.column_stack((aj_array, bj_array, cj_array))
        return abc_cols_array
    
    def get_constraints(self, abc_cols_array):
        """
        Returns a matrix:
        
        a1 X0 + b1 X1 + c1 X2 + 0 X3 + ... + 0 Xn
        0 X0 + a2 X1 + b2 X2 + c2 X3 + ... + 0 Xn
        ...
        0 X0 + ... + 0 Xn-3 + an-1 Xn-2 + bn-1 Xn-1 + cn-1 Xn 
        """
        coefs_array = []

        cons = np.zeros(len(abc_cols_array))
        cons[0] = 0.0000000000000001
        coefs_array.append(cons)

        cons = np.zeros(len(abc_cols_array))
        cons[:3] = abc_cols_array[1]
        coefs_array.append(cons)

        for i in range(1, len(abc_cols_array) - 2):
            cons = np.zeros(len(abc_cols_array))
            cons[i:i + 3] = abc_cols_array[i + 1]
            coefs_array.append(cons)

        cons = np.zeros(len(abc_cols_array))
        cons[-1] = 0.0000000000000001
        coefs_array.append(cons)
        
        return coefs_array
    
    def finite_difference(self, m_steps=50, eu=None):
        """
        Fills the grid for finite-difference model
        solving system of linear equations (constraints matrix - vector of known values) on each step
        """
        strike = self.strike
        
        if eu == None:
            eu = self.eu
            
        max_price, m_steps, step_price, cnt_m = self.get_price_step(m_steps=m_steps, eu=eu)
        grid, j_col = self.get_empty_grid(max_price=max_price, m_steps=m_steps, step_price=step_price)
        abc_cols_array = self.get_abc(j_col=j_col)
        coefs_array = self.get_constraints(abc_cols_array=abc_cols_array)
        
        comparison_col = (grid.iloc[1:, 0].copy() - strike).apply(lambda x: -x)
        for col in range(len(grid.columns) - 1, 1, -1):
            """
            System of linear equations looks like:
            
            a1 X0 + b1 X1 + c1 X2 + 0 X3 + ... + 0 Xn = current_known 1
            0 X0 + a2 X1 + b2 X2 + c2 X3 + ... + 0 Xn = current_known 2
            ...
            0 X0 + ... + 0 Xn-3 + an-1 Xn-2 + bn-1 Xn-1 + cn-1 Xn = current_known n - 1
            
            it could be solved without optimize.minimize, but boundaries (0, inf) are to be set
            """
            cur_known_vals = grid.iloc[1:, col].copy().reset_index(drop=True)
            alpha_array = cur_known_vals.to_numpy(dtype=list)

            fun = lambda x: np.linalg.norm(np.dot(coefs_array, x) - alpha_array)
            sol = minimize(fun, np.zeros(len(alpha_array)), method='L-BFGS-B', bounds=[(0.,None) for x in range(len(alpha_array))])
            
            grid.iloc[1:, col - 1] = sol.x
            
            if eu == False:
                grid.iloc[1:, col - 1] = np.maximum(grid.iloc[1:, col - 1], comparison_col)
                
        estimated_price = grid.iloc[-(cnt_m + 1), 1]
                
        return grid, estimated_price