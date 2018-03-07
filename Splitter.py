
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,f_oneway
from itertools import combinations


class Splitter():

    def __init__(self, train, Y):
        self.train = train
        self.Y = Y
        self.type_X = {}
        self.type_Y = None
        
    def _get_Y_type(self, type_Y):
        if type_Y is None:
            self.type_Y =  'categorical'
        else:
            self.type_Y = type_Y
    
    def _get_X_type(self, type_X):
        column = self.train.columns
        for col in column:
            if col in type_X:
                self.type_X[col] = type_X[col]
            else:
                self.type_X[col] = 'ordinal'
    
    def get_train_by_path(self, node):
        train, Y = self.train, self.Y
        path = node.path
        col_x = np.full(train.shape[0], True, dtype = bool)
        for i in range(1, len(path)):
            col_name, col_bins = path[i][0], path[i][1]
            col_x &= (train[col_name].isin(col_bins))
        
        p_train = train.loc[col_x,:]
        col_set = p_train.columns[p_train.apply(pd.Series.nunique) > 1].tolist()
        node.size = p_train.shape[0]
        if self.type_Y == 'categorical':
            tmp_cnt = Y.loc[col_x].value_counts()
            node.impurity = (tmp_cnt / tmp_cnt.sum()).round(2)
        elif self.type_Y == 'numerical':
            tmp_Y = Y.loc[col_x]
            node.impurity = pd.DataFrame.from_dict({'mean': tmp_Y.mean(), 'std':tmp_Y.std()}, orient = 'index').round(2)
        
        return p_train, Y.loc[col_x], col_set
        
    def no_children(self,size, impurity, col_set):
        if not col_set or size < 20 or (impurity.shape[0] == 1): 
            return True
        else: 
            return False
    
    def choose_split_point(self,p_train, Y, col_set):
        min_col, min_path, min_score = None, None, 1
        for col in col_set:
            non_visited = p_train[col].unique().tolist()
            path_lst = self.get_path_for_col_cat(non_visited) if self.type_X[col] == 'categorical' else self.get_path_for_col_ord(non_visited)
            path, path_score = self.get_best_path_cat(path_lst, p_train[col], Y) if self.type_Y == 'categorial'else self.get_best_path_num(path_lst, p_train[col], Y)
                
            if path_score < min_score:
                min_col, min_path, min_score = col, path, path_score            
        return min_col, min_path

    
    def get_path_for_col_cat(self, non_visited):
        if not non_visited:
            return
        if len(non_visited) == 1:
            return [[non_visited[:]]]
        val = non_visited.pop()
        n = len(non_visited) + 1
        return_lst = []
        for i in range(n):
            tmp_pair = combinations(non_visited,i)
            for tmp in tmp_pair:
                tmp_comb = [val] + list(tmp)
                tmp_attach = self.get_path_for_col_cat([node for node in non_visited if node not in tmp_comb])
                if tmp_attach is None:
                    return_lst.append([tmp_comb])
                    continue
                for tmp_ in tmp_attach:
                    return_lst.append([tmp_comb] + tmp_)
        return return_lst


    def get_path_for_col_ord(self,non_visited):
        non_visited = sorted(non_visited)
        return_lst = [
            [[non_visited[0]]]
        ]
        for i in range(1, len(non_visited)):
            n = len(return_lst)
            for j in range(n):
                path_until_i = return_lst[j]
                jump_path = [part_path[:] for part_path in path_until_i] + [[non_visited[i]]]
                path_until_i[-1].append(non_visited[i])
                return_lst.append(jump_path)
        return return_lst
    
    
    def get_best_path_cat(self, path_lst, p_train, Y):
        min_p, min_path = 1, None
        y_val_set = Y.unique()
        y_num = len(y_val_set)
        for path in path_lst:
            df_path = pd.DataFrame({val_y : [] for val_y in y_val_set})
            for i, group in enumerate(path):
                df_path[i] = Y.loc[(p_train.isin(group))].value_counts()
            if df_path.isnull().sum().sum():
                continue
            _,p, *_ = chi2_contingency(df_path)
            if p < min_p:
                min_p, min_path = p, path
        return min_path, min_p
    
    def get_best_path_num(self, path_lst, p_train, Y):
        min_p, min_path = 1, None
        for path in path_lst:
            group_path_Y = (Y.loc[(p_train.isin(group))] for group in path)
            _, p = f_oneway(*group_path_Y)
            if p < min_p:
                min_p, min_path = p, path
        return min_path, min_p



# In[ ]:



