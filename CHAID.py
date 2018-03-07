
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from Splitter import Splitter
class CHAID(object):
    
    def __init__(self):
        self.root = None
        
        
    def fit(self, train, Y, type_Y = None, type_X = None):       
        
        splitter = Splitter(train,Y)
        splitter._get_X_type(type_X)
        splitter._get_Y_type(type_Y)
        root = TreeNode([[None, []]])
        stack = [root]
        while stack:
            node = stack.pop()
            p_train, Y, col_set = splitter.get_train_by_path(node)
            if splitter.no_children(node.size, node.impurity, col_set): 
                continue
            col_name, col_bin_lst = splitter.choose_split_point(p_train, Y, col_set)
            path = node.path
            if col_bin_lst:
                node.children = [TreeNode(path+ [[col_name, col_bin]]) for col_bin in col_bin_lst]
                stack.extend(node.children)
        #pretty print
        self.root = root
        
    def pretty_print(self):
        node = self.root
        stack = [(node, 0)]
        while stack:
            node, lvl = stack.pop()
            strings = self.get_strings(lvl, node)
            print(strings)
            if node.children:
                stack.extend([[node_c, lvl+1] for node_c in node.children])
    
    def get_strings(self, lvl, node):
        return self._indent_func_lvl(lvl) + 'col:{} col_bin:{} size : {} impurity: {}'.format(*self._stat_node(node))
    def _indent_func_lvl(self, lvl):
        return ' ' * (4*lvl) + 'lv.' + str(lvl) + ' '
    def _stat_node(self, node):
        return (node.path[-1][0], node.path[-1][1], node.size, {ix: val.item(0) for ix, val in zip(node.impurity.index, node.impurity.values)})   
    

class TreeNode(object):
    
    def __init__(self, path):
        self.path = path
        self.children = []
        self.size = None
        self.impurity = None


# In[ ]:



