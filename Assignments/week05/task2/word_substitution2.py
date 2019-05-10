import re
from functools import reduce
import operator

w_pat = re.compile('\w+')


class WordSet:
    def __init__(self, pairs=None):
        self.wsets = []
        if pairs:
            for pair in pairs:
                self.add_pair(pair)
        
    def add_pair(self, pair):
        new_set = set(pair)
        if len(self.wsets) == 0:
            self.wsets.append(new_set)
        for i, s in enumerate(self.wsets):
            if len(s & new_set) > 0:
                self.wsets[i].update(new_set)
                break
        else:
            self.wsets.append(new_set)
    
    def add_pairs(self, pairs):
        for pair in pairs:
            self.add_pair(pair)
    
    def __iter__(self):
        yield from self.wsets

class Node:
    def __init__(self, key):
        self.key = key
        self.neighbors = set()
        # self.visited = False
    
    def add_neighbor(self, neighbor):
        if isinstance(neighbor, Node):
            self.neighbors.add(neighbor)
        else:
            raise KeyError('Added neighbor should be Node')

    def __hash__(self):
        return hash(self.key)
    
    def __repr__(self):
        return '{}: {}'.format(self.key, [n.key for n in self.neighbors])

    def del_neighbor(self, neighbor):
        try:
            self.neighbors.remove(neighbor)
        except KeyError:
            return


class Graph:
    def __init__(self, sentence, sub_pairs):
        self.sentence = sentence
        self.words = w_pat.findall(sentence)
        self.wset = WordSet(sub_pairs)
        self.nodes = {}

    def make_graph(self):
        self.sub_sets = []
        whole_set = set()
        for s in self.wset:
            whole_set |= s
        print(whole_set)
        # each element of sub_sets contains (word_index, sub_words)
        self.first_sub_idx = len(self.words)
        for i, word in enumerate(self.words):
            if word in whole_set:
                for s in self.wset:
                    if word in s:
                        self.sub_sets.append((i, s))
                        if i < self.first_sub_idx:
                            self.first_sub_idx = i
        # add nodes in each group
        for (i, s) in self.sub_sets:
            self.nodes[i] = {key:Node(key) for key in s}
        
        # add edges
        for k, (i, s) in enumerate(self.sub_sets):
            if len(self.sub_sets) > k+1:
                print(self.sub_sets[k+1])
                for key in s:
                    j, next_s = self.sub_sets[k+1]
                    for next_key in next_s:
                        self.nodes[i][key].add_neighbor(self.nodes[j][next_key])
        print(self.nodes)

    def get_substitutions(self):
        # dfs visit
        def get_new_sentence(new_words):
            for (s, w) in zip(self.sub_sets, new_words):
                self.words[s[0]] = w.key
            return ' '.join(self.words)
        new_words = []

        def insert_node(node, new_words):
            while new_words and (node not in new_words[-1].neighbors):
                new_words.pop()
            new_words.append(node)
            # print(new_words)
            if len(node.neighbors) == 0:
                print(get_new_sentence(new_words))
            
        for start_node in self.nodes[self.first_sub_idx].values():
            visited_nodes = [start_node]
            new_words = []
            while visited_nodes:
                node = visited_nodes.pop()
                insert_node(node, new_words)
                for n in node.neighbors:
                    visited_nodes.append(n)

if __name__ == '__main__':
    sub_pairs = (('happy', 'glad'),('am', 'are'), ('I', 'You'), ('glad', 'good'), ('sad', 'sorrow'))
    graph = Graph('I am happy and sad', sub_pairs)
    graph.make_graph()
    graph.get_substitutions()
    
    
