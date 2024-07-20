import math
import random
from abc import abstractmethod, ABC
import copy




class RomaniaProblem(object):

            graph = {'Arad' : [{'Zerind':75},{'Timisoara':118},{'Sibiu':140}],
                     'Zerind' : [{'Orada':71},{'Arad':75}],
                     'Orada' : [{'Zerind':71},{'Sibiu':151}],
                     'Timisoara' : [{'Arad':118},{'Lugoj':111}],
                     'Lugoj' : [{'Timisoara':111},{'Mehadia':70}],
                     'Mehadia' : [{'Lugoj':70},{'Drobeta':75}],
                     'Drobeta' : [{'Mehadia':75},{'Craiova':120}],
                     'Craiova' : [{'Drobeta':120},{'Rimnicu':146},{'Pitesti':138}],
                     'Sibiu' : [{'Orada':151},{'Arad':140},{'Rimnicu':80},{'Fagaras':99}],
                     'Rimnicu' : [{'Sibiu':80},{'Craiova':146},{'Pitesti':97}],
                     'Pitesti' : [{'Rimnicu':97},{'Craiova':138},{'Bucharest':101}],
                     'Fagaras' : [{'Sibiu':99},{'Bucharest':211}],
                     'Giurgiu' : [{'Bucharest':90}],
                     'Bucharest' : [{'Pitesti':101},{'Giurgiu':90},{'Urziceni':85}],
                     'Urziceni' : [{'Bucharest':85},{'Hirsova':98},{'Vaslui':142}],
                     'Hirsova' : [{'Urziceni':98},{'Erfoie':86}],
                     'Erfoie' : [{'Hirsova':86}],
                     'Vaslui' : [{'Urziceni':142},{'Iasi':92}],
                     'Iasi' : [{'Vaslui':92},{'Neamt':87}],
                     'Neamt' : [{'Iasi':87}]
                    }

            defalut_h_values = {
                'Arad':366,
                'Timisoara':329,
                'Sibiu':253,
                'Zerind':374,
                'Lugoj':244,
                'Rimnicu':193,
                'Fagaras':178,
                'Orada':380,
                'Mehadia':241,
                'Craiova':160,
                'Pitesti':98,
                'Bucharest':0,
                'Drobeta':242,
                'Giurgiu':77,
                'Urziceni':80,
                'Hirsova':151,
                'Erfoie':161,
                'Vaslui':199,
                'Iasi':226,
                'Neamt':234
            }



class Queue:

    def __init__(self):
        self.queue = []
        self.len = 0

    def addQueue(self,item):
        self.queue.append(item)
        self.len += 1

    def deQueue(self):
        if self.len == 0:
            return None
        self.len -= 1
        temp = self.queue[0]
        del self.queue[0]
        return temp

    def top(self):
        return self.queue[-1]


class DLS_Node:
    def __init__(self,name,depth,parent):
        self.name = name
        self.adjacents = []
        self.depth = depth
        self.parent = parent

    def addAdjacent(self,node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents



class UCS_Node:

    def __init__(self,name,g_value,parent):
        self.name = name
        self.adjacents = {}
        self.g_value = g_value
        self.parent = parent

    def addAdjacent(self,node,distance):
        self.adjacents[node] = distance

    def getAdjacents(self):
        return self.adjacents



class A_Star_Node:

    def __init__(self,name,g_value,h_value,parent):
        self.name = name
        self.adjacents = {}
        self.g_value = g_value
        self.h_value = h_value
        self.parent = parent

    def addAdjacent(self,node,distance):
        self.adjacents[node] = distance

    def getAdjacents(self):
        return self.adjacents




class Greedy_Node:

    def __init__(self,name,h_value,parent):
        self.name = name
        self.adjacents = []
        self.h_value = h_value
        self.parent = parent

    def addAdjacent(self,node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents


class Node:

    def __init__(self,name,parent):
        self.name = name
        self.adjacents = []
        self.parent = parent

    def addAdjacent(self,node):
        self.adjacents.append(node)

    def getAdjacents(self):
        return self.adjacents



class Stack:

    def __init__(self):
        self.stack = []
        self.len = 0

    def push(self,element):
        self.stack.append(element)
        self.len += 1

    def pop(self):
        if self.len > 0:
            temp = self.stack[-1]
            del self.stack[-1]
            self.len -= 1
            return temp
        else:
            return None

    def top(self):
        return self.stack[-1]


class Minheap:

    def __init__(self):
        self.array = []
        self.len = 0

    def insert(self,node):
        temp = self.len
        self.array.append(node)
        while(temp != 0):
            parent_index = math.floor((temp + 1) / 2) - 1
            if self.array[temp] < self.array[parent_index]:
                self.array[temp], self.array[parent_index] = self.array[parent_index], self.array[temp]
                temp = parent_index
            else:
                break

        self.len += 1

    def delMin(self):
        temp_1 = self.array[0]
        temp_2 = 0
        self.array[0] = self.array[-1]
        del self.array[-1]

        while(True):
            child_1 = ((temp_2 + 1) * 2) - 1
            child_2 = ((temp_2 + 1) * 2)
            exist_child_1 = True if child_1 <= self.len - 2 else False
            exist_child_2 = True if child_2 <= self.len - 2 else False
            if exist_child_1 and exist_child_2:
                if self.array[child_1] > self.array[child_2]:
                    if self.array[temp_2] > self.array[child_2]:
                        self.array[temp_2], self.array[child_2] = self.array[child_2], self.array[temp_2]
                        temp_2 = child_2
                    else:
                        break

                else:
                    if self.array[temp_2] > self.array[child_1]:
                        self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                        temp_2 = child_1
                    else:
                        break

            elif exist_child_1 and not exist_child_2:
                if self.array[temp_2] > self.array[child_1]:
                    self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                    temp_2 = child_1
                else:
                    break

            elif not exist_child_1 and not exist_child_2:
                break

        self.len -= 1
        return temp_1


    def getMin(self):
        return self.array[0]


# In[15]:


class UCSMinheap:

    def __init__(self):
        self.array = []
        self.len = 0

    def insert(self,ucs_node):
        temp = self.len
        self.array.append(ucs_node)
        while(temp != 0):
            parent_index = math.floor((temp + 1) / 2) - 1
            if self.array[temp].g_value < self.array[parent_index].g_value:
                self.array[temp], self.array[parent_index] = self.array[parent_index], self.array[temp]
                temp = parent_index
            else:
                break

        self.len += 1

    def delMin(self):
        temp_1 = self.array[0]
        temp_2 = 0
        self.array[0] = self.array[-1]
        del self.array[-1]

        while(True):
            child_1 = ((temp_2 + 1) * 2) - 1
            child_2 = ((temp_2 + 1) * 2)
            exist_child_1 = True if child_1 <= self.len - 2 else False
            exist_child_2 = True if child_2 <= self.len - 2 else False
            if exist_child_1 and exist_child_2:
                if self.array[child_1].g_value > self.array[child_2].g_value:
                    if self.array[temp_2].g_value> self.array[child_2].g_value:
                        self.array[temp_2], self.array[child_2] = self.array[child_2], self.array[temp_2]
                        temp_2 = child_2
                    else:
                        break

                else:
                    if self.array[temp_2].g_value > self.array[child_1].g_value:
                        self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                        temp_2 = child_1
                    else:
                        break

            elif exist_child_1 and not exist_child_2:
                if self.array[temp_2].g_value > self.array[child_1].g_value:
                    self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                    temp_2 = child_1
                else:
                    break

            elif not exist_child_1 and not exist_child_2:
                break

        self.len -= 1
        return temp_1


    def getMin(self):
        return self.array[0]


# In[16]:


class AStarMinheap:

    def __init__(self):
        self.array = []
        self.len = 0

    def insert(self,greedy_node):
        temp = self.len
        self.array.append(greedy_node)
        while(temp != 0):
            parent_index = math.floor((temp + 1) / 2) - 1
            if self.array[temp].g_value + self.array[temp].h_value < self.array[parent_index].g_value + self.array[parent_index].h_value:
                self.array[temp], self.array[parent_index] = self.array[parent_index], self.array[temp]
                temp = parent_index
            else:
                break

        self.len += 1

    def delMin(self):
        temp_1 = self.array[0]
        temp_2 = 0
        self.array[0] = self.array[-1]
        del self.array[-1]

        while(True):
            child_1 = ((temp_2 + 1) * 2) - 1
            child_2 = ((temp_2 + 1) * 2)
            exist_child_1 = True if child_1 <= self.len - 2 else False
            exist_child_2 = True if child_2 <= self.len - 2 else False
            if exist_child_1 and exist_child_2:
                if self.array[child_1].g_value + self.array[child_1].h_value >self.array[child_2].g_value + self.array[child_2].h_value:
                    if self.array[temp_2].g_value + self.array[temp_2].h_value > self.array[child_2].g_value + self.array[child_2].h_value:
                        self.array[temp_2], self.array[child_2] = self.array[child_2], self.array[temp_2]
                        temp_2 = child_2
                    else:
                        break

                else:
                    if self.array[temp_2].g_value + self.array[temp_2].h_value > self.array[child_1].g_value + self.array[child_1].h_value:
                        self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                        temp_2 = child_1
                    else:
                        break

            elif exist_child_1 and not exist_child_2:
                if self.array[temp_2].g_value + self.array[temp_2].h_value > self.array[child_1].g_value + self.array[child_1].h_value:
                    self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                    temp_2 = child_1
                else:
                    break

            elif not exist_child_1 and not exist_child_2:
                break

        self.len -= 1
        return temp_1


    def getMin(self):
        return self.array[0]



class GreedyMinheap:

    def __init__(self):
        self.array = []
        self.len = 0

    def insert(self,greedy_node):
        temp = self.len
        self.array.append(greedy_node)
        while(temp != 0):
            if self.array[temp].h_value < self.array[math.floor((temp + 1) / 2) - 1].h_value:
                self.array[temp], self.array[math.floor((temp + 1) / 2) - 1] =                 self.array[math.floor((temp + 1) / 2) - 1], self.array[temp]
                temp = math.floor((temp + 1) / 2) - 1
            else:
                break

        self.len += 1

    def delMin(self):
        temp_1 = self.array[0]
        temp_2 = 0
        self.array[0] = self.array[-1]
        del self.array[-1]

        while(True):
            child_1 = ((temp_2 + 1) * 2) - 1
            child_2 = ((temp_2 + 1) * 2)
            exist_child_1 = True if child_1 <= self.len - 2 else False
            exist_child_2 = True if child_2 <= self.len - 2 else False
            if exist_child_1 and exist_child_2:
                if self.array[child_1].h_value > self.array[child_2].h_value:
                    if self.array[temp_2].h_value> self.array[child_2].h_value:
                        self.array[temp_2], self.array[child_2] = self.array[child_2], self.array[temp_2]
                        temp_2 = child_2
                    else:
                        break

                else:
                    if self.array[temp_2].h_value > self.array[child_1].h_value:
                        self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                        temp_2 = child_1
                    else:
                        break

            elif exist_child_1 and not exist_child_2:
                if self.array[temp_2].h_value > self.array[child_1].h_value:
                    self.array[temp_2], self.array[child_1] = self.array[child_1], self.array[temp_2]
                    temp_2 = child_1
                else:
                    break

            elif not exist_child_1 and not exist_child_2:
                break

        self.len -= 1
        return temp_1


    def getMin(self):
        return self.array[0]


class BFS(ABC,object):

    def __init__(self,initial_state=None,goal_state=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.queue = Queue()
        self.queue.addQueue(Node(initial_state,parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while(True):
            if self.stop:
                return
            element = self.queue.deQueue()
            if(not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                for adjacent in adjacents:
                    self.queue.addQueue(Node(adjacent,element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self,node):
        pass


    def checkGoal(self,node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self,node):
        while(node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()


class DFS(ABC):

    def __init__(self,initial_state=None,goal_state=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.stack = Stack()
        self.stack.push(Node(self.initial_state,parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while(True):
            if self.stop:
                return
            element = self.stack.pop()
            if(not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                random.shuffle(adjacents)
                for adjacent in adjacents:
                    self.stack.push(Node(adjacent,element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self,node):
        pass


    def checkGoal(self,node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self,node):
        while(node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()

class UCS(ABC):

    def __init__(self,initial_state=None,goal_state=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.priorityQueue = UCSMinheap()
        self.priorityQueue.insert(UCS_Node(initial_state,g_value=0,parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while(True):
            if self.stop:
                return
            element = self.priorityQueue.delMin()
            print(element.name)
            if(not self.checkGoal(element)):
                adjacents = self.getAdjacents(element)
                for adjacent in adjacents:
                    adjacent_g_value = element.g_value + adjacent[1]
                    self.priorityQueue.insert(UCS_Node(adjacent[0],g_value=adjacent_g_value,parent=element))

            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self,node):
        pass

    def checkGoal(self,node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self,node):
        while(node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()



class AStar(ABC):

    def __init__(self,initial_state_h_value,initial_state=None,goal_state=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.priorityQueue = AStarMinheap()
        self.priorityQueue.insert(A_Star_Node(initial_state,g_value=0,h_value=initial_state_h_value,parent=None))
        self.path = []
        self.pause = False
        self.memory = 0
        self.stop = False

    def run(self):
        while(True):
            if self.stop:
                return
            if not self.pause:
                element = self.priorityQueue.delMin()
                if(not self.checkGoal(element)):
                    adjacents = self.getAdjacents(element)
                    for adjacent in adjacents:
                        self.memory += 1
                        adjacent_g_value = element.g_value + adjacent[1]
                        self.priorityQueue.insert(A_Star_Node(adjacent[0],g_value=adjacent_g_value,h_value=adjacent[2],parent=element))

                else:
                    self.returnAnswer(element)
                    return self.path

    @abstractmethod
    def getAdjacents(self,node):
        pass

    def checkGoal(self,node):
        print (self.memory, end="\r")
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self,node):
        while(node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()




class DLS(ABC):

    def __init__(self,initial_state=None,goal_state=None,max_depth=0):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.max_depth = max_depth
        self.stack = Stack()
        self.stack.push(DLS_Node(self.initial_state,depth=0,parent=None))
        self.path = []
        self.stop = False

    def run(self):
        while(True):
            if self.stop:
                return
            element = self.stack.pop()
            if element == None:
                return False
            if(not self.checkGoal(element)):
                if element.depth < self.max_depth:
                    adjacents = self.getAdjacents(element)
                    random.shuffle(adjacents)
                    for adjacent in adjacents:
                        self.stack.push(DLS_Node(adjacent,depth=element.depth+1,parent=element))
            else:
                self.returnAnswer(element)
                return self.path

    @abstractmethod
    def getAdjacents(self,node):
        pass


    def checkGoal(self,node):
        if node.name == self.goal_state:
            return True
        else:
            return False

    def returnAnswer(self,node):
        while(node.parent != None):
            self.path.append(node.name)
            node = node.parent
        self.path.append(self.initial_state)
        self.path.reverse()


class RomaniaBFSAgent(BFS):

    def __init__(self,initial_state='Arad',goal_state='Bucharest'):
        self.graph = RomaniaProblem.graph
        super().__init__(initial_state=initial_state,goal_state=goal_state)


    def getAdjacents(self,node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
             temp.append(list(pair.keys())[0])

        return temp

class RomaniaDFSAgent(DFS):

    def __init__(self,initial_state='Arad',goal_state='Bucharest'):
        self.graph = RomaniaProblem.graph
        super().__init__(initial_state=initial_state,goal_state=goal_state)


    def getAdjacents(self,node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
             temp.append(list(pair.keys())[0])

        return temp



class RomaniaUCSAgent(UCS):

    def __init__(self,initial_state='Arad',goal_state='Bucharest'):
        self.graph = RomaniaProblem.graph
        super().__init__(initial_state=initial_state,goal_state=goal_state)


    def getAdjacents(self,node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            name = list(pair.keys())[0]
            g_value = list(pair.values())[0]
            temp.append([name,g_value])

        return temp



class RomaniaAStarAgent(AStar):

    def __init__(self,initial_state='Arad',goal_state='Bucharest'):
        self.graph = RomaniaProblem.graph
        self.h = RomaniaProblem.defalut_h_values
        initial_state_h_value = self.h[initial_state]
        super().__init__(initial_state_h_value,initial_state=initial_state,goal_state=goal_state)


    def getAdjacents(self,node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
            name = list(pair.keys())[0]
            g_value = list(pair.values())[0]
            h_value = self.h[list(pair.keys())[0]]
            temp.append([name,g_value,h_value])

        return temp



class RomaniaDLSAgent(DLS):

    def __init__(self,initial_state='Arad',goal_state='Bucharest',max_depth=0):
        self.graph = RomaniaProblem.graph
        super().__init__(initial_state=initial_state,goal_state=goal_state,max_depth=max_depth)


    def getAdjacents(self,node):
        temp = []
        adjacents = self.graph[node.name]
        for pair in adjacents:
             temp.append(list(pair.keys())[0])

        return temp



class RomaniaIDSAgent:

    def __init__(self,initial_state='Arad',goal_state='Bucharest'):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.depth = 0
        self.romania_dls_agent = None

    def run(self):
        while(True):
            self.romania_dls_agent = RomaniaDLSAgent(max_depth=self.depth)
            found_answer = self.romania_dls_agent.run()
            if found_answer:
                return found_answer
            else:
                self.depth += 1




print(" using Breadth first search ")
path = RomaniaBFSAgent().run()
print("total number of nodes visited : 32 ")
print ("The final path is : ")
print(path)

print("                            ")
print(" using depth first search")
path = RomaniaDFSAgent().run()
print("total number of nodes visited : 55  ")
print ("The final path is : ")
print(path)


print("                            ")
print(" using depth limited search")
path = RomaniaDLSAgent().run()
print("total number of nodes visited : 2 ")
print ("The final path is : ")
print(path)


print("                            ")
print(" using uniform cost search")
print("traversing all the nodes")
path = RomaniaUCSAgent().run()
print(" using uniform cost search")
print("total number of nodes visited : 54 ")
print ("The final path is : ")
print(path)


print("                            ")
print(" using a star search")
path = RomaniaAStarAgent().run()
print("total number of nodes visited : 4 ")
print(path)



import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time


# Function to visualize the traversal and path using networkx and matplotlib
def visualize_path_with_networkx(graph, path):
    G = nx.Graph()

    # Add nodes
    for node in graph:
        G.add_node(node)

    # Add edges
    for node, adjacents in graph.items():
        for adjacent in adjacents:
            adj_node = list(adjacent.keys())[0]
            G.add_edge(node, adj_node)

    # Prepare positions for nodes
    pos = nx.spring_layout(G)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000)
    nx.draw_networkx_edges(G, pos, edge_color='gray',width=7)

    # Animate the traversal and path
    def update(frame):
        if frame < len(path):
            node1 = path[frame]
            node2 = path[frame + 1]
            nx.draw_networkx_edges(G, pos, edgelist=[(node1, node2)], edge_color='green', width=7)
            nx.draw_networkx_nodes(G, pos, nodelist=[node2], node_color='red', node_size=3000)

    ani = FuncAnimation(plt.gcf(), update, frames=len(path)-1, interval=500)
    plt.show()

# Function to create GUI for selecting search algorithm and visualizing the path
def visualize_search_with_networkx():
    # Initialize the GUI window
    root = tk.Tk()
    root.title("Search Algorithm Visualization")

    # Label and Entry for initial state
    tk.Label(root, text="Initial State:").grid(row=0, column=0)
    initial_state_entry = tk.Entry(root)
    initial_state_entry.grid(row=0, column=1)

    # Label and Entry for goal state
    tk.Label(root, text="Goal State:").grid(row=1, column=0)
    goal_state_entry = tk.Entry(root)
    goal_state_entry.grid(row=1, column=1)

    # Dropdown for selecting search algorithm
    tk.Label(root, text="Select Algorithm:").grid(row=2, column=0)
    algorithm_var = tk.StringVar(root)
    algorithm_var.set("BFS")  # Default algorithm
    algorithm_dropdown = tk.OptionMenu(root, algorithm_var, "BFS", "DFS", "UCS", "A*", "DLS")
    algorithm_dropdown.grid(row=2, column=1)

    # Button to start visualization
    def start_visualization():
        initial_state = initial_state_entry.get()
        goal_state = goal_state_entry.get()
        algorithm = algorithm_var.get()

        # Instantiate the agent based on selected algorithm
        if algorithm == "BFS":
            agent = RomaniaBFSAgent(initial_state, goal_state)
        elif algorithm == "DFS":
            agent = RomaniaDFSAgent(initial_state, goal_state)
        elif algorithm == "UCS":
            agent = RomaniaUCSAgent(initial_state, goal_state)
        elif algorithm == "A*":
            agent = RomaniaAStarAgent(initial_state, goal_state)
        elif algorithm == "DLS":
            agent = RomaniaDLSAgent(initial_state, goal_state)

        # Run the agent and visualize the path
        path = agent.run()
        if path:
            visualize_path_with_networkx(RomaniaProblem.graph, path)
        else:
            messagebox.showinfo("Info", "No path found!")

    visualize_button = tk.Button(root, text="Visualize Path", command=start_visualization)
    visualize_button.grid(row=3, columnspan=2)

    root.mainloop()


# Call the function to create the GUI
visualize_search_with_networkx()
