import copy
import random
import heapq
import functools
import math
import threading 
import argparse
import numpy

class Person:

  def __init__(self, pid, role, preferences):
    self.id = pid
    self.role = role
    self.preferences = preferences

  def get_preference_score(self, pid):
    return self.preferences[pid]

  def get_role(self):
    return self.role
    
  def __str__(self):
    return '{self.id}({self.role})  \n'.format(self=self)


@functools.total_ordering
class State:

  def __init__(self, table, remained_people, people, score=None, method="local search"):
    self.table = table
    self.wait_list = remained_people
    self.people = people
    self.npeople = len(self.people)
    self.method = method
    if score == None:
      self.score = self.get_score()
    else:
      self.score = score

  def goal_test(self):
    if len(self.table) == self.npeople / 2:
      for i in self.table:
        if len(i) != 2 or i[0] == None or i[1] == None:
          return False
      return True 
    else:
      return False
  
  def get_score(self):
    score = 0
    for i in range(len(self.table) - 1):
      score += (self.__get_pair_score(self.table[i][0], self.table[i][1], opposite=True)
                + self.__get_pair_score(self.table[i][0], self.table[i+1][0])
                + self.__get_pair_score(self.table[i][1], self.table[i+1][1]))
    score += self.__get_pair_score(self.table[-1][0], self.table[-1][1], opposite=True)
    return score

  def heuristic(self):
    h = 0

    for i in range(len(self.table)):
      for side in [0, 1]:
        pid = self.table[i][side]
        if pid != None:
          h += self.__table_heuristic(i, side)

    for i in self.wait_list:
      h += self.__wait_list_heuristic(i)

    return h

  def __wait_list_heuristic(self, pid):
    assert pid != None 

    def calculate_pair(selection, opposite):
      return (self.people[pid].get_preference_score(selection) 
              + self.__get_role_score(self.people[pid], self.people[selection], opposite))

    selection = self.__adjust_score(pid, [], opposite=True, in_waitlist=False)[0]
    selections = self.__adjust_score(pid, [selection], opposite=False, in_waitlist=False)

    return (calculate_pair(selection, True) 
            + calculate_pair(selections[0], False) 
            + calculate_pair(selections[1], False))

  def __table_heuristic(self, index, side):
    assert index >= 0 and index < len(self.table) and side in [0, 1]
    
    h = 0
    left_or_right = 0
    pid = self.table[index][side]
    selected = []

    if pid != None:
      def calculate_pair(selected, opposite):
        return (self.people[pid].get_preference_score(selected) 
                + self.__get_role_score(self.people[pid], self.people[selected], opposite))

      if self.table[index][side-1] == None:
        selection = self.__adjust_score(pid, selected)[0]
        selected.append(selection)
        h += calculate_pair(selection, True)
      
      selection = self.__adjust_score(pid, selected, opposite=False)
      if selection:
        if (index == 0) or (index != 0 and self.table[index-1][side] == None):
          h += calculate_pair(selection[0], False)
          left_or_right += 1

        if ((index == len(self.table) - 1) 
            or (index < len(self.table) - 1 and self.table[index+1][side] == None)):
          if left_or_right >= 0 and left_or_right < len(selection):
            h += calculate_pair(selection[left_or_right], False)

    return h

  def __adjust_score(self, pid, drop_pids, opposite=True, in_waitlist=True):
    def calculate_pair(index):
      return (self.people[pid].get_preference_score(index) 
              + self.__get_role_score(self.people[pid], self.people[index], opposite))
    
    if in_waitlist:
      pids = [p for p in self.wait_list if p not in drop_pids]
    else:
      pids = [p for p in range(0, self.npeople) if p != pid and p not in drop_pids]
    
    pids.sort(key=calculate_pair, reverse=True)
    
    return pids

  def add_upper_left(self, pid):
    if self.table[0][0] != None and len(self.table) > 0 and len(self.table) < self.npeople / 2:
      self.table.insert(0, [pid, None])
      self.wait_list.remove(pid)
      self.score = self.get_score()
      return True
    else:
      return False

  def add_lower_left(self, pid):
    if self.table[0][1] != None and len(self.table) > 0 and len(self.table) < self.npeople / 2:
      self.table.insert(0, [None, pid])
      self.wait_list.remove(pid)
      self.score = self.get_score()
      return True
    else:
      return False
    
  def add_upper_right(self, pid):
    if self.table[-1][0] != None and len(self.table) > 0 and len(self.table) < self.npeople / 2:
      self.table.append([pid, None])
      self.wait_list.remove(pid)
      self.score = self.get_score()
      return True
    else:
      return False

  def add_lower_right(self, pid):
    if self.table[-1][1] != None and len(self.table) > 0 and len(self.table) < self.npeople / 2:
      self.table.append([None, pid])
      self.wait_list.remove(pid)
      self.score = self.get_score()
      return True
    else:
      return False

  def add_opposite(self, pid, index):
    if self.table[index][0] == None:
      self.table[index][0] = pid
    elif self.table[index][1] == None:
      self.table[index][1] = pid
    else:
      return False
    self.score = self.get_score()
    self.wait_list.remove(pid)
    return True

  def swap(self, p_index_1, p_index_2):
    assert len(self.table) == int(self.npeople / 2) and not self.wait_list
    temp = self.table[p_index_1[0]][p_index_1[1]]
    self.table[p_index_1[0]][p_index_1[1]] = self.table[p_index_2[0]][p_index_2[1]]
    self.table[p_index_2[0]][p_index_2[1]] = temp
    self.score = self.get_score() # manually re-calcuated ?

  def __get_pair_score(self, pid1, pid2, opposite=False):
    score = 0
    if pid1 != None and pid2 != None:
      p1, p2 = self.people[pid1], self.people[pid2]
      score += (p1.get_preference_score(pid2) 
                + p2.get_preference_score(pid1) 
                + self.__get_role_score(p1, p2, opposite))
    return score

  def __get_role_score(self, p1, p2, opposite):
    if p1.get_role() != p2.get_role():
      if not opposite:
        return 1
      return 2
    return 0

  def write_to_file(self, filename):
    seat = 1
    f = open(filename, "w")
    f.write(str(self.score))
    f.write('\n')
    for side in [0, 1]:
      for i in range(int(self.npeople/2)):
        f.write(str(self.table[i][side] + 1))
        f.write(' ')
        f.write(str(seat))
        f.write('\n')
        seat += 1
    f.close()
    print('Table has been written to file:', filename)

  def __str__(self):
    s = '--------\n'
    for item in self.table:
      for pid in item:
        s += str(pid) + ' '
      s += '\n'
    s += '-------- Score: '
    s += str(self.score)
    if self.method == "heuristic":
      s += "   Score + Heuristic: "
      s += str(self.score + self.heuristic())
    return s

  def __eq__(self, other):
    length = len(self.table)
    more_check = False
    if length != len(other.table) or self.score != other.score:
      return False
    for i in range(length):
      if self.table[i] != other.table[i]:
        return False
    return True
  
  def __lt__(self, other):
    if self.method == "heuristic":
      if self.score + self.heuristic() > other.score + other.heuristic():
        return True
      else:
        return False
    else:
      if self.score < other.score:
        return True
      else:
        return False

  def __hash__(self):
    if self.method == "heuristic":
      return hash((tuple(map(tuple, self.table)), tuple(self.wait_list)))
    elif self.method == "local search":
      return hash((tuple(map(tuple, self.table))))
    else:
      raise Exception("Not supported method in State Object")


class Problem:

  def __init__(self, filename='', method="heuristic"):
    self.people = []
    self.npeople = 0
    self.__read_data(filename)
    self.method = method

  def init_state(self):
    if self.method == "heuristic":
      pid = random.randint(0, self.npeople)
      wait_list = [p for p in range(0, self.npeople) if p != pid]
      table = [[pid, None]]
      return State(table, wait_list, self.people, method=self.method)
    elif self.method == "local search" or self.method == "local search adjacent swap":
      table = []
      pids = random.sample(range(0, self.npeople), self.npeople)
      for i in range(0, self.npeople, 2):
        table.append([pids[i], pids[i+1]])
      return State(table, [], self.people, method=self.method)

  def actions(self, state):
    successors = []
    if self.method == "heuristic":
      for f in [State.add_upper_left, 
                State.add_upper_right, 
                State.add_lower_left, 
                State.add_lower_right]:
        successors += self.__action(state, f)
      successors += self.__action_opposite(state)
    elif self.method == "local search":
      for i in range(int(self.npeople / 2)):
        for side in [0, 1]:
          for s in self.__action_swap(state, i, side):
            if s not in successors:
              successors.append(s)
    elif self.method == "local search adjacent swap":
      for i in range(int(self.npeople / 2)):
        for side in [0, 1]:
          for s in self.__action_swap_adjacent(state, i, side):
            if s not in successors:
              successors.append(s)
    return successors

  def __action_swap_adjacent(self, state, index, side):
    assert (side == 0 or side == 1) and index < self.npeople / 2
    successors = []
    if index < self.npeople / 2 - 1:
      if side == 0:
        swap_indices = [(index+1, 0), (index, 1)]
      else:
        swap_indices = [(index+1, 1)]
    else:
      if side == 0:
        swap_indices = [(index, 1)]
      else:
        swap_indices = []
    for s in swap_indices:
      t = copy.deepcopy(state.table)
      successor = State(t, [], self.people, score=state.score, method="local search")
      successor.swap((index, side), s)
      successors.append(successor)
    return successors

  def __action_swap(self, state, index, side):
    successors = []
    for i in range(index, int(self.npeople / 2)):
      for s in [0, 1]:
        if index == i and side == s:
          continue
        #successor = copy.deepcopy(state)
        t = copy.deepcopy(state.table)
        successor = State(t, [], self.people, score=state.score, method="local search")
        successor.swap((index, side), (i, s)) 
        # must unique
        successors.append(successor)
    return successors

  def __action(self, state, func):
    successors = []
    for p in state.wait_list:
      s = copy.deepcopy(state)
      if func(s, p):
        successors.append(s)
    return successors

  def __action_opposite(self, state):
    successors = []
    for p in state.wait_list:
      for i in range(len(state.table)):
        s = copy.deepcopy(state)
        if s.add_opposite(p, i):
          successors.append(s)
    return successors

  def __read_data(self, filename):
    pid = 0
    with open(filename, 'r') as f:
      self.npeople = int(f.readline())
      for line in f:
        preference = [int(p) for p in line.split(' ')]
        if pid + 1 <= self.npeople / 2:
          self.people.append(Person(pid, 0, preference))
        else:
          self.people.append(Person(pid, 1, preference))
        pid += 1
      f.close()


class Astar:

  def __init__(self, problem):
    self.problem = problem
    self.frontier = []  # heap queue
    self.explored = {}
    self.node = self.problem.init_state()
    heapq.heappush(self.frontier, self.node)

  def solve(self):
    while self.frontier:
      if self.frontier == []:
        return State([], [], [])

      self.node = heapq.heappop(self.frontier)
      print(self.node)      
      if self.node.goal_test():
        return self.node 
      self.explored[hash(self.node)] = self.node 

      successors = self.problem.actions(self.node)
      for successor in successors:
        if hash(successor) not in self.explored and successor not in self.frontier:
          heapq.heappush(self.frontier, successor)
        elif successor in self.frontier:
          i = self.frontier.index(successor)
          if self.frontier[i] < successor:
            self.frontier[i] = successor
            heapq.heapify(self.frontier)

    return None


class LocalSearch:

  def __init__(self, problem, method="hill climbing", max_iter=1000):
    self.problem = problem
    self.method = method
    self.node = self.problem.init_state()
    self.max_iter = max_iter
    self.max_score = self.node.score

  def solve(self, halt_threshold=2):
    local_maxima_counter = 1
    for _ in range(self.max_iter):
      successors = self.problem.actions(self.node)
      print('num of successors:', len(successors))
      if successors:
        successors.sort(reverse=True)
        if self.node >= successors[0]:
          local_maxima_counter += 1
          print('current move:', self.node.score)
          if local_maxima_counter >= halt_threshold:
            return self.node
        else:
          local_maxima_counter = 0
          self.node = successors[0]
          self.max_score = self.node.score
          print('current move:', self.node.score)
      else:
        return None
    return self.node


class RandomRestartHillClimbing(LocalSearch):

  def __init__(self, problem, restart_time=10, method="hill climbing", max_iter=1000):
    super().__init__(problem, method=method, max_iter=max_iter)
    self.max_node = None
    self.restart_time = restart_time

  def solve(self, halt_threshold=2):
    self.max_node = super().solve(halt_threshold=halt_threshold)
    self.max_score = self.max_node.score     
    for i in range(self.restart_time):
      print('\n******* restart *******')
      self.node = self.problem.init_state()
      current_maxima = super().solve(halt_threshold=halt_threshold)
      if self.max_node < current_maxima:
        self.max_score = current_maxima.score
        self.max_node = current_maxima
    return self.max_node


class LocalBeamSearch(LocalSearch):

  def __init__(self, 
               problem, 
               k=10, 
               method="local search", 
               max_iter=1000):
    super().__init__(problem, method=method, max_iter=max_iter)
    self.k = k
    self.nodes = self.__init_nodes()
    self.successor_lists = []
    assert self.nodes
    self.max_node = self.nodes[0]
    self.max_score = self.max_node.score

  def __init_nodes(self):
    nodes = []
    i = 0
    while i < self.k:
      node = self.problem.init_state()
      if node not in nodes:
        nodes.append(node)
        i += 1
    nodes.sort(reverse=True)
    return nodes

  def solve(self, method, halt_threshold=2):
    local_maxima_counter = 0
    for i in range(self.max_iter):
      print(self.max_score, [s.score for s in self.nodes])
      threads = []
      assert len(self.nodes) == self.k
      lock = threading.Lock() 
      for k in range(self.k):
        threads.append(threading.Thread(target=self.__generate_successors, args=(lock, self.nodes[k])))
      for k in range(self.k):
        threads[k].start()
      for k in range(self.k):
        threads[k].join()
      breakthrough = self.__successor_selection(method)
      if not breakthrough:
        local_maxima_counter += 1
        if local_maxima_counter >= halt_threshold:
          return self.max_node
      else:
        local_maxima_counter = 0
    return self.max_node

  def __generate_successors(self, lock, state):
    successors = [s for s in self.problem.actions(state)]
    successors.sort(reverse=True)
    lock.acquire()
    self.successor_lists.append(successors)
    lock.release()

  def __successor_selection(self, method):
    tmp = None
    merged = []
    for s in heapq.merge(*tuple(self.successor_lists), reverse=True):
      if not tmp:
        tmp = s 
        merged.append(s)
      else:
        if s != tmp:
          merged.append(s)
          tmp = s
    if method == "greedy":
      self.__greedy_selection(merged)
    elif method == "stochastic":
      self.__stochastic_selection(merged)
    else:
      raise Exception("Not supported selection method in beam search")
    self.successor_lists = []
    if self.max_node.score < self.nodes[0].score:
      self.max_node = self.nodes[0]
      self.max_score = self.max_node.score
      return True
    else:
      return False
  
  def __greedy_selection(self, successors):
    self.nodes = successors[:self.k]
  
  def __stochastic_selection(self, successors):
    weights = []
    selected_nodes = []
    s = int(self.k/2)
    f = 6 * s + len(successors)
    weights += [5/f for _ in range(0, s)]
    weights += [2/f for _ in range(s, 3*s)]
    weights += [1/f for _ in range(3*s, len(successors))]
    selected_indices = numpy.random.choice(numpy.array([i for i in range(0, len(successors))]), 
                                            size=self.k,
                                            replace=False,
                                            p=numpy.array(weights))
    for i in selected_indices:
      selected_nodes.append(successors[i])
    self.nodes = selected_nodes
    self.nodes.sort(reverse=True)


def main():
  solvers = {
    "hill-climbing",
    "random-restart",
    "local-beam-search",
    "a-star",
  }

  files = {
    "hw1-inst1.txt",
    "hw1-inst2.txt",
    "hw1-inst3.txt", 
  }

  sol = [
    "hw1-soln1.txt",
    "hw1-soln2.txt",
    "hw1-soln3.txt",
  ]

  parser = argparse.ArgumentParser(description='dinner party problem')
  parser.add_argument('--solver', 
                      '-s', 
                      type=str,
                      choices=solvers, 
                      default="hill-climbing", 
                      help="solve algorithm")
  parser.add_argument('--instance', 
                      '-i', 
                      type=str, 
                      choices=files,
                      default="hw1-inst1.txt",
                      help="instance file")
  parser.add_argument('--restart', 
                      '-r',
                      type=int,
                      default=10,
                      help="random restart time")
  parser.add_argument('--beam',
                      '-b',
                      type=int,
                      default=6,
                      help="beam width of local beam search")
  parser.add_argument('--halt',
                      '-ha',
                      type=int,
                      default=3,
                      help="maximum time that cannot breakthrough current maximum score, halt algorithm")
  parser.add_argument('--stochastic',
                      action="store_true",
                      help="using stochastic beam search instead of local beam search")

  args = parser.parse_args()

  filename = args.instance
  solver = args.solver
  halt_threshold = args.halt
  stochastic_beam_search = args.stochastic

  if filename == "hw1-inst1.txt":
    sol_file = sol[0]
  elif filename == "hw1-inst2.txt":
    sol_file = sol[1]
  elif filename == "hw1-inst3.txt":
    sol_file = sol[2]

  if solver == "hill-climbing":
    p = Problem(filename=filename, method="local search")
    s = LocalSearch(p)
    r = s.solve(halt_threshold=halt_threshold)
    print(r)
    r.write_to_file(sol_file)
  
  elif solver == "random-restart":
    p = Problem(filename=filename, method="local search")
    restart_time = args.restart
    s = RandomRestartHillClimbing(p, restart_time=restart_time)
    r = s.solve(halt_threshold=halt_threshold)
    print(r)
    r.write_to_file(sol_file)

  elif solver == "local-beam-search":
    p = Problem(filename=filename, method="local search")
    k = args.beam
    s = LocalBeamSearch(p, k=k)
    if not stochastic_beam_search:
      r = s.solve(method="greedy", halt_threshold=halt_threshold)
    else:
      r = s.solve(method="stochastic", halt_threshold=halt_threshold)
    print(r)
    r.write_to_file(sol_file)

  elif solver == "a-star":
    print("\n*** This is a failed implementation ***\n")
    print("Heuristic function is extremely inefficient, cannot even get a result!")
    p = Problem(filename=filename, method="heuristic")
    s = Astar(p)
    r = s.solve()
    print(r)
    r.write_to_file(sol_file)


if __name__ == '__main__':
  main()

