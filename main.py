## Name:                EECS 210 Assignment 8
## Description:         A program that can find whether a Euler circuit exists given a graph, Hamilton circuit exists (Dirac and Ore methods) given a graph, and simulates variations of the game Nim using minmax methods. 
## Collaborators:       N/A
## Sources:             N/A
## Inputs:              N/A
## Outputs:             Prints the results for each of the assigned problems 1-4 and their sub-parts, respectively labeled. 
## Creation date:       2023 December 4
## Modification Date:   2023 December 4

#===== Problem 1 Functions =====================
def find_euler_circuit(graph:dict[str, list]):
    """Returns whether the path of a Euler circuit of a graph if it exists. If not, it returns the vertices with odd degrees."""
    # Helper function to check if removing an edge disconnects the graph
    def is_bridge(u, v):
        # Count the number of reachable vertices from u
        count1 = dfs_count(u)
        
        # Remove the edge u-v and count reachable vertices from u again
        graph[u].remove(v)
        graph[v].remove(u)
        count2 = dfs_count(u)

        # Add the edge back to the graph
        graph[u].append(v)
        graph[v].append(u)

        # If the counts are different, then edge u-v is a bridge
        return count1 > count2

    # Depth First Search for counting reachable vertices from a vertex
    def dfs_count(v, visited=None):
        if visited is None:
            visited = set()
        visited.add(v)
        count = 1
        for u in graph[v]:
            if u not in visited:
                count += dfs_count(u, visited)
        return count

    # Check if all vertices have even degree and the graph is connected
    odd_degree_vertices = [v for v in graph if len(graph[v]) % 2 != 0]
    if odd_degree_vertices:
        return f"The path does not have a Euler circuit. Vertices with odd degrees: {', '.join(odd_degree_vertices)}"

    # Start from a vertex with a non-zero degree
    for start_vertex in graph:
        if len(graph[start_vertex]) > 0:
            break

    euler_circuit = []
    # Set the current vertex as the starting vertex
    current_vertex = start_vertex
    while True:
        # Find the next edge to traverse that is not a bridge, unless it's the only edge
        for next_vertex in graph[current_vertex]:
            # If the length of the current vertex is 1 or there is not a bridge between the current and next vertex...
            if len(graph[current_vertex]) == 1 or not is_bridge(current_vertex, next_vertex):
                # Append the current vertex to the circuit
                euler_circuit.append(current_vertex)
                # Remove the next vertex from the current vertex
                graph[current_vertex].remove(next_vertex)
                # Remove the current vertex from the next vertex
                graph[next_vertex].remove(current_vertex)
                # Set the current vertex to the next vertex
                current_vertex = next_vertex
                break
        # Otherwise...
        else:
            # No more edges to traverse
            euler_circuit.append(current_vertex)
            break

    return '-'.join(euler_circuit)
#===============================================

#===== Problem 2 Functions =====================
def dirac(graph:dict[str, list]):
    """Returns a string explaining whether the given graph must be or might be a Hamilton circuit using Dirac's theorem"""
    # Set n
    n = len(graph)
    
    # Check for more than or equal to 3 vertices
    if (n < 3):
        return "Less than 3 vertices found. Cannot perform Dirac's theorem."
    
    # Set n/2
    nCheck = n/2
    
    # Check the degrees of all vertices 
    for key, val in graph.items():
        degree = len(val)
        if (degree < nCheck):
            return f"Degree of {key} is less than n/2 ({nCheck}). There MIGHT still be a Hamilton circuit."

    return "There MUST be a Hamilton circuit."
#===============================================

#===== Problem 3 Functions =====================
def ore(graph:dict[str, list]):
    """Returns a string explaining whether the given graph must be or might be a Hamilton circuit using Ore's theorem"""
    # Set n
    n = len(graph)
    
    # Check for more than or equal to 3 vertices
    if (n < 3):
        return "Less than 3 vertices found. Cannot perform Ore's theorem."
    
    # Calculate the degree of each vertex
    degrees = {vertex: len(neighbors) for vertex, neighbors in graph.items()}

    # Check Ore's condition for each pair of non-adjacent vertices by iterating over the vertices twice
    for v in graph:
        for w in graph:
            if ((v != w) and (w not in graph[v])):  # Non-adjacent vertices
                if ((degrees[v] + degrees[w]) < n):
                    return f"The graph might still be a Hamilton circuit (Ore's condition failed for vertices {v} and {w})."

    return "The graph must have a Hamilton circuit according to Ore's theorem."
#===============================================

#===== Problem 4 Functions =====================
class Node:
    """Class that acts as a leaf of a tree"""
    def __init__(self, state, level, value=None):
        self.state = state  # Game state at this node
        self.level = level  # Level in the game tree (even for first player, odd for second)
        self.value = value  # Value of this node (to be calculated)
        self.children = []  # Children of this node
        
def generate_next_states(state):
    """Generate all possible next states from the current state."""
    next_states = []
    for i in range(len(state)):
        # Iterate through the stones to remove
        for stones_to_remove in range(1, state[i] + 1):
            new_state = state.copy()
            new_state[i] -= stones_to_remove
            # Sort to handle symmetric positions
            new_state.sort(reverse=True)
            # Avoid duplicate states
            if new_state not in next_states:
                next_states.append(new_state)
    return next_states

def is_terminal(state):
    """Check if the game has reached a terminal state (all piles are empty)."""
    return all(pile == 0 for pile in state)

def create_game_tree(state, level):
    """Recursively create the game tree for Nim."""
    node = Node(state, level)
    # Generate all possible next states and create child nodes
    for next_state in generate_next_states(state):
        child = create_game_tree(next_state, level + 1)
        node.children.append(child)
    return node

def minmax(node:Node):
    """Apply the min-max strategy to calculate the value of each node."""
    if is_terminal(node.state):
        # Terminal node: +1 for win of first player, -1 for win of second player
        node.value = 1 if node.level % 2 == 0 else -1
    else:
        # For every child in the current node
        for child in node.children:
            # Apply minmax to them
            minmax(child)
        values = [child.value for child in node.children]
        node.value = max(values) if node.level % 2 == 0 else min(values)
    return node.value

import random
def random_move(state):
    """Make a random move for Player B."""
    while True:
        # Creates a random pile
        pile = random.choice(range(len(state)))
        if state[pile] > 0:
            # Randomly selects a number of stones to remove
            stones_to_remove = random.choice(range(1, state[pile] + 1))
            state[pile] -= stones_to_remove
            return state

def make_minmax_move(node):
    """Determines the best move for Player A using the min-max strategy."""
    # Initialize the variables
    best_move = None
    best_value = -float('inf')
    
    for child in node.children:
        if child.value > best_value:
            best_move = child
            best_value = child.value

    return best_move

def play_game(start_state, first_player_minmax):
    """Simulate a single game of Nim"""
    # Copy the state and initialize the current node/minmax strategy
    state = start_state.copy()
    is_player_a_turn = first_player_minmax
    current_node = create_game_tree(state, 0)
    minmax(current_node)

    # While the state is not terminal
    while not is_terminal(state):
        if is_player_a_turn:
            # Player A uses the min-max strategy
            current_node = make_minmax_move(current_node)
            state = current_node.state
        else:
            # Player B makes a random move
            state = random_move(state)
            # Update the current node based on the new state
            for child in current_node.children:
                if child.state == state:
                    current_node = child
                    break

        # Print current player's turn
        print(f"{'Player A' if is_player_a_turn else 'Player B'}: {state}")
        is_player_a_turn = not is_player_a_turn

    # Print and return winner
    winner = "Player A" if not is_player_a_turn else "Player B"
    print(f"{winner} wins")
    return winner

def main():
    # Initialize graphs
    G1_1 = {
        'a': ['b', 'e'],
        'b': ['a', 'e'],
        'c': ['d', 'e'],
        'd': ['c', 'e'],
        'e': ['a', 'b', 'c', 'd']
    }
    
    G2_1 = {
        'a': ['b', 'd', 'e'],
        'b': ['a', 'c', 'e'],
        'c': ['b', 'd', 'e'],
        'd': ['a', 'c', 'e'],
        'e': ['a', 'b', 'c', 'd']
    }
    
    G3_1 = {
        'a': ['b', 'c', 'd'],
        'b': ['a', 'd', 'e'],
        'c': ['a', 'd'],
        'd': ['a', 'b', 'c', 'e'],
        'e': ['b', 'd']
    }
    
    BRIDGE = {
        'a': ['b', 'c', 'd'],
        'b': ['a', 'd'],
        'c': ['a', 'd'],
        'd': ['a', 'b', 'c']
    }
    
    P1_TEST = {
        'a': ['b', 'd'],
        'b': ['a', 'c', 'd', 'e'],
        'c': ['b', 'f'],
        'd': ['a', 'b', 'e', 'g'],
        'e': ['b', 'd', 'f', 'h'],
        'f': ['c', 'e', 'h', 'i'],
        'g': ['d', 'h'],
        'h': ['e', 'f', 'g', 'i'],
        'i': ['f', 'h']
    }
    
    G1_5 = {
        'a': ['b', 'c', 'e'],
        'b': ['a', 'c', 'e'],
        'c': ['a', 'b', 'e'],
        'e': ['a', 'b', 'c'],
        'd': ['c', 'e']
    }
    
    G2_5 = {
        'a': ['b'],
        'b': ['a', 'c', 'd'],
        'c': ['b', 'd'],
        'd': ['b']
    }
    
    G3_5 = {
        'a': ['b'],
        'b': ['a', 'c', 'g'],
        'c': ['b', 'd', 'e'],
        'd': ['c'],
        'e': ['c', 'f', 'g'],
        'f': ['e'],
        'g': ['b', 'e']
    }
    
    P2_TEST = {
        'a': ['b', 'c'],
        'b': ['a', 'c'],
        'c': ['a', 'b', 'f'],
        'd': ['e', 'f'],
        'e': ['d', 'f'],
        'f': ['c', 'd', 'e']
    }
    
    #=========== Problem 1
    #-- A
    print("========= Problem 1 =========")
    print("-- A --")
    
    # G1
    print("G1:")
    print(find_euler_circuit(G1_1))
    print()
    
    # G2
    print("G2:")
    print(find_euler_circuit(G2_1))
    print()
    
    # G3
    print("G3:")
    print(find_euler_circuit(G3_1))
    print()
    
    # Bridge of Koinsberg
    print("Bridge of Koingsberg:")
    print(find_euler_circuit(BRIDGE))
    print()
    
    #-- B
    print("-- B --")
    
    # Test Graph
    print("Test Graph:")
    print(find_euler_circuit(P1_TEST))
    print()
    
    #============ Problem 2
    print("========= Problem 2 =========")
    print("-- A --")
    
    # G1
    print("G1:")
    print(dirac(G1_5))
    print()
    
    # G2
    print("G2:")
    print(dirac(G2_5))
    print()
    
    # G3
    print("G3:")
    print(dirac(G3_5))
    print()
    
    #-- B
    print("-- B --")
    
    # Test Graph
    print("Test Graph:")
    print(dirac(P2_TEST))
    print()
    
    #============ Problem 3
    print("========= Problem 3 =========")
    print("-- A --")
    
    # G1
    print("G1:")
    print(ore(G1_5))
    print()
    
    # G2
    print("G2:")
    print(ore(G2_5))
    print()
    
    # G3
    print("G3:")
    print(ore(G3_5))
    print()
    
    #-- B
    print("-- B --")
    
    # Test Graph
    print("Test Graph:")
    print(ore(P2_TEST))
    print()
    
    #============= Problem 4
    print("========= Problem 4 =========")
    #-- A
    print("-- A --")
    # Set the initial state for debugging
    initial_debug_state = [2, 2, 1]

    # Simulate a single game with this initial state
    print("Debugging Game with Initial State [2, 2, 1]:")
    winner = play_game(initial_debug_state, first_player_minmax=True)
    print()
        
    #-- B
    print("-- B --")
    # Set the initial state for testing
    initial_test_state = [1, 2, 3]

    # Simulate a single game with this initial state
    print("Testing Game with Initial State [1, 2, 3]:")
    winner = play_game(initial_test_state, first_player_minmax=True)
    print()
    
    #-- C
    print("-- C --")
    # Initialize the players' wins
    player_a_wins = 0
    player_b_wins = 0
    # Run the game 100 times
    for game in range(100):
        print(f"\nGame {game + 1}:")
        print("Start: 2 2 1")
        winner = play_game([2, 2, 1], first_player_minmax=game % 2 == 0)
        
        # Increment the winner's win count
        if winner == "Player A":
            player_a_wins += 1
        else:
            player_b_wins += 1

    # Display the results
    print(f"\nPlayer A wins: {player_a_wins}")
    print(f"Player B wins: {player_b_wins}")
    
if __name__ == '__main__':
    main()