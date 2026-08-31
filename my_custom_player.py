"""
CustomPlayer for the knights Isolation project.

This implements:
  - Minimax search with alpha-beta pruning
  - Iterative deepening (searches deeper and deeper until time runs out,
    always keeping the best move found so far in self.queue)
  - A simple "#my_liberties - #opponent_liberties" evaluation function
    to use as your baseline heuristic

NOTE: This is written against the standard AIND/Udacity `isolation` library
interface (state.actions(), state.result(action), state.terminal_test(),
state.utility(player_id), state.ply_count, state.locs, state.liberties()).
Open the isolation.py file included in your starter code and confirm these
method names match — some course versions differ slightly (e.g. some use
state.player() instead of parity on ply_count). Adjust names if needed.
"""

import random


class SearchTimeout(Exception):
    """Raised when the search runs out of time."""
    pass


class CustomPlayer:
    """
    Fill this in per your starter file's exact constructor signature —
    most versions auto-populate self.player_id, self.queue, self.data,
    and self.context for you, and give you a self.time_left() callable.
    Shown here for clarity in case you need to wire it yourself.
    """

    def __init__(self, player_id):
        self.player_id = player_id
        self.queue = None          # output queue: self.queue.put(action)
        self.data = None           # loaded from data.pickle if present
        self.context = None        # persists across turns of ONE game
        self.time_left = None      # callable -> ms remaining (set by caller)
        self.TIMER_THRESHOLD = 10  # ms safety buffer before forfeiting

    # ------------------------------------------------------------------
    # Entry point called once per turn
    # ------------------------------------------------------------------
    def get_action(self, state):
        # Always queue *some* legal move immediately in case we run out
        # of time before the first full ply completes.
        self.queue.put(random.choice(state.actions()))

        depth = 1
        try:
            while True:
                best_move = self.alpha_beta_search(state, depth)
                if best_move is not None:
                    self.queue.put(best_move)
                depth += 1
        except SearchTimeout:
            # Time's up — whatever we already queued is our final answer.
            pass

    # ------------------------------------------------------------------
    # Alpha-beta search, depth-limited (called repeatedly with increasing
    # depth by iterative deepening above)
    # ------------------------------------------------------------------
    def alpha_beta_search(self, state, depth):
        self._check_time()

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        for action in state.actions():
            v = self._min_value(state.result(action), depth - 1, alpha, beta)
            if v > best_score:
                best_score = v
                best_move = action
            alpha = max(alpha, best_score)

        return best_move

    def _min_value(self, state, depth, alpha, beta):
        self._check_time()

        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)

        v = float("inf")
        for action in state.actions():
            v = min(v, self._max_value(state.result(action), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _max_value(self, state, depth, alpha, beta):
        self._check_time()

        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)

        v = float("-inf")
        for action in state.actions():
            v = max(v, self._min_value(state.result(action), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _check_time(self):
        if self.time_left is not None and self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    # ------------------------------------------------------------------
    # Baseline heuristic: #my_liberties - #opponent_liberties
    # (This is the lecture heuristic — fine as your Option 2 baseline,
    # but remember it doesn't count as your "custom" contribution.)
    # ------------------------------------------------------------------
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)