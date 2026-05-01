# [ADD YOUR AGENT HERE]
from MiniPokerGame import BaseAgent, Card, NEW_VALUES, SUITS, evaluate_hand
import random

class BayesianAgent(BaseAgent):
    # This agent maintains a belief distribution over the opponent's possible cards and updates it using Bayesian inference based on the opponent's actions and the revealed table card. 
    # It then estimates the probability of winning against the opponent's hand and makes decisions accordingly.
    def __init__(self):
        self.deck = [Card(v, s) for v in NEW_VALUES for s in SUITS]
        self.belief = {}
        self._my_id = None          # 'P1' or 'P2', set once per hand

    # Belief initialisation
    def init_belief(self, my_card):
        """Uniform prior over all cards except my own."""
        self.belief = {}
        remaining = [c for c in self.deck if str(c) != str(my_card)]
        prob = 1.0 / len(remaining)
        for c in remaining:
            self.belief[str(c)] = prob

    # Bayesian update
    def update_belief(self, action, table_card=None):
        new_belief = {}
        for card_str, p in self.belief.items():
            opp_card = Card(card_str[0], card_str[1])
            rank = opp_card.value_rank()

            # Base likelihood from the action type
            if action == 'R2':
                base = 0.2 + 0.20 * rank
            elif action == 'R1':
                base = 0.5 + 0.10 * rank
            elif action in ('C', 'C1', 'C2'):
                base = 1.2 - 0.10 * rank
            elif action == 'F':
                base = 0.05 if rank < 2 else 0.15
            else:
                base = 1.0

            # After the flop, adjust likelihood using the board
            if table_card is not None:
                rank_tuple, _ = evaluate_hand(opp_card, table_card)
                if rank_tuple >= 2.5:           # pair or straight flush
                    factor = 1.5
                elif rank_tuple >= 1:           # straight or flush
                    factor = 1.2
                else:
                    factor = 0.8
                base *= factor

            new_belief[card_str] = p * max(base, 1e-6) # avoid zero probabilities (used for numerical stability)

        # Normalise
        total = sum(new_belief.values())
        for k in new_belief:
            new_belief[k] /= total
        self.belief = new_belief

    # Win probability estimation
    # This function gives us a win probability and our action is based on whether the probability crosses certain thresholds. We can adjust these thresholds to be more aggressive or more passive.
    def compute_win_prob(self, my_card, table_card):
        """Probability that my hand beats a random opponent hand from the belief."""
        win_prob = 0.0
        for opp_str, p_opp in self.belief.items():
            opp_card = Card(opp_str[0], opp_str[1])

            if table_card is None:
                # Pre‑flop: average over all possible future table cards
                wins = 0.0
                total = 0
                for t in self.deck: #No table card yet, so we consider all possibilities for it
                    if str(t) in (str(my_card), opp_str):
                        continue
                    h1 = evaluate_hand(my_card, t)
                    h2 = evaluate_hand(opp_card, t)
                    if h1 > h2:
                        wins += 1.0
                    elif h1 == h2:
                        wins += 0.5
                    total += 1
                p_win_given_opp = wins / total if total > 0 else 0.5
            else:
                # Post‑flop: directly compare hand strengths, table card is known so we don't need to itterate over possibilities.
                h1 = evaluate_hand(my_card, table_card)
                h2 = evaluate_hand(opp_card, table_card)
                if h1 > h2:
                    p_win_given_opp = 1.0
                elif h1 == h2:
                    p_win_given_opp = 0.5
                else:
                    p_win_given_opp = 0.0

            win_prob += p_opp * p_win_given_opp
        return win_prob

    # Decision making
    def act(self, card, table_card, stage, history):
        # Determine our identity once per hand
        if len(history) == 0:
            self._my_id = 'P1'
            self.init_belief(card)
        elif self._my_id is None:
            self._my_id = 'P2'
            self.init_belief(card)

        # Update belief from OPPONENT'S last action only
        if history:
            last_entry = history[-1]
            player, last_action = last_entry.split('-')
            if player != self._my_id:          # opponent's move
                self.update_belief(last_action, table_card)

        # Win probability
        p_win = self.compute_win_prob(card, table_card)

        # Helper: are we facing a raise?
        def facing_raise():
            if not history:
                return False
            # The last entry is the opponent's action (since we're called after they act)
            act = history[-1].split('-')[1]
            return act in ('R1', 'R2')

        # --- Decision ---
        if facing_raise():
            last_opp_action = history[-1].split('-')[1]
            call = 'C1' if last_opp_action == 'R1' else 'C2'

            if p_win > 0.45:          # call with any decent hand
                return call
            elif p_win < 0.20:        # only fold real trash
                return 'F'
            else:
                return random.choice([call, 'F'])
        else:
            if p_win > 0.65:          # raise large with premiums
                return 'R2'
            elif p_win > 0.40:        # raise small with average+ hands
                return 'R1'
            else:
                return 'C'