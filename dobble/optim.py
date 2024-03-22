# /usr/bin/python3
"""Solve the Dobble symbols distribution"""

from typing import List


def get_cards() -> List[List[int]]:
    """
    Given 57 symbols, find 57 cards with 8 symbols per card such that there's
    only one symbol in common between any two cards

    Could use backtracking, but the solution can be computed directly once you
    understand what's going on
    """
    cards = []

    # 0: 8 Rows
    base_cards = [[0] + [1+i + 7*k for i in range(7)]
                  for k in range(8)]
    cards += base_cards

    # 1: 7 Columns
    cards += [[1] + [base_cards[1+i][1+k] for i in range(7)]
              for k in range(7)]

    # 2: 7 Diag
    cards += [[2] + [base_cards[1+i][1+(i+k) % 7] for i in range(7)]
              for k in range(7)]

    # 3: 7 Anti-Diag
    cards += [[3] + [base_cards[1+i][1+(-i+k) % 7] for i in range(7)]
              for k in range(7)]

    # 2: 4 * 7 Modified Diag
    cards += [[4] + [base_cards[1+i][1+(2*i+k) % 7] for i in range(7)]
              for k in range(7)]
    cards += [[5] + [base_cards[1+i][1+(3*i+k) % 7] for i in range(7)]
              for k in range(7)]
    cards += [[6] + [base_cards[1+i][1+(5*i+k) % 7] for i in range(7)]
              for k in range(7)]
    cards += [[7] + [base_cards[1+i][1+(4*i+k+1) % 7] for i in range(7)]
              for k in range(7)]

    # Assert cards are valid
    for k1, c1 in enumerate(cards):
        assert len(c1) == 8
        for k2, c2 in enumerate(cards):
            if k1 != k2:
                assert len(set(c1) & set(c2)) == 1, \
                    f"{set(c1)} and {set(c2)} => {set(c1) & set(c2)}"

    assert set([s for card in cards for s in card]) == set(range(57))

    return cards
