# /usr/bin/python3
"""Solve the Dobble symbols distribution."""
from dobble.utils.asserts import assert_len
from dobble.utils.asserts import assert_same_keys


def get_n_cards(n_symbols_per_card: int) -> int:
    """Get the number of cards for a given number of symbols per card."""
    return n_symbols_per_card**2 - n_symbols_per_card + 1


def get_cards(n_symbols: int) -> list[list[int]]:
    """Get Cards.

    Given S = N^2 + N + 1 cards, find M cards with N symbols per card such that there's
    only one symbol in common between any two cards

    Could use backtracking, but the solution can be computed directly once you
    understand what's going on.
    """
    # The number of symbols on a card has to be a prime number + 1

# ----------------------------------------------------------

    cards = []
    # Work out the prime number
    n = n_symbols - 1

    # Total number of cards that can be generated following the Dobble rules
    number_of_cards = get_n_cards(n_symbols)

    # Add first set of n+1 cards (e.g. 8 cards)
    for i in range(n+1):
        # Add new card with first symbol
        cards.append([0])
        # Add n+1 symbols on the card (e.g. 8 symbols)
        for j in range(n):
            cards[i].append((j+1)+(i*n))

    # Add n sets of n cards
    for i in range(0, n):
        off = 0
        if n % 2 == 1 and i % (n/2) == 1:
            off = 1
        for j in range(0, n):
            # Append a new card with 1 symbol
            cards.append([i+1])
            # Add n symbols on the card (e.g. 7 symbols)
            for k in range(0, n):
                val = (n+1 + n*k + (i*k+j+off) % n)
                cards[len(cards)-1].append(val)

    # Assert cards are valid
    for k1, c1 in enumerate(cards):
        assert_len(c1, n+1)
        for k2, c2 in enumerate(cards):
            if k1 != k2:
                assert len(set(c1) & set(c2)) == 1, f"{set(c1)} and {set(c2)} => {set(c1) & set(c2)}"

    assert_same_keys([s for card in cards for s in card], range(number_of_cards))
    return cards
