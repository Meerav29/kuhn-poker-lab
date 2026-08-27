from kuhn import game


def test_terminal_histories():
    assert game.is_terminal("pp")
    assert game.is_terminal("bp")
    assert game.is_terminal("bb")
    assert game.is_terminal("pbp")
    assert game.is_terminal("pbb")
    assert not game.is_terminal("")
    assert not game.is_terminal("p")
    assert not game.is_terminal("b")
    assert not game.is_terminal("pb")


def test_whose_turn():
    assert game.whose_turn("") == 0
    assert game.whose_turn("p") == 1
    assert game.whose_turn("b") == 1
    assert game.whose_turn("pb") == 0


def test_payoff_table():
    # card1=2 (King) beats card2=0 (Jack) throughout
    assert game.payoff("pp", 2, 0) == 1     # showdown, pot=2, P1 wins
    assert game.payoff("pp", 0, 2) == -1    # showdown, pot=2, P1 loses
    assert game.payoff("bp", 0, 2) == 1     # P2 folded, P1 wins regardless of cards
    assert game.payoff("bp", 2, 0) == 1
    assert game.payoff("bb", 2, 0) == 2     # showdown, pot=4, P1 wins
    assert game.payoff("bb", 0, 2) == -2    # showdown, pot=4, P1 loses
    assert game.payoff("pbp", 2, 0) == -1   # P1 folded, P2 wins regardless of cards
    assert game.payoff("pbp", 0, 2) == -1
    assert game.payoff("pbb", 2, 0) == 2    # showdown, pot=4
    assert game.payoff("pbb", 0, 2) == -2


def test_payoff_consistent_across_all_deals():
    """Fold-histories pay a fixed 1 regardless of cards; showdown histories
    pay according to card comparison and pot size -- checked across every deal."""
    for c1, c2 in game.all_deals():
        assert game.payoff("bp", c1, c2) == 1
        assert game.payoff("pbp", c1, c2) == -1
        sign = 1 if c1 > c2 else -1
        assert game.payoff("pp", c1, c2) == sign * 1
        assert game.payoff("bb", c1, c2) == sign * 2
        assert game.payoff("pbb", c1, c2) == sign * 2


def test_infoset_key_format():
    assert game.infoset_key(0, "") == "0"
    assert game.infoset_key(2, "pb") == "2pb"


def test_all_infosets_count():
    # decision points: hist in {"", "p", "b", "pb"}, times 3 cards each = 12
    infosets = game.all_infosets()
    assert len(infosets) == 12
    assert len(set(infosets)) == 12  # no duplicates


def test_all_deals():
    deals = game.all_deals()
    assert len(deals) == 6  # 3 cards, ordered pairs, no repeats
    assert all(c1 != c2 for c1, c2 in deals)
