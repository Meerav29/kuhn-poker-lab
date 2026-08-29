from kuhn.bots.cfr import CFRTrainer
from kuhn.evaluate import exploitability


def test_average_strategy_is_valid_distribution():
    trainer = CFRTrainer(seed=0)
    bot, _ = trainer.train(iterations=1000)
    from kuhn import game
    for infoset in game.all_infosets():
        history = infoset[1:]
        legal = game.legal_actions(history)
        dist = bot.action_probs(infoset, legal)
        assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_exploitability_converges_toward_zero():
    trainer = CFRTrainer(seed=0)
    bot, _ = trainer.train(iterations=10000)
    # verified reference: iter=10000 -> exploitability ~= 0.0086 with this
    # exact algorithm/seed. Loose bound to tolerate minor implementation drift.
    assert exploitability(bot) < 0.05


def test_convergence_log_trends_downward():
    trainer = CFRTrainer(seed=0)
    _, log = trainer.train(iterations=10000, log_every=1000)
    assert len(log) == 10
    first_half_avg = sum(exp for _, exp in log[:5]) / 5
    second_half_avg = sum(exp for _, exp in log[5:]) / 5
    assert second_half_avg < first_half_avg
