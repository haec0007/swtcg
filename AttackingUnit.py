from copy import copy
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from scipy.stats import binom


def compositions(n, k):
    if n < 0 or k < 0:
        return None
    elif k == 0:
        # the empty sum, by convention, is zero, so only return something if
        # n is zero
        if n == 0:
            yield []
        return None
    elif k == 1:
        yield [n]
        return None
    else:
        for i in range(0, n+1):
            for comp in compositions(n - i, k - 1):
                yield [i] + comp


class AttackingUnit:
    def __init__(self, power, critical_hit=0, accuracy=0, lucky=0):
        self.power = power
        self.critical_hit = critical_hit
        self.accuracy = accuracy
        self.lucky = lucky

    def crit_prob(self):
        if self.critical_hit > 0:
            return 1 - (5/6) ** self.power
        else:
            return 0

    def damage_dist_old(self):
        """
        This method only considers Accuracy and Critical Hit.
        It does not take into account Lucky.
        """
        hit_prob = min(1.0, max(0.0, (3 + self.accuracy) / 6))
        dist = [
            binom.pmf(h, self.power, hit_prob) * (1 - 1/6 / hit_prob) ** h +
            binom.pmf(h - self.critical_hit, self.power, hit_prob) *
            (1 - (1 - 1/6 / hit_prob) ** (h - self.critical_hit))
            for h in range(self.power + self.critical_hit + 1)
        ]
        return dist

    def damage_dist(self):
        if self.critical_hit > 0:
            crit_prob = 1/6
        else:
            crit_prob = 0.0
        hit_prob = min(1.0, max(0.0, (3 + self.accuracy) / 6)) - crit_prob
        miss_prob = 1 - hit_prob - crit_prob

        dist = [0] * (self.power + self.critical_hit + 1)
        for hits in range(self.power + 1):
            for sixes in range(self.power - hits + 1):
                misses = self.power - hits - sixes
                combos = comb(self.power, hits) * comb(self.power - hits, sixes) * comb(self.power - hits - sixes, misses)
                prob = combos * hit_prob ** hits * crit_prob ** sixes * miss_prob ** misses

                initial_roll = {"hit": hits, "miss": misses, "critical_hit": sixes}
                max_lucky_damage_dist = [0] * (self.power + self.critical_hit + 1)
                max_expected_damage = -1
                for lucky_dice in range(self.lucky + 1):
                    for c in compositions(lucky_dice, len(initial_roll.keys())):
                        if (
                            initial_roll['hit'] >= c[0]
                            and initial_roll['miss'] >= c[1]
                            and initial_roll['critical_hit'] >= c[2]
                        ):
                            rerolls = {"hit": c[0], "miss": c[1], "critical_hit": c[2]}
                            lucky_damage_dist, expected_damage = self.reroll(initial_roll, rerolls)
                            if expected_damage > max_expected_damage:
                                max_lucky_damage_dist = lucky_damage_dist
                                max_expected_damage = expected_damage

                #if sixes > 0:
                #    damage = hits + sixes + self.critical_hit
                #else:
                #    damage = hits + sixes
                for i in range(len(max_lucky_damage_dist)):
                    dist[i] += prob * max_lucky_damage_dist[i]
        return dist

    def plot_dist(self):
        hit_prob = self.damage_dist()
        hits = list(range(len(hit_prob)))

        plt.bar(hits, hit_prob)
        plt.show()

    def expected_damage(self):
        damage_dist = self.damage_dist()
        return sum([i * damage_dist[i] for i in range(len(damage_dist))])

    def reroll(self, initial_roll, rerolls):
        """
        :param initial_roll: dictionary containing the number dice in each category,
        e.g. {'hit': 2, 'miss': 1, 'critical_hit': 1}. Note that each category is disjoint,
        that is, hits are only non-critical hits.
        :param rerolls: dictionary containing the number of dice in each category to re-roll.
        Same format as `initial_roll`
        :return: list of damage distribution, expected damage
        """
        # Sanity checks
        if sum(initial_roll.values()) != self.power:
            raise Exception("Number of dice rolled does not equal unit's power.")
        if sum(rerolls.values()) > self.power:
            raise Exception("Number of dice to re-roll exceeds unit's power.")
        for k in rerolls.keys():
            if rerolls[k] > initial_roll[k]:
                raise Exception(f"Number of {k} re-rolls exceeds initial roll.")

        reroll_unit = copy(self)
        reroll_unit.power = sum(rerolls.values())
        kept_dice = {k: max(initial_roll[k] - rerolls.get(k, 0), 0) for k in initial_roll.keys()}
        kept_damage = kept_dice.get('hit', 0) + kept_dice.get('critical_hit', 0)
        if kept_dice.get('critical_hit', 0) > 0:
            kept_damage = kept_damage + self.critical_hit
            reroll_unit.critical_hit = 0

        damage_dist = ([0] * kept_damage) + reroll_unit.damage_dist_old()
        expected_damage = sum([i * damage_dist[i] for i in range(len(damage_dist))])
        return damage_dist, expected_damage

    def reroll_sim(self, initial_roll, reroll_pos, sims=10000):
        """
        :param initial_roll: list of dice results, e.g. [4,5,2,3,6,1,1]
        :param reroll_pos: the 0-indexed positions of dice to reroll in the initial_roll list
        :param sims: number of simulations to run
        :return: list of damage distribution, expected damage
        """
        kept_dice = np.array([initial_roll[i] for i in range(len(initial_roll)) if i not in reroll_pos])
        kept_hits = (kept_dice > 3 - self.accuracy).sum()

        sim_results = np.hstack((np.tile(kept_dice, (sims, 1)), randint(1, 7, (sims, len(reroll_pos)))))
        damage = (sim_results > 3 - self.accuracy).sum(axis=1) + (sim_results == 6).any(axis=1) * self.critical_hit

        damage_dist = np.divide(
            np.histogram(damage, bins=[i for i in range(kept_hits + len(reroll_pos) + self.critical_hit + 2)])[0],
            sims
        )
        expected_damage = sum([i * damage_dist[i] for i in range(len(damage_dist))])
        return damage_dist, expected_damage
