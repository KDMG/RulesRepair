from enum import Enum
from dataclasses import dataclass
from typing import FrozenSet


class Operator(str, Enum):
    EQ = "=="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    NE = "!="


@dataclass(frozen=True)
class Condition:
    attribute: str
    operator: Operator
    threshold: float

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.attribute} {self.operator} {self.threshold}"


@dataclass(frozen=True)
class Rule:
    label: str
    conditions: FrozenSet[Condition]

    def __getitem__(self, key):
        return self.conditions[key]

    def __str__(self):
        str_rep = " ˄ ".join([str(c) for c in self.conditions])
        str_rep += f" → {self.label}"
        return str_rep

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Ruleset:
    rules: FrozenSet[Rule]

    def __getitem__(self, key):
        return self.rules[key]

    def __str__(self):
        return "\n".join([f"r{i}: {str(r)}" for i, r in enumerate(self.rules)])

    def __repr__(self):
        return str(self)


def extract_substructure(rule: Rule):
    # Substructure span from the root node
    length = len(rule.conditions)
    return [rule.conditions[:i] for i in range(1, length + 1)]


def create_substructures(ruleset: Ruleset):
    return [(rule, extract_substructure(rule)) for rule in ruleset]


def condition_similarity(condition1: Condition, condition2: Condition):
    if condition1.attribute != condition2.attribute:
        return 0
    if condition1.operator != condition2.operator:
        return 0
    if condition1.threshold != condition2.threshold:
        return 0
    return 1


def substructure_similarity(rule1: Rule, rule2: Rule, substructure):
    s = len(substructure)
    sim_k_sum = 0
    for index, sub in enumerate(substructure):
        if index >= len(rule1.conditions) or index >= len(rule2.conditions):
            return 0

        sim_cond = condition_similarity(rule1[index], rule2[index])

        if sim_cond == 0:
            # break
            return 0

        sim_k_sum += sim_cond
    return sim_k_sum / s


# Symmetric
def rule_similarity(rule1: Rule, subs1, rule2: Rule, subs2):
    n = max(len(subs1), len(subs2))

    if n == 0:
        return 1.0

    subs = subs1 if len(subs1) > len(subs2) else subs2

    sim_ij_sum = sum([substructure_similarity(rule1, rule2, sub) for sub in subs])
    return sim_ij_sum / n


def rule_set_similarity(ruleset1: Ruleset, ruleset2: Ruleset):
    substructure1 = create_substructures(ruleset1)
    substructure2 = create_substructures(ruleset2)

    l = len(ruleset1.rules)

    sim_d_list = []
    for rule1, subs1 in substructure1:
        rule_sims = []

        for rule2, subs2 in substructure2:
            rule_sims.append(rule_similarity(rule1, subs1, rule2, subs2))

        sim_d_list.append(max(rule_sims))

    return sum(sim_d_list) / l


def rule_similarity_labeled(rule1: Rule, subs1, rule2: Rule, subs2):
    if rule1.label != rule2.label:
        return 0.0
    return rule_similarity(rule1, subs1, rule2, subs2)


def rule_set_similarity_labeled(ruleset1: Ruleset, ruleset2: Ruleset):
    substructure1 = create_substructures(ruleset1)
    substructure2 = create_substructures(ruleset2)

    l = len(ruleset1.rules)

    sim_d_list = []
    for rule1, subs1 in substructure1:
        rule_sims = []

        for rule2, subs2 in substructure2:
            rule_sims.append(rule_similarity_labeled(rule1, subs1, rule2, subs2))

        sim_d_list.append(max(rule_sims))

    return sum(sim_d_list) / l


def jaccard_rule_set_similarity(ruleset1: Ruleset, ruleset2: Ruleset):
    def as_set(ruleset):
        return {
            (tuple((c.attribute, c.operator, c.threshold) for c in rule.conditions), rule.label)
            for rule in ruleset.rules
        }

    set1 = as_set(ruleset1)
    set2 = as_set(ruleset2)
    union = set1 | set2
    if not union:
        return 1.0
    return len(set1 & set2) / len(union)
