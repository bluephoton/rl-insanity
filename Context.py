from munch import munchify
from Insanity.Initializers import OptimisticInitializer, ZeroInitializer
from Insanity.Selectors import EpsilonGreedyActionSelector, UCBActionSelector, GradientSelector
from ActionValuesProvider import ActionValuesProvider
from UCBActionValuesProvider import UCBActionValuesProvider
from GradientActionValuesProvider import GradientActionValuesProvider

ctxs = {
    1: munchify({
        "num_arms": 10,
        "num_runs": 200,
        "num_pulls_per_run": 1000,
        "ms": [0, 0, 0], #µs
        "σs": [1, 1, 1],
        "εs": [0, 0.01, 0.1],
        "initializers": [lambda : ZeroInitializer() for _ in range(3)],
        "providers": [lambda a, b: ActionValuesProvider(a, b) for _ in range(3)],
        "selectors": [lambda a, b: EpsilonGreedyActionSelector(a, b) for _ in range(3)],
        "dont_average_reward": False
    }),

    2: munchify({
        "num_arms": 10,
        "num_runs": 2000,
        "num_pulls_per_run": 1000,
        "ms": [0, 0], #µs
        "σs": [1, 1],
        "εs": [0, 0.1],
        "initializers": [lambda : OptimisticInitializer(5, 0.1), lambda : ZeroInitializer()],
        "providers": [lambda a, b: ActionValuesProvider(a, b) for _ in range(2)],
        "selectors": [lambda a, b: EpsilonGreedyActionSelector(a, b) for _ in range(2)],
        "dont_average_reward": False
    }),

    3: munchify({
        "num_arms": 10,
        "num_runs": 1000,
        "num_pulls_per_run": 1000,
        "ms": [0, 0], #µs
        "σs": [1, 1],
        "εs": [0.1, 0],
        "initializers": [lambda : ZeroInitializer() for _ in range(2)],
        "providers": [lambda a, b: ActionValuesProvider(a, b), lambda a, b: UCBActionValuesProvider(a, b, 2)],
        "selectors": [lambda a, b: EpsilonGreedyActionSelector(a, b), lambda a, b: UCBActionSelector()],
        "dont_average_reward": True
    }),

    4: munchify({
        "num_arms": 10,
        "num_runs": 1000,
        "num_pulls_per_run": 1000,
        "ms": [4, 4, 4, 4], #µs
        "σs": [1, 1, 1, 1],
        "εs": [0, 0, 0, 0],
        "initializers": [lambda : ZeroInitializer() for _ in range(4)],
        "providers": [
            lambda a, b: GradientActionValuesProvider(a, b, 0.1),
            lambda a, b: GradientActionValuesProvider(a, b, 0.4),
            lambda a, b: GradientActionValuesProvider(a, b, 0.1, True),
            lambda a, b: GradientActionValuesProvider(a, b, 0.4, True)
        ],
        "selectors": [lambda a, b: GradientSelector() for _ in range(4)],
        "dont_average_reward": False
    })
}

active_context = 0

def ctx():
    return ctxs[active_context]

def use_context(i):
    global active_context
    active_context = i

