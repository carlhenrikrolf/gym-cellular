import numpy as np


def cellular2tabular(alist, intracellular_size, n_cells):

    assert len(alist) == n_cells
    assert max(alist) < intracellular_size
    assert min(alist) >= 0

    anint = sum([(alist[cell] * intracellular_size ** cell) for cell in range(n_cells)])
    return int(anint)


def tabular2cellular(anint, intracellular_size, n_cells):

    assert anint < intracellular_size ** n_cells
    assert anint >= 0

    alist = np.zeros(n_cells, dtype=int)
    for cell in range(n_cells):
        alist[cell] = anint % intracellular_size
        anint = anint // intracellular_size
    return alist


def pruned_cellular2tabular(alist, cellular_state_space):
    
    assert len(alist) == len(cellular_state_space)

    n_cells = len(cellular_state_space)
    anint = 0
    place_value = 1
    for cell in range(n_cells):
        anint += (alist[cell] - min(cellular_state_space[cell])) * place_value
        place_value *= len(cellular_state_space[cell])
    return anint

def pruned_tabular2cellular(anint, cellular_state_space):

    n_cells = len(cellular_state_space)
    alist = np.zeros(n_cells, dtype=int)
    for cell in range(n_cells):
        alist[cell] = anint % len(cellular_state_space[cell])
        alist[cell] += min(cellular_state_space[cell])
        anint = anint // len(cellular_state_space[cell])
    return alist