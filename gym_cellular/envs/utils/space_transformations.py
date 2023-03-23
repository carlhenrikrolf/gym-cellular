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


def pruned_cellular2tabular(alist, intracellular_space_set):
    
    assert len(alist) == len(intracellular_space_set)

    n_cells = len(intracellular_space_set)
    anint = 0
    place_value = 1
    for cell in range(n_cells):
        shift = (alist[cell] - min(intracellular_space_set[cell]))
        anint += shift * place_value
        place_value *= len(intracellular_space_set[cell])
    return anint


def pruned_tabular2cellular(anint, intracellular_space_set):
    
    n_cells = len(intracellular_space_set)
    alist = np.zeros(n_cells, dtype=int)
    for cell in range(n_cells):
        alist[cell] = anint % len(intracellular_space_set[cell])
        alist[cell] += min(intracellular_space_set[cell])
        anint = anint // len(intracellular_space_set[cell])
    return alist