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