def generalized_cellular2tabular(alist, intracellular_space_set):
    
    assert len(alist) == len(intracellular_space_set)

    n_cells = len(intracellular_space_set)
    anint = 0
    place_value = 1
    for cell in range(n_cells):
        shift = (alist[cell] - min(intracellular_space_set[cell]))
        anint += shift * place_value
        place_value *= len(intracellular_space_set[cell])
    return anint


def generalized_tabular2cellular(anint, intracellular_space_set):
    
    n_cells = len(intracellular_space_set)
    alist = [0] * n_cells
    for cell in range(n_cells):
        alist[cell] = anint % len(intracellular_space_set[cell])
        alist[cell] += min(intracellular_space_set[cell])
        anint = anint // len(intracellular_space_set[cell])
    return alist