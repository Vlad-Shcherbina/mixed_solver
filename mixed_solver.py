"""
Mixed game solver.
Method is described in GAME THEORY by Thomas S. Ferguson, p.42
(http://www.math.ucla.edu/~tom/Game_Theory/mat.pdf)
"""

import numpy


def pivot_transform(a, i, j):
    # p r              1/p       r/p
    # c q   becomes   -c/p  q - rc/p
    m, n = a.shape

    pivot = a[i, j]
    row = a[i].copy()
    column = a[:, j].copy()

    a -= row * column.reshape((m, 1)) / pivot
    a[i] = row / pivot
    a[:, j] = -column / pivot
    a[i, j] = 1.0 / pivot


def solve(a):
    """
    Solve matrix game and return triple (game_value, strategy1, strategy2)

    >>> solve(numpy.array([[0, 1], [1, 0]]))
    (0.5, array([ 0.5,  0.5]), array([ 0.5,  0.5]))
    """

    m, n = a.shape

    delta = 1 - a.min()

    tableau = numpy.zeros((m+1, n+1))
    tableau[:m, :n] = a + delta
    tableau[m, :n] = -1
    tableau[:m, n] = 1

    row_labels = range(m)
    col_labels = range(-n, 0)

    while True:
        q = numpy.argmin(tableau[m, :n])
        if tableau[m, q] > -1e-10:
            break

        ps = numpy.where(tableau[:m, q] > 0)[0]
        p = ps[numpy.argmin(tableau[ps, n] / tableau[ps, q])]

        pivot_transform(tableau, p, q)
        row_labels[p], col_labels[q] = col_labels[q], row_labels[p]

    game_value = 1.0 / tableau[m, n] - delta
    strategy1 = numpy.zeros((m,))
    for i, label in enumerate(col_labels):
        if label >= 0:
            strategy1[label] = tableau[m, i] / tableau[m, n]
    strategy2 = numpy.zeros((n,))
    for i, label in enumerate(row_labels):
        if label < 0:
            strategy2[label+n] = tableau[i, n] / tableau[m, n]
    return game_value, strategy1, strategy2


def check_solution(a, game_value, strategy1, strategy2):
    eps = 1e-6
    m, n = a.shape
    assert len(strategy1) == m
    assert len(strategy2) == n
    assert abs(sum(strategy1) - 1) < eps
    assert abs(sum(strategy2) - 1) < eps
    assert all(strategy1 > -eps)
    assert all(strategy2 > -eps)

    for i in range(n):
        assert numpy.dot(strategy1, a[:, i]) >= game_value - eps
    for i in range(m):
        assert numpy.dot(strategy2, a[i, :]) <= game_value + eps

    assert abs(numpy.dot(strategy1, numpy.dot(a, strategy2)) - game_value) < eps


def solve_and_check(a):
    print '---'
    print a
    sol = solve(a)
    print sol
    check_solution(a, *sol)


if __name__ == '__main__':
    import doctest
    result = doctest.testmod()
    print 'doctest:', result
    assert result.failed == 0

    solve_and_check(numpy.array([[0]]))
    solve_and_check(numpy.array([[0], [1]]))
    solve_and_check(numpy.array([[1, 1]]))
    solve_and_check(numpy.array([
        [2, 0],
        [0, 1]]))
    solve_and_check(numpy.array([
        [2, 0],
        [0, 1],
        [-1, -1]]))
    solve_and_check(numpy.array([
        [1, 0, 0.6],
        [0, 2, 0.6]]))

    import random
    random.seed(42)
    for m, n in [(50, 200), (200, 50), (100, 100)]:
        a = numpy.zeros((m, n))
        for i in range(m):
            for j in range(n):
                a[i, j] = random.random()
        solve_and_check(a)
    print 'all ok'
