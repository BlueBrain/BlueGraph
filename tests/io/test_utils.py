import math
from bluegraph.core.utils import *
from bluegraph.core.utils import _aggregate_values


def test_normalize_to_set():
    assert(
        normalize_to_set({1, 2, 3}) == {1, 2, 3})
    assert(
        normalize_to_set(1) == {1})
    assert(
        normalize_to_set(math.nan) == set())
    assert(
        normalize_to_set("lala") == {"lala"})
    assert(
        normalize_to_set("lala") == set(["lala"]))


def test_safe_intersection():
    assert(
        safe_intersection("lala", {"lala", 1}) == {"lala"})
    assert(
        safe_intersection("lala", "lala") == {"lala"})
    assert(
        safe_intersection("lala", math.nan) == set())


def test_str_to_set():
    assert(str_to_set("{'1', '2', '3'}") == {'1', '2', '3'})


def test_top_n():
    d = {
        "a": 4,
        "b": 5,
        "c": 1,
        "d": 6,
        "e": 10
    }
    assert(top_n(d, 3) == ['e', 'd', 'b'])
    assert(top_n(d, 3, smallest=True) == ["c", "a", "b"])


def test_aggregate_values():
    assert(
        _aggregate_values(["1", {"1", "2"}, {"2", "3"}]) ==
        {"1", "2", "3"})


def test_element_has_type():
    assert(element_has_type({"M", "N", "O"}, {"M", "N"}))
    assert(not element_has_type({"M", "N", "O"}, {"M", "K"}))
