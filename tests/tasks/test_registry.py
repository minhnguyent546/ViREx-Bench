"""Tests for the task registry — name resolution and discovery."""

import pytest

from virex_bench.tasks import get_task, list_tasks
from virex_bench.tasks.base import ReasoningTask


def test_list_tasks_is_sorted_and_contains_canonical_task() -> None:
    names = list_tasks()
    assert names == sorted(names)
    assert "vietnamese-logical-reasoning" in names


def test_get_task_returns_registered_task_instance() -> None:
    task = get_task("vietnamese-logical-reasoning")
    assert isinstance(task, ReasoningTask)
    assert task.name == "vietnamese-logical-reasoning"


def test_every_listed_task_resolves() -> None:
    # Every name surfaced by list_tasks() must construct via get_task.
    for name in list_tasks():
        assert isinstance(get_task(name), ReasoningTask)


def test_get_task_unknown_name_raises_keyerror_with_available_list() -> None:
    with pytest.raises(KeyError, match="Unknown task 'does-not-exist'"):
        get_task("does-not-exist")
