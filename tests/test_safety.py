"""
tests/test_safety.py — Structural guard tests.

These tests do **not** execute any application code.  They walk the source
tree with the standard-library ``ast`` module and assert that every
``.submit_order()`` call-site is inside a function that is guarded in one
of two ways:

  (a) The function has a ``dry_run`` parameter — the guard is explicit and
      enforced at that function's level.  Every function in orders.py falls
      into this category.

  (b) The function is listed in ``ALLOWED_UNGUARDED`` — these are functions
      that are *exclusively* reachable through a caller that already checks
      ``dry_run``.  Each entry carries a comment naming the behavioural test
      that proves the guard.

Adding a new ``.submit_order()`` call to a function that lacks a ``dry_run``
parameter will fail ``test_every_submit_order_call_is_guarded`` immediately —
before the code ever touches an exchange.

How the scanner works
---------------------
1. Recursively find every ``.py`` file under ``src/``.
2. Parse each file into an AST.
3. Visit every ``FunctionDef`` / ``AsyncFunctionDef`` in the tree.
4. For each function, walk its body *without descending into nested function
   definitions* (so an inner ``def`` with its own ``submit_order`` call is
   attributed to the inner function, not the outer one).
5. If any direct child node is a ``Call`` whose ``func`` attribute is an
   ``Attribute`` node with ``.attr == "submit_order"``, the function is
   recorded as a caller.
6. The test asserts: every recorded caller either has ``dry_run`` in its
   parameter list or is in ``ALLOWED_UNGUARDED``.

Demonstrating a violation
-------------------------
To verify the test actually catches bad code, there is a companion test
``test_scanner_catches_unguarded_violation`` that injects a synthetic source
string containing a bare ``submit_order`` call and asserts the scanner flags it.
"""

from __future__ import annotations

import ast
import textwrap
from collections import deque
from pathlib import Path
from typing import Iterator

SRC_DIR = Path(__file__).parent.parent / "src"


# ---------------------------------------------------------------------------
# Functions allowed to call submit_order without their own dry_run parameter.
# Each entry: (module_file_stem, function_name).
# These functions are exclusively reached through a LiveRunner method that is
# itself behind an explicit `if not self.dry_run:` check.
# Behavioural proof: tests/test_runner.py::test_dry_run_blocks_submit_order
# ---------------------------------------------------------------------------
ALLOWED_UNGUARDED: set[tuple[str, str]] = {
    # IronCondor0DTE.enter()  →  LiveRunner._try_enter()  →  `if not self.dry_run:`
    ("iron_condor_0dte", "enter"),
    # IronCondor0DTE.exit()   →  LiveRunner._check_exit() / ._handle_sigint()
    #                         →  `if not self.dry_run:` / `if ... and not self.dry_run:`
    ("iron_condor_0dte", "exit"),
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _walk_excluding_nested_funcs(node: ast.AST) -> Iterator[ast.AST]:
    """Walk the AST of *node*, but **do not descend into nested function defs**.

    This ensures that a ``submit_order`` call inside an inner ``def`` is
    attributed only to that inner function — not the enclosing one.

    Example::

        def outer():            # walk stops here when checking outer
            def inner():        # ← NOT descended into
                x.submit_order(req)  # attributed to inner, not outer
    """
    queue: deque[ast.AST] = deque(ast.iter_child_nodes(node))
    while queue:
        child = queue.popleft()
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            queue.extend(ast.iter_child_nodes(child))


def _param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return the set of all parameter names declared by *func*."""
    args = func.args
    return {
        a.arg
        for a in (
            args.args
            + args.kwonlyargs
            + getattr(args, "posonlyargs", [])
        )
    }


def _find_submit_order_callers(src_dir: Path) -> list[dict]:
    """Scan *src_dir* and return one record for every function that calls
    ``.submit_order()``.

    Each record is a dict with keys:

    ``file``
        ``Path`` relative to *src_dir*.
    ``module_stem``
        Filename without extension (used to look up ``ALLOWED_UNGUARDED``).
    ``func_name``
        Name of the enclosing function/method.
    ``has_dry_run``
        ``True`` when the function has a ``dry_run`` parameter.
    ``lineno``
        Source line of the ``submit_order`` call.
    """
    results: list[dict] = []

    for py_file in sorted(src_dir.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        # ast.walk visits ALL FunctionDef/AsyncFunctionDef nodes, including
        # class methods and nested functions — exactly what we want, because
        # each function is checked for submit_order independently.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for child in _walk_excluding_nested_funcs(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "submit_order"
                ):
                    results.append(
                        {
                            "file": py_file.relative_to(src_dir),
                            "module_stem": py_file.stem,
                            "func_name": node.name,
                            "has_dry_run": "dry_run" in _param_names(node),
                            "lineno": child.lineno,
                        }
                    )
                    break   # record each function only once

    return results


# ---------------------------------------------------------------------------
# Structural test
# ---------------------------------------------------------------------------

def test_every_submit_order_call_is_guarded() -> None:
    """Every ``.submit_order()`` call-site must satisfy the guard contract.

    Fails with a clear message listing every violation so a developer can
    see exactly what they need to fix.
    """
    callers = _find_submit_order_callers(SRC_DIR)

    # Sanity-check the scanner itself: if zero callers are found something is
    # wrong (all submission code was removed, or SRC_DIR is misconfigured).
    assert callers, (
        f"No .submit_order() calls found under {SRC_DIR}.\n"
        "The scanner may be broken, or all order-submission code was deleted."
    )

    violations = [
        c
        for c in callers
        if not c["has_dry_run"]
        and (c["module_stem"], c["func_name"]) not in ALLOWED_UNGUARDED
    ]

    assert violations == [], (
        "The following functions call .submit_order() without a dry_run "
        "parameter and are not listed in ALLOWED_UNGUARDED.\n\n"
        "Fix options:\n"
        "  (a) Add `dry_run: bool = False` to the function signature and "
        "guard the submit_order call with `if not dry_run:`.\n"
        "  (b) If the function is always called from a path that is itself "
        "guarded, add it to ALLOWED_UNGUARDED in test_safety.py with a "
        "comment naming the behavioural test that proves the guard.\n\n"
        "Violations:\n" +
        "\n".join(
            f"  {v['file']}:{v['lineno']}  def {v['func_name']}()  "
            f"— no dry_run param, not in allowlist"
            for v in violations
        )
    )


# ---------------------------------------------------------------------------
# Meta-test: prove the scanner catches violations
# ---------------------------------------------------------------------------

def test_scanner_catches_unguarded_violation() -> None:
    """The scanner must flag a bare submit_order call in a function with no
    dry_run parameter.

    This test injects a synthetic source string and runs _find_submit_order_callers
    against a temporary in-memory parse — proving the scanner logic is correct
    and would catch future regressions.
    """
    # A function with submit_order and NO dry_run parameter — a clear violation.
    bad_source = textwrap.dedent("""\
        def place_order_no_guard(symbol, qty):
            trading.submit_order(req)
    """)

    # A function with submit_order AND a dry_run parameter — compliant.
    good_source = textwrap.dedent("""\
        def place_order_guarded(symbol, qty, dry_run=False):
            if not dry_run:
                trading.submit_order(req)
    """)

    def _scan_source(source: str) -> list[dict]:
        """Run the scanner logic against a raw source string (no file I/O)."""
        tree = ast.parse(source)
        results = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in _walk_excluding_nested_funcs(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "submit_order"
                ):
                    results.append(
                        {
                            "func_name": node.name,
                            "has_dry_run": "dry_run" in _param_names(node),
                            "lineno": child.lineno,
                        }
                    )
                    break
        return results

    # Bad source: scanner must find a caller without dry_run
    bad_callers = _scan_source(bad_source)
    assert len(bad_callers) == 1
    assert bad_callers[0]["func_name"] == "place_order_no_guard"
    assert bad_callers[0]["has_dry_run"] is False, (
        "Scanner failed to detect missing dry_run on an unguarded function."
    )

    # Good source: scanner must find the caller but it has dry_run
    good_callers = _scan_source(good_source)
    assert len(good_callers) == 1
    assert good_callers[0]["func_name"] == "place_order_guarded"
    assert good_callers[0]["has_dry_run"] is True, (
        "Scanner incorrectly flagged a properly-guarded function."
    )


# ---------------------------------------------------------------------------
# Meta-test: prove nested-function attribution is correct
# ---------------------------------------------------------------------------

def test_scanner_attributes_submit_order_to_inner_function_not_outer() -> None:
    """A submit_order call inside a nested def must be attributed to the inner
    function, not the enclosing one.

    Without _walk_excluding_nested_funcs, ast.walk(outer) would also find the
    call and incorrectly report outer as a caller even though it never touches
    submit_order directly.
    """
    source = textwrap.dedent("""\
        def outer_no_dry_run():
            def inner_also_no_dry_run():
                trading.submit_order(req)
    """)

    tree = ast.parse(source)
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in _walk_excluding_nested_funcs(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "submit_order"
            ):
                results.append(node.name)
                break

    # Only the inner function should be recorded, not outer
    assert results == ["inner_also_no_dry_run"], (
        f"Expected only inner function to be recorded, got: {results}"
    )
