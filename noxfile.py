"""Nox sessions."""

import nox
from nox_poetry import session


nox.options.sessions = "lint", "mypy", "tests", "docs"


@session(python=["3.11", "3.10"])
def tests(session):
    """Run the test suite."""
    session.install("pytest", "pytest-cov", "pytest-datadir", ".")
    session.run("pytest", "--cov")


locations = "src", "tests", "noxfile.py", "docs/conf.py"


@session(python="3.11")
def lint(session):
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "pydoclint[flake8]",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@session(python="3.11")
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@session(python=["3.11", "3.10"])
def mypy(session):
    """Type-check using mypy."""
    args = session.posargs or locations
    session.install("mypy", "pandas-stubs")
    session.run("mypy", *args)


@session(python="3.11")
def docs(session):
    """Build the documentation."""
    session.install("sphinx", "sphinx-autodoc-typehints", ".")
    session.run("sphinx-build", "docs", "docs/_build")


@session(python="3.11")
def coverage(session):
    """Upload coverage data."""
    session.install("coverage[toml]", "codecov")
    session.run("coverage", "xml")
    session.run("codecov", *session.posargs)
