import nox
from nox_poetry import session


nox.options.sessions = "lint", "tests"


@session(python=["3.11", "3.10"])
def tests(session):
    session.install("pytest", "pytest-cov", "pytest-datadir", ".")
    session.run("pytest", "--cov")


locations = "src", "tests", "noxfile.py"


@session(python="3.11")
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black", "flake8-bugbear", "flake8-import-order")
    session.run("flake8", *args)


@session(python="3.11")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
