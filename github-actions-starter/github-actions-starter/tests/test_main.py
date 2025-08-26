import subprocess, sys
from hello import greet


def test_greet_basic():
    assert greet("World") == "Hello, World!"


def test_greet_trims_and_defaults():
    assert greet("  Alice  ") == "Hello, Alice!"
    assert greet("") == "Hello, World!"


def test_cli_runs():
    res = subprocess.run([sys.executable, "-m", "hello", "--name", "CI"], capture_output=True, text=True)
    assert res.returncode == 0
    assert res.stdout.strip() == "Hello, CI!"
