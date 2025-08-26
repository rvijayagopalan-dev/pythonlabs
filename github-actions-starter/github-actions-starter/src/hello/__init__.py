__all__ = ["greet"]

def greet(name: str) -> str:
    """Return a friendly greeting for *name*.

    >>> greet("World")
    'Hello, World!'
    """
    name = (name or "").strip() or "World"
    return f"Hello, {name}!"
