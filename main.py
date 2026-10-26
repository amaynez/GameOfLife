"""Compatibility shim; use `gameoflife` console script or `python -m gameoflife.cli`."""

from gameoflife.cli import main


if __name__ == "__main__":
    main()
