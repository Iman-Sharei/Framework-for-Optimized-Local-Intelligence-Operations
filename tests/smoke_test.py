"""Import smoke check for built wheel/sdist (used by publish-pypi CI)."""

from __future__ import annotations

import folio
from folio.cli import main


def main_smoke() -> None:
    assert folio.__version__
    assert callable(main)


if __name__ == "__main__":
    main_smoke()
    print("smoke_ok", folio.__version__)
