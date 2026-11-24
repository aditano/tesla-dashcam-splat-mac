from __future__ import annotations

from pathlib import Path

from dashsplat import bootstrap as bs


def test_bootstrap_install_command_is_non_editable():
    """Finder/app-bundle installs must not use `pip install -e`."""
    src = Path(bs.__file__).read_text()
    assert 'install", "--python", str(py), "-e"' not in src
    assert 'f"{pkg_root}[{extras}]"' in src
    assert '-e", f"{pkg_root}' not in src
