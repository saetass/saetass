"""
Execute the tutorials and save them with their outputs, as they are committed.

Run it with the Python environment of SAETASS, which needs the ``tutorials`` extra and ``nbclient``::

    python tutorials/execute.py                       # every tutorial
    python tutorials/execute.py 03_running_simulations.ipynb

A tutorial that raises an error is not saved. ``test/test_tutorials.py`` checks the saved outputs.
"""

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient

FOLDER = Path(__file__).parent

#: Columns of code and output that fit the page of the documentation, the line length of Ruff.
COLUMNS = 88


class ThisPythonKernelManager(AsyncKernelManager):
    """Manager of the kernel of the Python that runs this script, whatever the kernels installed are."""

    @property
    def kernel_spec(self):
        command = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        return KernelSpec(argv=command, display_name="Python 3", language="python")


def execute(path: Path) -> None:
    """Execute the notebook of ``path`` in its folder and save it, keeping its metadata."""
    notebook = nbformat.read(path, as_version=4)
    metadata = {k: v for k, v in notebook.metadata.items() if k != "widgets"}
    NotebookClient(
        notebook,
        kernel_manager_class=ThisPythonKernelManager,
        timeout=1800,
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    notebook.metadata = nbformat.from_dict(metadata)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path, help="Default is all.")
    paths = parser.parse_args().notebooks or sorted(FOLDER.glob("*.ipynb"))
    os.environ["JUPYTER_COLUMNS"] = str(COLUMNS)
    os.environ.pop("MPLBACKEND", None)  # the figures need the backend of the notebooks
    for path in paths:
        execute(path if path.exists() else FOLDER / path)
        print(f"Executed {path.name}")


if __name__ == "__main__":
    main()
