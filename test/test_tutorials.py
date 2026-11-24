import json
import re
from pathlib import Path

import pytest

TUTORIALS = sorted((Path(__file__).parents[1] / "tutorials").glob("*.ipynb"))

#: Paths of the machine that executed a notebook, such as those of warnings and tracebacks.
LOCAL_PATH = re.compile(
    r"/(Users|home|root|private|tmp|var|opt)/|[A-Za-z]:\\+Users|site-packages|ipykernel_\d"
)


@pytest.fixture(params=TUTORIALS, ids=lambda path: path.stem)
def notebook(request):
    return json.loads(request.param.read_text(encoding="utf-8"))


def _outputs(notebook):
    """Number of each code cell with each of its outputs."""
    for number, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            yield number, output


def test_tutorials_are_found():
    assert TUTORIALS


def test_every_cell_was_executed_in_order(notebook):
    counts = [
        cell["execution_count"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert counts == list(range(1, len(counts) + 1))


def test_no_errors_or_warnings(notebook):
    for number, output in _outputs(notebook):
        assert output["output_type"] != "error", f"error in cell {number}"
        assert output.get("name") != "stderr", f"stderr in cell {number}"


def test_no_local_paths(notebook):
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            # Encoded images contain any text
            output.get("data", {}).pop("image/png", None)
    match = LOCAL_PATH.search(json.dumps(notebook, ensure_ascii=False))
    assert match is None, match.string[match.start() - 40 : match.end() + 80]


def test_no_widgets(notebook):
    # Static pages, such as the documentation and GitHub, do not render them
    assert "widgets" not in notebook["metadata"]
    for number, output in _outputs(notebook):
        mime_types = output.get("data", {})
        assert not any("widget" in mime_type for mime_type in mime_types), number


def test_text_fits_the_page_of_the_documentation(notebook):
    # Rich wraps its output at the 88 columns that tutorials/execute.py sets
    for number, output in _outputs(notebook):
        if "text/html" in output.get("data", {}):
            lines = "".join(output["data"]["text/plain"]).splitlines()
            lines = [re.sub(r"\x1b\[[\d;]*m", "", line) for line in lines]
            assert all(len(line) <= 88 for line in lines), number
