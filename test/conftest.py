import pytest


def pytest_addoption(parser):
    """Adds command-line options to pytest."""
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="show plots for visual inspection",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="update/generate golden dataset on disk",
    )


@pytest.fixture
def plot_results(request):
    """Fixture that returns True if --plot was passed, False otherwise."""
    return request.config.getoption("--plot")


@pytest.fixture
def update_golden(request):
    """Fixture that returns True if --update-golden was passed, False otherwise."""
    return request.config.getoption("--update-golden")


@pytest.fixture
def saetass_caplog(caplog):
    """``caplog`` receiving the log messages of SAETASS, which its Rich handler keeps by default."""
    from saetass.cli import set_rich_logging

    set_rich_logging(False)
    yield caplog
    set_rich_logging(True)
