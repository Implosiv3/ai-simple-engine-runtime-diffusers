import pytest


@pytest.fixture(scope = 'session', autouse = True)
def setup_and_teardown_session(
    request
):
    """
    Method to remove the folder for temporary files when
    the testing process has finished.
    """
    # Code to run at the begining

    yield
    
    # Code to run after all tests have finished