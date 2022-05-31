import pytest
from acids_transforms.utils.misc import import_data

@pytest.fixture
def test_files():
    return import_data('test/source_files', sr=44100)