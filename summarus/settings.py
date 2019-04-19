import os
from pkg_resources import resource_filename

TESTS_DIR = resource_filename(__name__, "tests")
DATA_DIR = os.path.join(TESTS_DIR, "data")
TEST_URLS_FILE = os.path.join(DATA_DIR, "urls.txt")
TEST_CONFIG_DIR = os.path.join(TESTS_DIR, "configs")
TEST_STORIES_DIR = os.path.join(DATA_DIR, "stories")
RIA_EXAMPLE_FILE = os.path.join(DATA_DIR, "ria_20.json")
