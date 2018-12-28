import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data")
TEST_URLS_FILE = os.path.join(DATA_DIR, "urls.txt")
TEST_CONFIG = os.path.join(DATA_DIR, "test_config.json")
DEFAULT_CONFIG = os.path.join(DATA_DIR, "default_config.json")
TEST_STORIES_DIR = os.path.join(DATA_DIR, "stories")
