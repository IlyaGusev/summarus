import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data")
TEST_URLS_FILE = os.path.join(DATA_DIR, "urls.txt")
TEST_CONFIG = os.path.join(DATA_DIR, "config.json")
TEST_STORIES_DIR = os.path.join(DATA_DIR, "stories")
