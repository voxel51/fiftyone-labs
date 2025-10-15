import logging
from bs4 import BeautifulSoup

from fiftyone.utils.github import GitHubRepository
import fiftyone.plugins.utils as fopu

PLUGIN_METADATA_FILENAMES = ("fiftyone.yml", "fiftyone.yaml")

logger = logging.getLogger(__name__)


def list_labs_plugins(info=False):
    """Returns a list of available plugins registered in the
    `FiftyOne Labs repository <https://github.com/voxel51/labs>`_
    README.

    Args:
        info (False): whether to retrieve full plugin info for each plugin
            (True) or just return the available info from the README (False)

    Returns:
        a list of dicts describing the plugins
    """

    repo = GitHubRepository("https://github.com/voxel51/labs/tree/develop")
    content = repo.get_file("README.md").decode()
    soup = BeautifulSoup(content, "html.parser")

    plugins = []
    for row in soup.find_all("tr"):
        cols = row.find_all(["td"])
        if len(cols) != 2:
            continue

        try:
            name = cols[0].text.strip()
            url = cols[0].find("a")["href"]
            description = cols[1].text.strip()
            plugins.append(dict(name=name, url=url, description=description))
        except Exception as e:
            logger.debug("Failed to parse plugin row: %s", e)

    if not info:
        return plugins

    tasks = [(p["url"], None) for p in plugins]
    return fopu.get_plugin_info(tasks)
