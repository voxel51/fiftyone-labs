import logging
import re
from bs4 import BeautifulSoup

from fiftyone.utils.github import GitHubRepository
import fiftyone.plugins.utils as fopu
import fiftyone.plugins.core as fopc

PLUGIN_METADATA_FILENAMES = ("fiftyone.yml", "fiftyone.yaml")

logger = logging.getLogger(__name__)


def list_labs_features(info=False):
    """Returns a list of available feature plugins registered in the
    `FiftyOne Labs repository <https://github.com/voxel51/fiftyone-labs>`_
    README.

    Args:
        info (False): whether to retrieve full plugin info for each plugin
            (True) or just return the available info from the README (False)

    Returns:
        a list of dicts describing the features
    """

    repo = GitHubRepository("https://github.com/voxel51/fiftyone-labs")
    content = repo.get_file("README.md").decode()

    # Find h2 headings (##) in the readme
    h2_pattern = r"^## (.+)$"
    headings = []
    for match in re.finditer(h2_pattern, content, re.MULTILINE):
        headings.append(
            {"h2_heading": match.group(1), "h2_position": match.start()}
        )

    # Find tables in the readme
    table_pattern = r"<table>.*?</table>"
    tables = []
    for match in re.finditer(table_pattern, content, re.DOTALL):
        tables.append(
            {"table_content": match.group(0), "table_position": match.start()}
        )

    plugins = []
    for i, heading in enumerate(headings):
        heading_text = heading["h2_heading"]
        heading_pos = heading["h2_position"]

        next_heading_pos = (
            headings[i + 1]["h2_position"]
            if i + 1 < len(headings)
            else len(content)
        )

        for table in tables:
            if heading_pos < table["table_position"] < next_heading_pos:
                soup = BeautifulSoup(table["table_content"], "html.parser")
                table_elem = soup.find("table")

                for row in table_elem.find_all("tr"):
                    cols = row.find_all(["td"])
                    if len(cols) != 2:
                        continue

                    try:
                        name = cols[0].text.strip()
                        url = cols[0].find("a")["href"]
                        description = cols[1].text.strip()
                        plugins.append(
                            dict(
                                name=name,
                                url=url,
                                description=description,
                                category=heading_text,
                            )
                        )
                    except Exception as e:
                        logger.debug("Failed to parse plugin row: %s", e)

    if not info:
        return plugins

    return [fopu.get_plugin_info(p["url"], None) for p in plugins]


def add_version_info_to_features(lab_features):
    """Adds installation status and version information to each lab feature dicts.

    Args:
        lab_features: a list of dicts describing the features

    Returns:
        a list of dicts describing the features
    """
    curr_plugins_map = {
        pdef.name: pdef for pdef in fopc.list_plugins(enabled="all")
    }

    for p in lab_features:
        plugin_def = curr_plugins_map.get(p["name"])
        if plugin_def is None:
            # lab feature not installed
            p["status"] = "Not installed"
            p["curr_version"] = None
        else:
            p["status"] = "Installed"
            p["curr_version"] = plugin_def.version

    return lab_features
