import logging
import markdown
from bs4 import BeautifulSoup

import fiftyone as fo
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

    repo = GitHubRepository("https://github.com/voxel51/labs")
    content = repo.get_file("README.md").decode()
    html_content = markdown.markdown(content, extensions=["tables"])
    heading_tables = _read_tables_from_html(html_content)

    plugins = []
    for heading in heading_tables:
        table = heading_tables[heading]
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
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
                        category=heading,
                    )
                )
            except Exception as e:
                logger.debug("Failed to parse plugin row: %s", e)

    if not info:
        return plugins

    tasks = [(p["url"], None) for p in plugins]
    return fopu.get_plugin_info(tasks)


def _read_tables_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    is_enterprise = hasattr(fo.constants, "TEAMS_VERSION")

    headings = soup.find_all("h2")
    heading_tables = {}

    for heading in headings:
        heading_text = heading.get_text()

        if not is_enterprise and "Enterprise" in heading_text:
            continue
        table = heading.find_next("table")

        next_heading = heading.find_next(["h1", "h2", "h3", "h4", "h5", "h6"])
        if table and (
            not next_heading or table.sourceline < next_heading.sourceline
        ):
            heading_tables[heading_text] = table

    return heading_tables
