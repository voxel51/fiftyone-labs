import os

import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.utils.github import GitHubRepository

from .utils import list_labs_plugins


class LabsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="labs_panel",
            label="FiftyOne Labs",
        )

    def on_load(self, ctx):
        ctx.panel.state.logo = "https://github.com/voxel51/labs/blob/develop/plugins/labs_panel/assets/labs_logo.png"

        plugins = list_labs_plugins()
        ctx.panel.state.table = plugins

    def on_plugin_select(self, ctx):
        pass

    def render(self, ctx):
        panel = types.Object()

        panel.md(
            "# FiftyOne Labs",
            name="labs_header",
        )

        image_holder = types.ImageView()
        panel.view("logo", view=image_holder)

        panel.md(
            "_Machine Learning research solutions and experimental features_",
            name="labs_subtitle",
        )

        # List of all the Labs plugins
        table = types.TableView()
        table.add_column("name", label="Plugin")
        table.add_column("description", label="Description")
        table.add_column("url", label="URL")
        table.add_column("image")
        panel.list("table", types.Object(), view=table)

        plugins = ctx.panel.get_state("table")
        for idx, p in enumerate(plugins):
            repo = GitHubRepository(p["url"])
            content = repo.get_file("README.md").decode()
            panel.md(content, name=f"readme_{idx}")

        return types.Property(
            panel,
            view=types.ObjectView(
                align_x="center", align_y="center", orientation="vertical"
            ),
        )


def register(p):
    p.register(LabsPanel)
