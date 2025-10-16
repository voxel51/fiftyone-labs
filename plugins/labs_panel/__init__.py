import os

import fiftyone.operators as foo
import fiftyone.plugins as fop
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
        ctx.panel.state.plugin_url = None

    def alter_selection(self, ctx):
        ctx.panel.state.selection = ctx.params["value"]

    def install_plugin(self, ctx):
        plugins = ctx.panel.get_state("table")
        plugin_names = []
        for p in plugins:
            if p["url"] == ctx.panel.state.plugin_url:
                plugin_names = [p.get("name")]
                break

        fop.download_plugin(
            ctx.panel.state.plugin_url,
            plugin_names=plugin_names,
            overwrite=True,
        )
        ctx.ops.notify(f"{plugin_names[0]} installed!", variant="success")

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

        table = types.TableView()
        table.add_column("name", label="Plugin")
        table.add_column("description", label="Description")
        table.add_column("url", label="URL")
        table.add_column("category", label="Category")
        panel.list("table", types.Object(), view=table)

        plugins = ctx.panel.get_state("table")
        menu = panel.menu("menu", variant="square", color="secondary")

        # Define a dropdown menu and add choices
        dropdown = types.DropdownView()

        for p in plugins:
            dropdown.add_choice(
                p["name"],
                label=f"{p['name']}",
                description=p["description"],
            )

        menu.str(
            "dropdown",
            view=dropdown,
            label="Labs Menu",
            on_change=self.alter_selection,
        )

        for p in plugins:
            if ctx.panel.state.selection == p["name"]:
                ctx.panel.state.plugin_url = p["url"]
                menu.btn(
                    p["name"],
                    label="Install",
                    on_click=self.install_plugin,
                    color="51",
                )

        return types.Property(
            panel,
            view=types.ObjectView(
                align_x="center", align_y="center", orientation="vertical"
            ),
        )


def register(p):
    p.register(LabsPanel)
