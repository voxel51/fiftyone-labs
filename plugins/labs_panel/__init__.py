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
        # TODO: Change to assets path when repo is public
        ctx.panel.state.logo = "https://github.com/manushreegangwar/assets/blob/fee702d587ca56de89da55511199a9ce776cd4f2/labs_logo_full.png"

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

    def show_url(self, ctx):
        ctx.ops.notify(
            f"README available at {ctx.panel.state.plugin_url}",
            variant="success",
        )

    def render(self, ctx):
        panel = types.Object()

        # Panel header
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

        # Table of plugins
        table = types.TableView()
        table.add_column("name", label="Plugin")
        table.add_column("description", label="Description")
        table.add_column("url", label="URL")
        table.add_column("category", label="Category")
        panel.list("table", types.Object(), view=table)

        # Dropdown for installation
        menu = panel.menu("menu", variant="square", color="secondary")
        dropdown = types.DropdownView()
        plugins = ctx.panel.get_state("table")
        for p in plugins:
            dropdown.add_choice(
                p["name"],
                label=f"{p['name']}",
                description=p["description"],
            )

        menu.str(
            "dropdown",
            view=dropdown,
            label="Try Labs",
            on_change=self.alter_selection,
        )

        for p in plugins:
            if ctx.panel.state.selection == p["name"]:
                ctx.panel.state.plugin_url = p["url"]
                menu.btn(
                    f"{p['name']}_install",
                    label="Install",
                    on_click=self.install_plugin,
                    color="51",
                )
                menu.btn(
                    f"{p['name']}_info",
                    label="Learn More",
                    on_click=self.show_url,
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
