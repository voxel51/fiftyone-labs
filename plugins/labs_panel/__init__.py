import os

import fiftyone.operators as foo
import fiftyone.operators.types as types

from .utils import list_labs_plugins


class LabsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="labs_panel",
            label="FiftyOne Labs",
        )

    def on_load(self, ctx):
        plugins = list_labs_plugins()
        ctx.panel.state.table = plugins

    def on_plugin_select(self, ctx):
        pass

    def render(self, ctx):
        panel = types.Object()

        # TODO: use relative path?
        current_dir = os.path.dirname(os.path.abspath(__file__))
        markdown_path = os.path.join(current_dir, "assets/labs_title.md")

        with open(markdown_path, "r") as markdown_file:
            panel.md(markdown_file.read(), name="markdown_title")

        # List of all the Labs plugins
        table = types.TableView()
        table.add_column("name", label="Plugin")
        table.add_column("description", label="Description")
        table.add_column("url", label="URL")
        table.add_column("image")
        panel.list("table", types.Object(), view=table)

        return types.Property(panel, view=types.ObjectView())


def register(p):
    p.register(LabsPanel)
