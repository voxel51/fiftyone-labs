import fiftyone.operators as foo
import fiftyone.operators.types as types

from .utils import list_labs_plugins


class LabsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="labs_panel",
            label="FiftyOne Labs Panel",
        )

    def on_load(self, ctx):
        plugins = list_labs_plugins()

        plugins_table = []
        for p in plugins:
            plugins_table.append(
                {
                    "Plugin": p.get("name", ""),
                    "Description": p.get("description", ""),
                    "URL": p.get("url", ""),
                }
            )

        ctx.panel.state.table = plugins_table

    def render(self, ctx):
        panel = types.Object()

        table = types.TableView()
        table.add_column("Plugin", label="Plugin")
        table.add_column("Description", label="Description")

        panel.list("table", types.Object(), view=table, label="Labs Plugins")

        return types.Property(panel, view=types.ObjectView())


def register(p):
    p.register(LabsPanel)
