import fiftyone.operators as foo
import fiftyone.plugins as fop
import fiftyone.operators.types as types

from .utils import list_labs_features, add_version_info_to_features


class LabsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="labs_panel",
            label="FiftyOne Labs",
        )

    def on_load(self, ctx):
        # TODO: Change to assets path when repo is public
        ctx.panel.state.logo = "https://raw.githubusercontent.com/manushreegangwar/assets/refs/heads/main/labs_logo_full_dark.png"

        plugins = add_version_info_to_features(list_labs_features())
        ctx.panel.state.table = plugins
        ctx.panel.state.plugin_url = None

    def alter_selection(self, ctx):
        ctx.panel.state.selection = ctx.params["value"]

    def install_plugin(self, ctx):
        plugins = ctx.panel.get_state("table")
        for p in plugins:
            if p["url"] == ctx.panel.state.plugin_url:
                fop.download_plugin(
                    ctx.panel.state.plugin_url,
                    plugin_names=[p.get("name")],
                    overwrite=True,
                )
                pdef = fop.core.get_plugin(p["name"])
                stale_version = p.get("curr_version")
                curr_version = pdef.version
                if stale_version:
                    ctx.ops.notify(
                        f"{p['name']} updated from {stale_version} to {curr_version}",
                        variant="success",
                    )
                else:
                    ctx.ops.notify(
                        f"{p['name']} version {curr_version} installed",
                        variant="success",
                    )
                p["status"] = "Installed"
                p["curr_version"] = curr_version
                break

        ctx.panel.state.table = plugins

    def show_url(self, ctx):
        ctx.ops.notify(
            f"README available at {ctx.panel.state.plugin_url}",
            variant="success",
        )

    def render(self, ctx):
        panel = types.Object()

        # Panel header
        panel.img("logo", height="120px")
        panel.md(
            "_FiftyOne Labs brings research solutions and experimental features for machine learning_",
            name="labs_subtitle",
        )
        panel.md(
            "Please note that these features are experimental. They may not be production-ready. We encourage you to try them out and share your feedback.",
            name="labs_description",
        )

        # Table of plugins
        table = types.TableView()
        table.add_column("name", label="Labs Feature")
        table.add_column("description", label="Description")
        table.add_column("category", label="Category")
        table.add_column("status", label="Status")
        table.add_column("curr_version", label="Version")
        table.add_column("url", label="URL")
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
            view=types.GridView(align_x="center", align_y="center", gap=2),
        )


def register(p):
    p.register(LabsPanel)
