import logging
import requests
import zipfile
import io
import os
import shutil

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.plugins as fop
import fiftyone.operators.types as types
import fiftyone.core.utils as fou

from .utils import (
    list_labs_features,
    add_version_info_to_features,
)

fom = fou.lazy_import("fiftyone.management")
logger = logging.getLogger(__name__)


def is_enterprise():
    return hasattr(fo.constants, "TEAMS_VERSION")


class LabsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="labs_panel",
            label="FiftyOne Labs",
        )

    def on_load(self, ctx):
        ctx.panel.state.logo = "https://raw.githubusercontent.com/voxel51/labs/refs/heads/main/assets/labs_logo_transparent_dark.svg"

        plugins = add_version_info_to_features(list_labs_features())
        ctx.panel.state.table = plugins
        ctx.panel.state.plugin_url = None

    def alter_selection(self, ctx):
        ctx.panel.state.selection = ctx.params["value"]

    def install_plugin(self, ctx):
        plugins = ctx.panel.get_state("table")
        for p in plugins:
            if p["url"] == ctx.panel.state.plugin_url:
                if is_enterprise():
                    zip_path = _download_plugin_dir(
                        p["url"], extract_to="/tmp/plugins"
                    )
                    fom.upload_plugin(
                        zip_path, overwrite=p.get("curr_version") is not None
                    )
                    pdef = fom.get_plugin_info(p["name"])

                else:
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

    def uninstall_plugin(self, ctx):
        plugins = ctx.panel.get_state("table")
        for p in plugins:
            if p["name"] == ctx.panel.state.selection:
                curr_version = p.get("curr_version")
                if curr_version:
                    if is_enterprise():
                        fom.delete_plugin(ctx.panel.state.selection)
                    else:
                        fop.delete_plugin(ctx.panel.state.selection)
                    ctx.ops.notify(
                        f"{p['name']} uninstalled",
                        variant="success",
                    )
                    p["status"] = "Not installed"
                    p["curr_version"] = None
                else:
                    ctx.ops.notify(
                        f"{p['name']} is not currently installed",
                        variant="success",
                    )
                break
        ctx.panel.state.table = plugins

    def show_url(self, ctx):
        ctx.ops.notify(
            f"README available at {ctx.panel.state.plugin_url}",
            variant="success",
        )

    def render(self, ctx):
        panel = types.Object()

        panel.message("beta_tag", "BETA VERSION")

        # Panel header
        panel.img("logo", height="120px")
        panel.md(
            "##### FiftyOne Labs brings research solutions and experimental features for machine learning",
            name="labs_subtitle",
        )
        panel.md(
            (
                "Please note that these features are experimental. They may not be production-ready.\n\n"
                "We encourage you to try them out and share your feedback on the [FiftyOne Labs Discord Channel](https://discord.com/channels/1266527359511564372/1466492755214733625)"
            ),
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
                if p.get("curr_version") and p["name"] != "@51labs/labs_panel":
                    menu.btn(
                        f"{p['name']}_uninstall",
                        label="Uninstall",
                        on_click=self.uninstall_plugin,
                        color="51",
                    )

        return types.Property(
            panel,
            view=types.GridView(align_x="center", align_y="center", gap=2),
        )


def register(p):
    p.register(LabsPanel)


def _download_plugin_dir(
    plugin_url, plugin_branch="main", extract_to="/tmp", zip_name=None
):
    """Download a specific directory from GitHub URL

    Args:
        plugin_url: GitHub URL to plugin directory
                   "https://github.com/<owner>/<repo>/tree/<branch>/path/to/dir"
        plugin_branch: Branch of the plugin in the GitHub URL
        extract_to: local directory to extract contents to
        zip_name: name of the zip file for the directory
    """
    url_parts = plugin_url.rstrip("/").split("/tree/main/")
    owner_repo = url_parts[0].split("github.com/")[-1]
    dir_path = url_parts[1] if len(url_parts) > 1 else None
    zip_url = (
        f"https://api.github.com/repos/{owner_repo}/zipball/{plugin_branch}"
    )
    response = requests.get(zip_url)

    if response.status_code != 200:
        logger.info(f"Failed to download {zip_url}: {response.status_code}")
        return None

    # Create temporary extraction directory
    temp_dir = os.path.join(extract_to, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    extracted_files = []

    if zip_name is None:
        if dir_path:
            zip_name = dir_path.split("/")[-1]
        else:
            zip_name = owner_repo.replace("/", "_")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        for file_info in zip_ref.filelist:
            print(file_info)
            file_parts = file_info.filename.split("/", 1)

            if len(file_parts) > 1:
                relative_path = file_parts[1]

                if dir_path:
                    # Extract files in the dir path
                    if relative_path.startswith(dir_path + "/"):
                        content_path = relative_path[len(dir_path) + 1 :]

                        if content_path and not file_info.is_dir():
                            local_path = os.path.join(temp_dir, content_path)
                            os.makedirs(
                                os.path.dirname(local_path), exist_ok=True
                            )

                            with zip_ref.open(file_info) as source:
                                with open(local_path, "wb") as target:
                                    target.write(source.read())

                            extracted_files.append(local_path)
                else:
                    # Extract all files in the repo
                    if not file_info.is_dir():
                        local_path = os.path.join(temp_dir, relative_path)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)

                        with zip_ref.open(file_info) as source:
                            with open(local_path, "wb") as target:
                                target.write(source.read())

                        extracted_files.append(local_path)

    if not extracted_files:
        shutil.rmtree(temp_dir)
        return None

    zip_path = os.path.join(extract_to, f"{zip_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in extracted_files:
            zipf.write(file_path, os.path.relpath(file_path, temp_dir))

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    return zip_path
