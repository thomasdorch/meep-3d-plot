import meep as mp
from meep import Vector3, Volume
from meep.visualization import box_vertices, filter_dict
from typing import Optional, Union
from collections import namedtuple
from functools import partial
import pyvista

import numpy as np


def get_boundary_volumes(sim: mp.Simulation, boundary_thickness: float, direction: float, side) -> Volume:
    thickness = boundary_thickness

    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(
        sim.geometry_center, sim.cell_size, sim.is_cylindrical
    )

    if direction == mp.X and side == mp.Low:
        return Volume(
            center=Vector3(
                xmin + 0.5 * thickness, sim.geometry_center.y, sim.geometry_center.z
            ),
            size=Vector3(thickness, sim.cell_size.y, sim.cell_size.z),
        )
    elif (direction == mp.X and side == mp.High) or direction == mp.R:
        return Volume(
            center=Vector3(
                xmax - 0.5 * thickness, sim.geometry_center.y, sim.geometry_center.z
            ),
            size=Vector3(thickness, sim.cell_size.y, sim.cell_size.z),
        )
    elif direction == mp.Y and side == mp.Low:
        return Volume(
            center=Vector3(
                sim.geometry_center.x, ymin + 0.5 * thickness, sim.geometry_center.z
            ),
            size=Vector3(sim.cell_size.x, thickness, sim.cell_size.z),
        )
    elif direction == mp.Y and side == mp.High:
        return Volume(
            center=Vector3(
                sim.geometry_center.x, ymax - 0.5 * thickness, sim.geometry_center.z
            ),
            size=Vector3(sim.cell_size.x, thickness, sim.cell_size.z),
        )
    elif direction == mp.Z and side == mp.Low:
        xcen = sim.geometry_center.x
        if sim.is_cylindrical:
            xcen += 0.5 * sim.cell_size.x
        return Volume(
            center=Vector3(xcen, sim.geometry_center.y, zmin + 0.5 * thickness),
            size=Vector3(sim.cell_size.x, sim.cell_size.y, thickness),
        )
    elif direction == mp.Z and side == mp.High:
        xcen = sim.geometry_center.x
        if sim.is_cylindrical:
            xcen += 0.5 * sim.cell_size.x
        return Volume(
            center=Vector3(xcen, sim.geometry_center.y, zmax - 0.5 * thickness),
            size=Vector3(sim.cell_size.x, sim.cell_size.y, thickness),
        )
    else:
        raise ValueError("Invalid boundary type")

def plot_volume3d(
        plotter: pyvista.Plotter,
        volume: Volume,
        name: str = None,
        **plot_parameters
):
    actor = None
    if mp.am_master():
        vol = pyvista.Cube(
            center=[volume.center.x, volume.center.y, volume.center.z],
            x_length=volume.size.x, y_length=volume.size.y, z_length=volume.size.z
        )
        params = filter_dict(plot_parameters, pyvista.Plotter.add_mesh)
        if name is not None:
            params["name"] = name
        actor = plotter.add_mesh(vol, **params)
    return plotter, actor


default_eps_parameters_3d = {
    "cmap": "binary",
    "opacity": 0.85,
    "contour": False,
    "line_width": 1,
    "show_edges": False,
    "frequency": None,
    "resolution": None,
    "show_scalar_bar": False,
    "label": "epsilon"
}

default_field_parameters_3d = {
    "cmap": "jet",
    "opacity": 0.85,
    "line_width": 1,
    "show_edges": False,
    "frequency": None,
    "post_process": np.real,
    "show_scalar_bar": True,
    "scalar_bar_args": {"title": "field-values"},
    "label": "field"
}

default_source_parameters_3d = {
    "color": "r",
    "opacity": 0.7,
    "show_edges": True,
    "backface_culling": False,
    "label": "source"
}

default_monitor_parameters_3d = {
    "color": "b",
    "opacity": 0.7,
    "show_edges": True,
    "backface_culling": False,
    "label": "monitor"
}
default_boundary_parameters_3d = {
    "color": 'g',
    "opacity": 0.15,
    "edge_color": 'k',
    "line_width": 5,
    "show_edges": True,
    "render_lines_as_tubes": True,
    "label": "boundary"
}

translated_keys = {
    "alpha": "opacity",
    "contour_linewidth": "line_width",
    "colorbar": "show_scalar_bar"
}


def get_3d_ticks(vol: mp.Volume, resolution: int):
    res = resolution
    sim_size = vol.size
    sim_center = vol.center

    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(box_size=sim_size, box_center=sim_center)

    Nx = int(sim_size.x * res)
    Ny = int(sim_size.y * res)
    Nz = int(sim_size.z * res)

    xtics = np.linspace(xmin, xmax, Nx)
    ytics = np.linspace(ymin, ymax, Ny)
    ztics = np.linspace(zmin, zmax, Nz)

    return xtics, ytics, ztics

def plot_eps_3d(
        sim: mp.Simulation,
        eps_parameters: dict = None
):
    if eps_parameters is None:
        eps_parameters = default_eps_parameters_3d
    else:
        eps_parameters = dict(default_eps_parameters_3d, **eps_parameters)

    for key, trans in translated_keys.items():
        if key in eps_parameters:
            eps_parameters[trans] = eps_parameters.pop(key)

    res = eps_parameters["resolution"] or sim.resolution
    freq = eps_parameters["frequency"] or 0

    xtics, ytics, ztics = get_3d_ticks(mp.Volume(center=sim.geometry_center, size=sim.cell_size), res)

    return np.rot90(np.real(sim.get_epsilon_grid(xtics, ytics, ztics, frequency=freq)))


def get_field_data(
        sim: mp.Simulation,
        component,
        field_parameters: dict = None
):
    if not sim._is_initialized:
        sim.init_sim()

    if field_parameters is None:
        field_parameters = default_field_parameters_3d
    else:
        field_parameters = dict(default_field_parameters_3d, **field_parameters)

    for key, trans in translated_keys.items():
        if key in field_parameters:
            field_parameters[trans] = field_parameters.pop(key)

    vol = mp.Volume(center=sim.geometry_center, size=sim.cell_size)

    field_data = sim.get_array(vol=vol, component=component)

    if field_parameters:
        field_data = field_parameters["post_process"](field_data)

    if (sim.dimensions == mp.CYLINDRICAL) or sim.is_cylindrical:
        field_data = np.flipud(field_data)
    else:
        field_data = np.rot90(field_data)

    return field_data, field_parameters


def plot_sources3d(sim: mp.Simulation, pl: pyvista.Plotter, source_parameters: dict = None):
    # consolidate plotting parameters
    if source_parameters is None:
        source_parameters = default_source_parameters_3d
    else:
        source_parameters = dict(default_source_parameters_3d, **source_parameters)
    actors = []
    for i, src in enumerate(sim.sources):
        vol = Volume(center=src.center, size=src.size)
        pl, actor = plot_volume3d(plotter=pl, volume=vol, **source_parameters)
        actors.append(actor)
    return pl, actors


def plot_monitors3d(sim: mp.Simulation, pl: pyvista.Plotter, monitor_parameters: dict = None):
    # consolidate plotting parameters
    if monitor_parameters is None:
        monitor_parameters = default_monitor_parameters_3d
    else:
        monitor_parameters = dict(default_monitor_parameters_3d, **monitor_parameters)
    actors = []
    for i, mon in enumerate(sim.dft_objects):
        for j, reg in enumerate(mon.regions):
            vol = Volume(center=reg.center, size=reg.size)
            pl, actor = plot_volume3d(plotter=pl, volume=vol, **monitor_parameters)
            actors.append(actor)
    return pl, actors

def plot_boundaries3d(sim: mp.Simulation, pl: pyvista.Plotter, boundary_parameters: dict = None):
    import itertools

    # consolidate plotting parameters
    if boundary_parameters is None:
        boundary_parameters = default_boundary_parameters_3d
    else:
        boundary_parameters = dict(default_boundary_parameters_3d, **boundary_parameters)
    actors = []
    for boundary in sim.boundary_layers:
        # boundary on all four sides
        if boundary.direction == mp.ALL and boundary.side == mp.ALL:
            if sim.dimensions == 1:
                dims = [mp.X]
            elif sim.dimensions == mp.CYLINDRICAL or sim.is_cylindrical:
                dims = [mp.X, mp.Z]
            elif sim.dimensions == 2:
                dims = [mp.X, mp.Y]
            elif sim.dimensions == 3:
                dims = [mp.X, mp.Y, mp.Z]
            else:
                raise ValueError("Invalid simulation dimensions")
            for permutation in itertools.product(dims, [mp.Low, mp.High]):
                if ((permutation[0] == mp.X) and (permutation[1] == mp.Low)) and (
                        sim.dimensions == mp.CYLINDRICAL or sim.is_cylindrical
                ):
                    continue
                vol = get_boundary_volumes(sim, boundary.thickness, *permutation)
                pl, actor = plot_volume3d(pl, vol, **boundary_parameters)
                actors.append(actor)
        # boundary on only two of four sides
        elif boundary.side == mp.ALL:
            for side in [mp.Low, mp.High]:
                if ((boundary.direction == mp.X) and (side == mp.Low)) and (
                        sim.dimensions == mp.CYLINDRICAL or sim.is_cylindrical
                ):
                    continue
                vol = get_boundary_volumes(sim, boundary.thickness, boundary.direction, side)
                pl, actor = plot_volume3d(pl, vol, **boundary_parameters)
                actors.append(actor)
        # boundary on just one side
        else:
            if ((boundary.direction == mp.X) and (boundary.side == mp.Low)) and (
                    sim.dimensions == mp.CYLINDRICAL or sim.is_cylindrical
            ):
                continue
            vol = get_boundary_volumes(sim, boundary.thickness, boundary.direction, boundary.side)
            pl, actor = plot_volume3d(pl, vol, **boundary_parameters)
            actors.append(actor)
    return pl, actors

def setup_3d_view(plotter: pyvista.Plotter, camera_position: Union[str, tuple]) -> pyvista.Plotter:
    plotter.add_axes()
    plotter.show_bounds(
        grid='front',
        location='outer',
        all_edges=True,
    )
    plotter.camera_position = camera_position
    return plotter


class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actors):
        self.actors = actors

    def __call__(self, state):
        for actor in self.actors:
            actor.SetVisibility(state)


def plot3d(
        sim: mp.Simulation,
        field_component=None,
        pl: pyvista.Plotter = None,
        eps_parameters: Optional[dict] = None,
        boundary_parameters: Optional[dict] = None,
        source_parameters: Optional[dict] = None,
        monitor_parameters: Optional[dict] = None,
        field_parameters: Optional[dict] = None,
        plot_eps_flag: bool = True,
        plot_sources_flag: bool = True,
        plot_monitors_flag: bool = True,
        plot_boundaries_flag: bool = True,
        camera_position: Union[str, tuple] = "xz",
        shared_grid: Optional[pyvista.StructuredGrid] = None,
        notebook: bool = False
):
    if pl is None and mp.am_master():
        pl = pyvista.Plotter(notebook=notebook)

    if shared_grid is None:
        res = sim.resolution
        vol = mp.Volume(center=sim.geometry_center, size=sim.cell_size)

        shared_grid = pyvista.StructuredGrid(*np.meshgrid(*get_3d_ticks(vol=vol, resolution=sim.resolution)))

    xtics, ytics, ztics = get_3d_ticks(mp.Volume(center=sim.geometry_center, size=sim.cell_size), res)

    widgets = {}
    toggle_box_size = 50
    toggle_x, toggle_y_start = 5.0, 12
    actors = {}
    meshes = {}
    # Plot geometry
    if plot_eps_flag:
        if eps_parameters is None:
            eps_parameters = default_eps_parameters_3d
        else:
            eps_parameters = dict(default_eps_parameters_3d, **eps_parameters)

        for key, trans in translated_keys.items():
            if key in eps_parameters:
                eps_parameters[trans] = eps_parameters.pop(key)

        eps_data = np.rot90(np.real(sim.get_epsilon_grid(xtics, ytics, ztics)))

        if mp.am_master():
            eps_wrap = pyvista.wrap(eps_data)['values']
            shared_grid['epsilon'] = eps_wrap

            eps_parameters_all = filter_dict(eps_parameters, pyvista.Plotter.add_mesh)
            eps_parameters_all.update(filter_dict(eps_parameters, pyvista.Plotter.add_mesh_threshold))
            # shared_grid.contour(scalars='epsilon') if eps_parameters["contour"] else
            eps_mesh = shared_grid
            actors["eps"] = pl.add_mesh_threshold(
                eps_mesh,
                name="epsilon",
                scalars="epsilon",
                **eps_parameters_all
            )

            widgets["eps_toggle"] = pl.add_checkbox_button_widget(
                SetVisibilityCallback([actors["eps"]]),
                value=True,
                position=(toggle_x, toggle_y_start),
                border_size=1,
                color_on="black",
                color_off="grey",
                background_color="grey"
            )

            toggle_y_start = toggle_y_start + toggle_box_size + (toggle_box_size // 10)

    # Plot boundaries
    if plot_boundaries_flag:
        pl, bnd_actors = plot_boundaries3d(
            sim,
            pl=pl,
            boundary_parameters=boundary_parameters,
        )
        if mp.am_master():
            widgets["boundaries"] = bnd_actors
            widgets["boundary_toggle"] = pl.add_checkbox_button_widget(
                SetVisibilityCallback(bnd_actors),
                value=True,
                position=(toggle_x, toggle_y_start),
                border_size=1,
                color_on="g",
                color_off="grey",
                background_color="grey"
            )
        toggle_y_start = toggle_y_start + toggle_box_size + (toggle_box_size // 10)
    # Plot sources
    if plot_sources_flag:
        pl, src_actors = plot_sources3d(
            sim,
            pl=pl,
            source_parameters=source_parameters,
        )
        if mp.am_master():
            widgets["sources"] = src_actors
            widgets["source_toggle"] = pl.add_checkbox_button_widget(
                SetVisibilityCallback(src_actors),
                value=True,
                position=(toggle_x, toggle_y_start),
                border_size=1,
                color_on="r",
                color_off="grey",
                background_color="grey"
            )
        toggle_y_start = toggle_y_start + toggle_box_size + (toggle_box_size // 10)

    # Plot monitors
    if plot_monitors_flag:
        pl, mon_actors = plot_monitors3d(
            sim,
            pl=pl,
            monitor_parameters=monitor_parameters,
        )
        if mp.am_master():
            widgets["monitors"] = mon_actors
            widgets["monitor_toggle"] = pl.add_checkbox_button_widget(
                SetVisibilityCallback(mon_actors),
                value=True,
                position=(toggle_x, toggle_y_start),
                border_size=1,
                color_on="b",
                color_off="grey",
                background_color="grey"
            )
        toggle_y_start = toggle_y_start + toggle_box_size + (toggle_box_size // 10)

    # Plot fields
    if field_component is not None:
        # pl, field_actor = plot_field3d(
        #     sim,
        #     field_component,
        #     grid=shared_grid,
        #     pl=pl,
        #     field_parameters=field_parameters,
        # )
        if not sim._is_initialized:
            sim.init_sim()

        if field_parameters is None:
            field_parameters = default_field_parameters_3d
        else:
            field_parameters = dict(default_field_parameters_3d, **field_parameters)

        for key, trans in translated_keys.items():
            if key in field_parameters:
                field_parameters[trans] = field_parameters.pop(key)

        vol = mp.Volume(center=sim.geometry_center, size=sim.cell_size)

        field_data = sim.get_array(vol=vol, component=field_component)

        field_data = field_parameters["post_process"](field_data)

        if (sim.dimensions == mp.CYLINDRICAL) or sim.is_cylindrical:
            field_data = np.flipud(field_data)
        else:
            field_data = np.rot90(field_data)

        if mp.am_master():
            wrap = pyvista.wrap(field_data)["values"]
            shared_grid["field"] = wrap

            field_actor = pl.add_mesh(
                shared_grid.contour(scalars="field"),
                name="field",
                scalars="field",
                **filter_dict(field_parameters, pyvista.Plotter.add_mesh)
            )

            widgets["field_plot"] = field_actor
            widgets["field_toggle"] = pl.add_checkbox_button_widget(
                SetVisibilityCallback([field_actor]),
                value=True,
                position=(toggle_x, toggle_y_start),
                border_size=1,
                color_on="y",
                color_off="grey",
                background_color="grey"
            )

    if mp.am_master():
        pl = setup_3d_view(plotter=pl, camera_position=camera_position)
    return pl, actors, widgets

# import mpi4py
class Animate3D:
    def __init__(
            self,
            field_component,
            camera_position: Union[str, tuple] = "xz",
            aspect: float = None,
            update_eps: bool = False,
            save_file: str = None,
            notebook: bool = False,
            **customization_args
    ):

        self.field_component = field_component
        self.update_eps = update_eps
        self.customization_args = customization_args
        self.filtered_args = filter_dict(self.customization_args, plot3d)

        self.save_file = save_file
        self.camera_position = camera_position

        self.init = False

        vol = mp.Volume(center=sim.geometry_center, size=sim.cell_size)
        xtics, ytics, ztics, = get_3d_ticks(vol=vol, resolution=sim.resolution)

        self.shared_grid = pyvista.StructuredGrid(*np.meshgrid(xtics, ytics, ztics))

        _movie_plotter = None
        plotter = None

        if mp.am_master():
            plotter = pyvista.Plotter(off_screen=False, line_smoothing=True, notebook=notebook)

            if aspect:
                plotter.window_size = [int(1080 * aspect), int(1080 * (1 / aspect))]

            if save_file is not None:
                _movie_plotter = pyvista.Plotter(off_screen=True, notebook=notebook, line_smoothing=True)
                if aspect:
                    _movie_plotter.window_size = [int(2160 * aspect), int(2160 * (1 / aspect))]

        self.plotter = plotter
        self._movie_plotter = _movie_plotter

        # Needed for step functions
        self.__code__ = namedtuple("gna_hack", ["co_argcount"])
        self.__code__.co_argcount = 2

    def __call__(self, sim: mp.Simulation, todo: str) -> None:
        if todo == "step":
            extra_args = self.filtered_args.copy()
            if self.init:
                extra_args.update(
                    {
                        "plot_eps_flag": self.update_eps,
                        "plot_sources_flag": False,
                        "plot_monitors_flag": False,
                        "plot_boundaries_flag": False,
                    }
                )
            plotter, actors, widgets = plot3d(
                sim=sim,
                field_component=self.field_component,
                shared_grid=self.shared_grid,
                pl=self.plotter,
                **extra_args
            )
            if self._movie_plotter is not None:
                _movie_plotter, _movie_actors, _movie_widgets = plot3d(
                    sim=sim,
                    field_component=self.field_component,
                    shared_grid=self.shared_grid,
                    pl=self._movie_plotter,
                    **extra_args
                )
            else:
                _movie_plotter = None

            if not self.init:
                if mp.am_master():
                    plotter.show(interactive_update=True)

                    if _movie_plotter is not None:
                        _movie_plotter.open_movie(self.save_file, framerate=2, quality=10)
                self.init = True

            if _movie_plotter is not None and mp.am_master():
                _movie_plotter.write_frame()

            self.plotter = plotter
            self._movie_plotter = _movie_plotter

        elif todo == "finish":
            _movie_plotter = self._movie_plotter
            plotter = self.plotter

            if mp.am_master():
                if _movie_plotter is not None:
                    _movie_plotter.close()

            self.plotter = plotter
            self._movie_plotter = _movie_plotter


if __name__ == "__main__":

    f = 1 / 1.55

    src = mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=f, fwidth=0.1),
        center=mp.Vector3(-1.4),
        size=mp.Vector3(y=2, z=2),
        eig_match_freq=True
    )
    mon = mp.FluxRegion(
        center=mp.Vector3(1.4),
        size=mp.Vector3(y=2, z=2),
    )
    sim = mp.Simulation(
        geometry=[
            mp.Block(size=mp.Vector3(10, 0.5, 0.22), center=mp.Vector3(), material=mp.Medium(index=3.47)),
        ],
        cell_size=mp.Vector3(5, 5, 5),
        resolution=30,
        sources=[src],
        default_material=mp.Medium(index=1.44),
        boundary_layers=[mp.PML(thickness=1.0, direction=mp.ALL)]
    )
    sim.add_flux(f, 0.1, 10, mon)

    pl, ac, ws = plot3d(
        sim,
        field_parameters={"post_process": lambda x: np.abs(x)**2},
        plot_boundaries_flag=False
    )

    if mp.am_master():
        pl.show()

    # anim = Animate3D(
    #     field_component=mp.Ey,
    #     field_parameters={"post_process": lambda x: np.abs(x)**2},
    #     plot_boundaries_flag=False
    # )
    #
    # sim.run(mp.at_every(0.15, anim), until=1)
    #
    # if mp.am_master():
    #     anim.plotter.show()
