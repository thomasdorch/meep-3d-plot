from types import resolve_bases
import meep as mp
from meep import Vector3, Volume
from meep.simulation import re
from meep.visualization import box_vertices, filter_dict
from typing import Optional, Union
from collections import namedtuple
from functools import partial
import pyvista as pv

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

def clean_dict(input_dict: Optional[dict], defaults_dict: dict):
    # add default values if key not specified
    if input_dict is None:
        input_dict = defaults_dict 
    else:
        input_dict = dict(defaults_dict, **input_dict)

    # replace keys if using plot2d key names
    for key, trans in translated_keys.items():
        if key in input_dict:
           input_dict[trans] = input_dict.pop(key)
    return input_dict


def get_3d_ticks(size: mp.Vector3, center: mp.Vector3, resolution: int):
    xmin, xmax, ymin, ymax, zmin, zmax = box_vertices(box_size=size, box_center=center)

    Nx = int(size.x * resolution)
    Ny = int(size.y * resolution)
    Nz = int(size.z * resolution)

    xtics = np.linspace(xmin, xmax, Nx)
    ytics = np.linspace(ymin, ymax, Ny)
    ztics = np.linspace(zmin, zmax, Nz)

    return xtics, ytics, ztics

# class SetVisibilityCallback:
#     """Helper callback to keep a reference to the actor being modified."""
#
#     def __init__(self, actors):
#         self.actors = actors
#
#     def __call__(self, state):
#         for actor in self.actors:
#             actor.SetVisibility(state)

class Plot3D:
    def __init__(
            self,
            sim: mp.Simulation,
            field_component=None,
            plotter: Optional[pv.Plotter] = None,
            close_when_finished: bool = False,
            update_epsilon: bool = False,
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
            notebook: bool = False,
            **plotter_kwargs
    ):
        self.close_when_finished = close_when_finished
        self.pl = plotter

        if self.pl is None and mp.am_master():
            self.pl = pv.Plotter(
                off_screen=False,
                line_smoothing=True,
                notebook=notebook,
                **filter_dict(plotter_kwargs, pv.Plotter.__init__)
            )

            self.pl.add_axes()
            self.pl.show_bounds(
                grid='front',
                location='outer',
                all_edges=True,
            )
            self.pl.camera_position = camera_position

        self.update_epsilon = update_epsilon
        self.field_component = field_component

        self.eps_parameters = clean_dict(eps_parameters, default_eps_parameters_3d)
        self.source_parameters = clean_dict(source_parameters, default_source_parameters_3d)
        self.monitor_parameters = clean_dict(monitor_parameters, default_monitor_parameters_3d)
        self.boundary_parameters = clean_dict(boundary_parameters, default_boundary_parameters_3d)
        self.field_parameters = clean_dict(field_parameters, default_field_parameters_3d)

        self.sim = sim
        self.frequency = self.eps_parameters["frequency"] or 0

        self.resolution = sim.resolution
        self.volume = mp.Volume(center=sim.geometry_center, size=sim.cell_size)

        self.xtics, self.ytics, self.ztics = get_3d_ticks(
            size=self.volume.size, center=self.volume.center, resolution=self.resolution
        )

        xv, yv, zv = np.meshgrid(self.xtics, self.ytics, self.ztics)
        
        self.grid = pv.StructuredGrid(xv, yv, zv) if mp.am_master() else None

        self.field_actor = None
        self.eps_actor = None
        self.boundary_actors = []
        self.source_actors = []
        self.monitor_actors = []

        self.font_size = 10

        self.toggle_box_size = 30
        self.toggle_x, self.toggle_y_start = 5.0, 12
        self.toggle_text_offset_x = self.toggle_x + self.toggle_box_size + 5
        self.toggle_text_offset_y = 5

        self.toggle_boxes = {}
        self.text_actors = {}
        self.mesh_actors = {}

        # Needed for step functions
        self.__code__ = namedtuple("gna_hack", ["co_argcount"])
        self.__code__.co_argcount = 2

    def _plot_all(self):
        if self.eps_actor is None or self.update_epsilon:
            eps_data = np.rot90(
                np.real(
                    self.sim.get_epsilon_grid(self.xtics, self.ytics, self.ztics, self.frequency)
                    )
                ).flatten(order='F')

            if mp.am_master():
                self.plot_epsilon(eps_data)

        if mp.am_master():
            self.plot_sources()
            self.plot_monitors()
            self.plot_boundaries()

        if self.field_component:
            if not self.sim._is_initialized:
                print("initializing sim...")
                self.sim.init_sim()

            field_data = self.sim.get_array(vol=self.volume, component=self.field_component)
            field_data = self.field_parameters["post_process"](field_data)

            if (self.sim.dimensions == mp.CYLINDRICAL) or self.sim.is_cylindrical:
                field_data = np.flipud(field_data)
            else:
                field_data = np.rot90(field_data)
            field_data = field_data.flatten(order='F')

            if mp.am_master():
                self.plot_field_component(field_data)

    def plot(self, show: bool = False):
        self._plot_all()

        if mp.am_master() and show:
            self.pl.show()

    def __call__(self, sim: mp.Simulation, todo: str) -> None:
        if todo == "step":
            self._plot_all()

            if mp.am_master():
                self.pl.show(interactive_update=True)

        elif todo == "finish":
            if mp.am_master():
                if self.close_when_finished:
                    self.pl.close()

    # These all need to get the actor dynamically
    def toggle_epsilon(self, state):
        self.eps_actor.SetVisibility(state)

    def toggle_fields(self, state):
        self.field_actor.SetVisibility(state)

    def toggle_sources(self, state):
        for actor in self.source_actors:
            actor.SetVisibility(state)

    def toggle_monitors(self, state):
        for actor in self.monitor_actors:
            actor.SetVisibility(state)

    def toggle_boundaries(self, state):
        for actor in self.boundary_actors:
            actor.SetVisibility(state)

    def increment_checkbox_pos(self):
        self.toggle_y_start = self.toggle_y_start + self.toggle_box_size + (self.toggle_box_size // 10)

    def add_toggle_box_text(self, text: str):
        self.text_actors[text] = self.pl.add_text(
            text=text,
            position=(self.toggle_text_offset_x, self.toggle_y_start + self.toggle_text_offset_y),
            font_size=self.font_size
        )

    def plot_volume3d(self, size: Union[Vector3, tuple], center: Union[Vector3, tuple] = (0, 0, 0), **plot_parameters):
        return self.pl.add_mesh(
                    pv.Cube(
                        center=(center.x, center.y, center.z) if isinstance(center, Vector3) else center,
                        x_length=size[0],
                        y_length=size[1],
                        z_length=size[2]                
                        ),
                    **filter_dict(plot_parameters, pv.Plotter.add_mesh)
                    )

    def plot_epsilon(self, eps_data: np.ndarray):
        # Plot geometry
        print("Updating epsilon data...")

        self.grid["epsilon"] = eps_data

        eps_parameters_all = filter_dict(self.eps_parameters, pv.Plotter.add_mesh)
        eps_parameters_all.update(filter_dict(self.eps_parameters, pv.Plotter.add_mesh_threshold))

        if self.eps_parameters["contour"]:
            self.eps_actor = self.pl.add_mesh(
                self.grid.contour(scalars="epsilon", compute_normals=True, progress_bar=True),
                name="epsilon",
                **eps_parameters_all
                )
        else:
            self.eps_actor = self.pl.add_mesh_threshold(
                self.grid,
                name="epsilon",
                scalars="epsilon",
                **eps_parameters_all
                )

        if "epsilon" not in self.toggle_boxes:
            self.toggle_boxes["epsilon"] = (
                self.pl.add_checkbox_button_widget(
                    self.toggle_epsilon,
                    size=self.toggle_box_size,
                    value=True,
                    position=(self.toggle_x, self.toggle_y_start),
                    border_size=1,
                    color_on="black",
                    color_off="grey",
                    background_color="grey"
                ))
            self.add_toggle_box_text(text="epsilon")
            self.increment_checkbox_pos()

    # Plot fields
    def plot_field_component(self, field_data: np.ndarray):
        print("Updating field data...")

        self.grid["field"] = field_data

        self.field_actor = self.pl.add_mesh(
            # Add option to draw glyphs
            self.grid.contour(scalars="field"),
            name="field",
            **filter_dict(self.field_parameters, pv.Plotter.add_mesh)
        )
        if "field" not in self.toggle_boxes:
            self.toggle_boxes["field"] = (
                self.pl.add_checkbox_button_widget(
                    self.toggle_fields,
                    size=self.toggle_box_size,
                    value=True,
                    position=(self.toggle_x, self.toggle_y_start),
                    border_size=1,
                    color_on="y",
                    color_off="grey",
                    background_color="grey"
                )
            )
            self.add_toggle_box_text(text="field")
            self.increment_checkbox_pos()

    # Plot sources
    def plot_sources(self):
        # Don't redraw
        if "sources" in self.toggle_boxes.keys():
            return

        for src in self.sim.sources:
            self.source_actors.append(
                    self.plot_volume3d(center=src.center, size=src.size, **self.source_parameters)
                    )

        self.toggle_boxes["sources"] = (
            self.pl.add_checkbox_button_widget(
                self.toggle_sources,
                size=self.toggle_box_size,
                value=True,
                position=(self.toggle_x, self.toggle_y_start),
                border_size=1,
                color_on="r",
                color_off="grey",
                background_color="grey"
            )
        )
        self.add_toggle_box_text(text="source")
        self.increment_checkbox_pos()

    # Plot monitors
    def plot_monitors(self):
        if "monitors" in self.toggle_boxes.keys():
            return

        for mon in self.sim.dft_objects:
            for reg in mon.regions:
                self.monitor_actors.append(
                        self.plot_volume3d(center=reg.center, size=reg.size, **self.monitor_parameters)
                        )

        self.toggle_boxes["monitors"] = (
            self.pl.add_checkbox_button_widget(
                self.toggle_monitors,
                size=self.toggle_box_size,
                value=True,
                position=(self.toggle_x, self.toggle_y_start),
                border_size=1,
                color_on="b",
                color_off="grey",
                background_color="grey"
            )
        )
        self.add_toggle_box_text(text="monitor")
        self.increment_checkbox_pos()

    def plot_boundaries(self):
        # Don't redraw boundaries
        if "boundaries" in self.toggle_boxes.keys():
            return

        import itertools
        for boundary in self.sim.boundary_layers:
            # boundary on all four sides
            if boundary.direction == mp.ALL and boundary.side == mp.ALL:
                if self.sim.dimensions == 1:
                    dims = [mp.X]
                elif self.sim.dimensions == mp.CYLINDRICAL or self.sim.is_cylindrical:
                    dims = [mp.X, mp.Z]
                elif self.sim.dimensions == 2:
                    dims = [mp.X, mp.Y]
                elif self.sim.dimensions == 3:
                    dims = [mp.X, mp.Y, mp.Z]
                else:
                    raise ValueError("Invalid simulation dimensions")
                for permutation in itertools.product(dims, [mp.Low, mp.High]):
                    if ((permutation[0] == mp.X) and (permutation[1] == mp.Low)) and (
                            self.sim.dimensions == mp.CYLINDRICAL or self.sim.is_cylindrical
                    ):
                        continue
                    vol = get_boundary_volumes(self.sim, boundary.thickness, *permutation)
                    self.boundary_actors.append(
                            self.plot_volume3d(
                                size=vol.size,
                                center=vol.center,
                                **self.boundary_parameters
                                )
                            )
            elif boundary.side == mp.ALL:
                # boundary on only two of four sides
                for side in [mp.Low, mp.High]:
                    if ((boundary.direction == mp.X) and (side == mp.Low)) and (
                        self.sim.dimensions == mp.CYLINDRICAL or self.sim.is_cylindrical
                    ):
                        continue
                    vol = get_boundary_volumes(self.sim, boundary.thickness, boundary.direction, side)
                    self.boundary_actors.append(
                            self.plot_volume3d(
                                size=vol.size,
                                center=vol.center,
                                **self.boundary_parameters
                                )
                        )
            # boundary on just one side
            else:
                if ((boundary.direction == mp.X) and (boundary.side == mp.Low)) and (
                        self.sim.dimensions == mp.CYLINDRICAL or self.sim.is_cylindrical
                ):
                    continue
                vol =  get_boundary_volumes(self.sim, boundary.thickness, boundary.direction, boundary.side)
                self.boundary_actors.append(
                        self.plot_volume3d(
                            size=vol.size,
                            center=vol.center,
                            **self.boundary_parameters
                            )
                        )

        self.toggle_boxes["boundaries"] = (
            self.pl.add_checkbox_button_widget(
                self.toggle_boundaries,
                size=self.toggle_box_size,
                value=True,
                position=(self.toggle_x, self.toggle_y_start),
                border_size=1,
                color_on="g",
                color_off="grey",
                background_color="grey"
            )
        )
        self.add_toggle_box_text(text="boundaries")
        self.increment_checkbox_pos()
       
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


        self.save_file = save_file
        self.camera_position = camera_position

        self.init = False

        _movie_plotter = None
        plotter = None

        self.actors = {}
        self._movie_actors = {}
        
        self.widgets = {}
        self._movie_widgets = {}


        if mp.am_master():
            plotter = pv.Plotter(off_screen=False, line_smoothing=True, notebook=notebook)

            if aspect:
                plotter.window_size = [int(1080 * aspect), int(1080 * (1 / aspect))]

            if save_file is not None:
                _movie_plotter = pv.Plotter(off_screen=True, notebook=notebook, line_smoothing=True)
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

            if mp.am_master():
                for actor in self.plotter.pickable_actors:
                   self.plotter.remove_actor(actor)

            plotter, actors, widgets = plot3d(
                sim=sim,
                field_component=self.field_component,
                pl=self.plotter,
                **extra_args
            )
            if self._movie_plotter is not None:
                _movie_plotter, _movie_actors, _movie_widgets = plot3d(
                    sim=sim,
                    field_component=self.field_component,
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

    plotter = Plot3D(sim)

    #
    # pl, ac, ws = plot3d(
    #     sim,
    #     field_parameters={"post_process": lambda x: np.abs(x)**2},
    #     plot_boundaries_flag=False
    # )
    #
    # if mp.am_master():
    #     pl.show()
    # anim = Animate3D(
    #     field_component=mp.Ey,
    #         #     # plot_boundaries_flag=False
    # )
   
    # sim.run(mp.at_every(0.15, plotter), until=1)

    plotter.plot(show=True)
