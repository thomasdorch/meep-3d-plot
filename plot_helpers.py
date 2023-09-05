import math
from enum import Enum
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import meep as mp


def set_lims(ax, xlim, ylim):
    if mp.am_master():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


def plot_grad(sim: mp.Simulation, ax: plt.Axes, component, xlim=None, ylim=None, zlim=None):
    from meep.visualization import _add_colorbar

    slice_size = mp.Vector3(
        (xlim[1] - xlim[0]) if xlim is not None else sim.cell_size.x,
        (ylim[1] - ylim[0]) if ylim is not None else sim.cell_size.y,
        (zlim[1] - zlim[0]) if zlim is not None else sim.cell_size.z
    )
    slice_center = mp.Vector3(
        ((xlim[1] + xlim[0]) / 2) if xlim is not None else sim.geometry_center.x,
        ((ylim[1] + ylim[0]) / 2) if ylim is not None else sim.geometry_center.y,
        ((zlim[1] + zlim[0]) / 2) if zlim is not None else sim.geometry_center.z
    )

    data = sim.get_array(center=slice_center, size=slice_size, component=component)

    if mp.am_master():
        for cl in ax.collections:
            ax.collections.remove(cl)

        x = np.linspace(*(xlim or ax.get_xlim()), data.shape[0])
        y = np.linspace(*(ylim or ax.get_ylim()), data.shape[1])

        dx, dy = np.gradient(data)

        color = np.sqrt(dx**2 + dy**2)
        ax.quiver(
            x, y, dx.T, dy.T, color.T,
            angles='xy',
            scale_units='xy',
            scale=25,
            minlength=2,
            alpha=0.8
        )
        _add_colorbar(
            ax=ax,
            cmap='jet',
            vmin=np.amin(color),
            vmax=np.amax(color),
            default_label="field value"
        )
    return ax


def plot_contour(sim: mp.Simulation, ax: plt.Axes, component, xlim=None, ylim=None, zlim=None):
    slice_size = mp.Vector3(
        (xlim[1] - xlim[0]) if xlim is not None else sim.cell_size.x,
        (ylim[1] - ylim[0]) if ylim is not None else sim.cell_size.y,
        (zlim[1] - zlim[0]) if zlim is not None else sim.cell_size.z
    )
    slice_center = mp.Vector3(
        ((xlim[1] + xlim[0]) / 2) if xlim is not None else sim.geometry_center.x,
        ((ylim[1] + ylim[0]) / 2) if ylim is not None else sim.geometry_center.y,
        ((zlim[1] + zlim[0]) / 2) if zlim is not None else sim.geometry_center.z
    )

    data = sim.get_array(center=slice_center, size=slice_size, component=component)
    data = abs(data)**2

    if mp.am_master():
        x = np.linspace(*(xlim or ax.get_xlim()), data.shape[0])
        y = np.linspace(*(ylim or ax.get_ylim()), data.shape[1])

        ax.contour(x, y, data.T, cmap='jet', alpha=0.7, levels=11)
    return ax



def get_volume_info(
        volume: mp.Volume
) -> (tuple[str, str], str):
    if volume.size.x == 0:
        slice_ax = "x"
        nz_axes = ("y", "z")
    elif volume.size.y == 0:
        slice_ax = "y"
        nz_axes = ("z", "x")
    elif volume.size.z == 0:
        slice_ax = "z"
        nz_axes = ("x", "y")
    else:
        raise ValueError("Exactly 2 of the slice's axes must be non-zero.")
    return nz_axes, slice_ax


class FIELDS(Enum):
    E = "E"
    H = "H"
    S = "S"


def animate_slices(
        field_components: dict[FIELDS, tuple | list],
        slices: list[mp.Volume],
        field_params: dict = None,
        ax_label: str = "<field>, <slice-center>",
        figsize: tuple[int, int] = None
) -> list[mp.Animate2D]:
    """Animate each slice for each component."""
    from matplotlib.gridspec import GridSpec

    # Figure out how big the figure will need to be
    vol_widths = []
    vol_heights = []
    for vol in slices:
        axs, _ = get_volume_info(vol)
        vol_widths.append(math.ceil(getattr(vol.size, axs[0])))
        vol_heights.append(math.ceil(getattr(vol.size, axs[1])))

    max_height = max(vol_heights)

    tot_width = math.ceil(sum(vol_widths))  # max_width * len(slices)

    anim_list = []
    for field_idx, (field, components) in enumerate(field_components.items()):

        tot_height = max_height * len(components)

        if mp.am_master():
            import matplotlib.pyplot as plt

            # If a list of figsizes is supplied, use that
            fig = plt.figure(
                figsize=figsize[field_idx] if isinstance(figsize[0], Iterable) else figsize,
                constrained_layout=True
            )
            gs = GridSpec(nrows=tot_height, ncols=tot_width, figure=fig)

            fig.suptitle(f"{field.name}", weight="bold")

        for comp_idx, component in enumerate(components):

            for slice_idx, vol in enumerate(slices):
                nz_axs, slice_ax = get_volume_info(vol)

                ax_width_start = sum(vol_widths[:slice_idx])
                ax_width_stop = ax_width_start + vol_widths[slice_idx]

                ax_height_start = max_height * comp_idx
                ax_height_stop = ax_height_start + max_height

                if mp.am_master():
                    ax = fig.add_subplot(gs[ax_height_start:ax_height_stop, ax_width_start:ax_width_stop])
                    title = (
                        ax_label
                        .replace("<field>", f"{field.name}$_{component}$({nz_axs[0]}, {nz_axs[1]})")
                        .replace("<slice-center>", f"{slice_ax}={getattr(vol.center, slice_ax)}")
                    )
                    ax.set_title(title)
                    if comp_idx < len(components) - 1:
                        ax.tick_params(labelbottom=False)

                anim = mp.Animate2D(
                    f=fig if mp.am_master() else None,
                    realtime=True,
                    fields=getattr(mp, f"{field.name}{component}"),
                    field_parameters=field_params,
                    output_plane=vol
                )

                anim_list.append(anim)
    return anim_list
