from typing import List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.colors import ListedColormap, Colormap
import seaborn as sns

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

POSTER_BLUE = '#01589C'

from sciart.markers.flower import get_flower_markers



@dataclass
class DataEntry:
    class_names: List[str]
    count: int
    label: str = None
    annot_rpos: float = 0 # rho coordinate
    annot_tpos = [] # set of theta coordinates
    leaf_rpos: float = 1
    leaf_tpos: float = 0
    leaf_size: float = 0.1
    # has_one_class:bool = False
    bgc_class_index = None
    color = POSTER_BLUE
    marker = 'o'
    marker_size = 10

    @property
    def has_one_class(self):
        return len(self.class_names) == 1

    def pick_position(self, available_positions, bgc2position, num_entries):
        def get_distance(pos_list, new_pos):
            return np.min([min((p - new_pos)% num_entries, (new_pos - p)% num_entries) for p in pos_list])

        pos = [bgc2position[bgc_class] for bgc_class in self.class_names]
        if len(pos) == 1:
            # self.has_one_class = True
            return pos[0], pos
        min_pos = None
        min_dist = None
        for a in available_positions:
            a_dist = get_distance(pos, a)
            if min_dist is None or a_dist < min_dist:
                min_dist = a_dist
                min_pos = a
        available_positions.remove(min_pos)
        return min_pos, pos


def extract_data(
        df: pd.DataFrame, class_columns: List[str], 
        label_column: str = None, 
        debug: bool = False) -> tuple[list[DataEntry], float]:
 
    if label_column is not None and not label_column in df.columns:
        label_column = None
    num_entries = len(df)
    num_classes = len(class_columns)
    num_offset_entries = num_entries / num_classes
    theta_step_size = 2 * np.pi / (num_offset_entries*num_classes)

    bgc_class2position = {bgc_class: int(i*num_offset_entries) for i, bgc_class in enumerate(class_columns)}
    available_positions = set(range(num_entries)) - set(bgc_class2position.values())
    # print(available_positions)
    theta_positions = np.arange(num_entries)*theta_step_size
    # r_start = 0.3 # before adding mibig
    # r_step_size = 0.1
    # ##r_end = 2.8
    r_start = 0.7
    r_step_size = (2.8-r_start) / (num_entries - num_classes)
    r_leaf_offset = 0.2
    r_big_leaf_offset = 0.5
    max_r_level = 0
    r_end = r_start + (num_entries - num_classes)*r_step_size
    # print("r_end", r_end)
    all_entries = []
    for idx, row in df[df.num_classes == 1].iterrows():
        ids = row[class_columns].values.astype(bool)
        if debug:
            print(class_columns, ids, class_columns.dtype, ids.dtype)
        sample_classes = list(class_columns[ids])
        print(sample_classes)
        count = row['count']
        
        label = row[label_column] if label_column is not None and label_column in row.index else None
        entry = DataEntry(sample_classes, count, label=label)
        pos, annot_pos = entry.pick_position(available_positions, bgc_class2position, num_entries)
        # print(pos, annot_pos)
        # convert to coordinates and add to entry
        entry.leaf_tpos = theta_positions[pos]
        entry.annot_tpos = [theta_positions[p] for p in annot_pos]
        assert entry.has_one_class, str(row)
        entry.annot_rpos = r_start
        entry.leaf_size = 10
        entry.leaf_rpos = r_end + r_big_leaf_offset
        entry.bgc_class_index = list(class_columns).index(entry.class_names[0])
        all_entries.append(entry)

    counts_df_reordered = df[df.num_classes > 1].sort_values(
        by=["num_classes", "count"]+list(class_columns), 
        ascending=[False, False]+[False]*num_classes)

    for _, row in counts_df_reordered.iterrows():
        ids = row[class_columns].values.astype(bool)
        sample_classes = list(class_columns[ids])
        count = row['count']
        label = row[label_column] if label_column is not None and label_column in row.index else None
        entry = DataEntry(sample_classes, count, label=label)
        pos, annot_pos = entry.pick_position(available_positions, bgc_class2position, num_entries)
        # print(pos, annot_pos)
        # convert to coordinates and add to entry
        entry.leaf_tpos = theta_positions[pos]
        entry.annot_tpos = [theta_positions[p] for p in annot_pos]
        assert not entry.has_one_class, str(row)
        entry.leaf_size = 5
        max_r_level += 1
        entry.annot_rpos = r_start + max_r_level * r_step_size
        entry.leaf_rpos = r_end + r_leaf_offset

        all_entries.append(entry)

    return all_entries, theta_step_size


def get_arc_points(entry: DataEntry, theta_step_size: float = 0.1):
    if entry.has_one_class:
        return

    for a_point in entry.annot_tpos:
        #print()
        min_dist = None
        b_closest = None
        for b_point in entry.annot_tpos:
            #print(a_point, b_point)
            if np.abs(a_point - b_point) < theta_step_size/4: 
                #print(a_point, b_point)
                continue
            dist_value = min((a_point - b_point) % (2*np.pi), (b_point - a_point) % (2*np.pi))
            #print('dist', dist_value, min_dist)
            if min_dist is None or dist_value < min_dist:
                #print("set new value", min_dist, b_closest, dist_value, b_point)
                min_dist = dist_value
                b_closest = b_point
        # print("b_closest", a_point, b_closest)
        # get coordinates and draw plot segment
        # simple version
        if a_point > b_closest:
            if (a_point - b_closest) % (2*np.pi) < (b_closest - a_point) % (2*np.pi):
                theta_points = list(np.arange(b_closest,a_point, theta_step_size/4)) + [a_point]
            else:
                #print(b_closest, a_point, b_closest+2*np.pi, "---", (a_point - b_point) % (2*np.pi) , (b_point - a_point) % (2*np.pi))
                theta_points = list(np.arange(a_point, b_closest + 2*np.pi, theta_step_size/4)) + [b_closest]
                pass
        else:
            if (a_point - b_closest) % (2*np.pi) < (b_closest - a_point) % (2*np.pi):
                theta_points = list(np.arange(b_closest, a_point+2*np.pi,theta_step_size/4)) + [a_point]
                pass
            else:
                theta_points = list(np.arange(a_point, b_closest, theta_step_size/4)) + [b_closest]
                pass
            #theta_points = list(np.arange(a_point, b_closest, theta_step_size/4)) + [b_closest]
        rho_points = [entry.annot_rpos]*len(theta_points)
        yield theta_points, rho_points
        # ax_polar.plot(theta_points, rho_points, color=entry.color, linewidth=1)
    tn = min(entry.annot_tpos, key=lambda x: min((x - entry.leaf_tpos) % (2*np.pi), (entry.leaf_tpos-x) % (2*np.pi)))
    if tn < entry.leaf_tpos:
        if (tn - entry.leaf_tpos) % (2*np.pi) <  (entry.leaf_tpos - tn) % (2*np.pi):
            tn += 2*np.pi
            theta_points = list(np.arange(entry.leaf_tpos, tn, theta_step_size/4))+[tn]
            pass
        else:
            theta_points = list(np.arange(tn, entry.leaf_tpos, theta_step_size/4))+[entry.leaf_tpos]
            pass
        # theta_points = list(np.arange(tn, entry.leaf_tpos, theta_step_size/4))+[entry.leaf_tpos]
    else:
        if (tn - entry.leaf_tpos) % (2*np.pi) <  (entry.leaf_tpos - tn) % (2*np.pi):
            theta_points = list(np.arange(tn, entry.leaf_tpos, -theta_step_size/4))+[entry.leaf_tpos]
            pass
        else:
            theta_points = list(np.arange(tn, entry.leaf_tpos + 2*np.pi, theta_step_size/4))+[entry.leaf_tpos]
        # theta_points = list(np.arange(tn, entry.leaf_tpos, -theta_step_size/4))+[entry.leaf_tpos]
    rho_points = [entry.annot_rpos]*len(theta_points)
    yield theta_points, rho_points



def draw_flowerplot_inner(data_entries: List[DataEntry], 
                          mpl_cmap: Colormap = None, 
                          markers=None, 
                          savepath=None, 
                          title=None, 
                          theta_step_size=1, 
                          ax_polar=None):
    if markers is None:
        markers = get_flower_markers()
    if mpl_cmap is None:
        palette = sns.color_palette("pastel") #, n_colors=7)
        #palette_with_gray = sns.color_palette(palette[:6] +['#cfcfcf'])
        #list(palette_with_gray.as_hex())
        # palette_with_gray
        mpl_cmap = ListedColormap(palette.as_hex())
    flower_marker, flower_marker2 = markers
    if ax_polar is None:
        fig, ax_polar = plt.subplots(
            subplot_kw={'polar': True, 'clip_on':False}
        ) # note we must use plt.subplots, not plt.subplot

    for entry in data_entries:
        # draw classes
        if entry.has_one_class:
            entry.color = mpl_cmap(entry.bgc_class_index)
            entry.marker = 'o'
            entry.marker_size = 15

        # for each point in annot_tpos: find in entry.annot_tpos the closest and not equal one, connect with line segment, repeat
        # entry.annot_tpos
        if not entry.has_one_class:
            for theta_points, rho_points in get_arc_points(entry, theta_step_size=theta_step_size):
                ax_polar.plot(theta_points, rho_points, color=entry.color, linewidth=1)
        ax_polar.scatter(
            entry.annot_tpos, [entry.annot_rpos]*len(entry.annot_tpos), 
            color=entry.color, marker=entry.marker, 
            s=entry.marker_size**2//4, linewidth=1,
            zorder=2.5)

    for entry in sorted(data_entries, key=lambda x: x.has_one_class):
        # draw line and leaf
        leaf_size = entry.marker_size # + entry.count
        if entry.count > 50:
            leaf_size += 13 + (entry.count//300)*10
        leaf_rcenter = entry.leaf_rpos
        if entry.count > 100:
            leaf_rcenter += 0.1*(entry.count>100)
        ax_polar.plot([entry.leaf_tpos]*2, [entry.annot_rpos, leaf_rcenter], color=entry.color, zorder=2.1)
        if entry.has_one_class:
            rotation_transform = Affine2D().rotate_deg(90).scale(1.5, 3).rotate_deg(np.rad2deg(entry.leaf_tpos)+90)
            marker = MarkerStyle(flower_marker2, transform=rotation_transform)
        else: 
            marker = 'o'
            rotation_transform = Affine2D().rotate_deg(90).scale(1.5, 3).rotate_deg(np.rad2deg(entry.leaf_tpos)+90)
            marker = MarkerStyle(flower_marker, transform=rotation_transform)

        ax_polar.scatter([entry.leaf_tpos], [leaf_rcenter], marker=marker, s=leaf_size**2, color=entry.color)
        text = entry.count if entry.label is None else entry.label
        if entry.has_one_class:
            ax_polar.annotate(
                text, 
                (entry.leaf_tpos, leaf_rcenter), 
                horizontalalignment='center', 
                verticalalignment='center', 
                fontsize=24,)
            # print(entry.biosyn_classes[0])
        else:
            ax_polar.annotate(
                text, 
                (entry.leaf_tpos, entry.leaf_rpos+ 0.15+ 0.25*(entry.count> 100)), 
                horizontalalignment='center',
                verticalalignment='center',
                color=POSTER_BLUE,
                # xytext=(0, 0.1),
                # textcoords='offset fontsize',
                xycoords='data',
                fontsize=18,
                #rotation=np.rad2deg(entry.leaf_tpos)
                # fc='red'
            )
            # ax_polar.text(entry.leaf_tpos, entry.leaf_rpos+0.5, entry.count, ha='center', va='center')
    
    ax_polar.scatter([0], [0], s=100**2, color=POSTER_BLUE)
    if title is not None:
        ax_polar.annotate(title, [0,0], color='white', ha='center', va='center')
    ax_polar.grid(False)
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.set_ylim((0, 4))
    # ax_polar.set_ylim()
    ax_polar.spines['polar'].set_visible(False)
    # ax_polar.spines['polar'].set_edgecolor(POSTER_BLUE)
    plt.tight_layout()
    # legend = ax_polar.legend()
    # handles, labels = ax_polar.get_legend_handles_labels()
    # legend.remove()
    # path = Path(".") / f"flowerplot.png"
    if savepath is not None:
        fig.savefig(savepath, dpi=50, bbox_inches="tight", transparent=True)
    return ax_polar
    # plt.show()


def draw_flowerplot(data_df: pd.DataFrame, 
                    class_columns: list[str], 
                    mpl_cmap: Colormap = None, 
                    markers: tuple[mpath.Path, mpath.Path] = None, 
                    title: str = None, 
                    ax_polar: plt.Axes = None, 
                    label_column="label",
                    debug: bool = False) -> plt.Axes: 
    df = data_df.copy()
    if isinstance(class_columns, list):
        class_columns = np.asarray(class_columns)
    if not "num_classes" in df.columns:
        df["num_classes"] = df[class_columns].sum(1)
    entries, theta_step_size = extract_data(
        df, 
        class_columns, 
        label_column=label_column, 
        debug=debug
    )
    ax = draw_flowerplot_inner(
        entries, 
        mpl_cmap=mpl_cmap, 
        markers=markers, 
        savepath=None, 
        title=title, 
        theta_step_size=theta_step_size,
        ax_polar=ax_polar
    )
    return ax