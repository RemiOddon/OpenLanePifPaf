import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf



NUMBER_KEYPOINTS = 25

LANE_KEYPOINTS = [str(x) for x in range(1,NUMBER_KEYPOINTS+1)]

LANE_SKELETON = [(x, x+1) for x in range(1,NUMBER_KEYPOINTS)]

LANE_SIGMAS = [0.05] * NUMBER_KEYPOINTS # !!!! 0.05 defined like in for apollo dataset but does it make sense ??

LANE_CATEGORIES = ['lane']

LANE_WEIGHTS = [1.0] * NUMBER_KEYPOINTS #!!! arbitrary

LANE_KEYPOINTS_REVERSED = LANE_KEYPOINTS.copy()
LANE_KEYPOINTS_REVERSED.reverse()
HFLIP = { x : y for x, y in zip(LANE_KEYPOINTS, LANE_KEYPOINTS_REVERSED)}

LANE_POSE = np.array([[ 12.41126658,  16.48739856,  20.55770894,  24.61052013,
         28.67932284,  32.7014157 ,  36.75407828,  40.77249745,
         44.80880103,  48.86299185,  52.93246616,  57.00114084,
         61.05549988,  65.12093974,  69.1802232 ,  73.25429169,
         77.27043152,  81.33252536,  85.38659938,  89.43784533,
         93.44103355,  97.47088613, 101.5303368 , 105.55162999,
        109.61944657],
       [  1.51427833,   1.47890974,   1.45933463,   1.42592057,
          1.39396408,   1.35032604,   1.31602426,   1.28355886,
          1.2530039 ,   1.23225756,   1.19862066,   1.16022512,
          1.12766988,   1.08854834,   1.04766283,   1.03270131,
          0.99644134,   0.99526874,   0.97650612,   0.93945108,
          0.91786379,   0.89553675,   0.87606017,   0.80161687,
          0.86501082],
       [ -2.12752483,  -2.126506  ,  -2.09083744,  -2.13745813,
         -2.15233696,  -2.15159873,  -2.14894286,  -2.15249341,
         -2.08631737,  -2.18919669,  -2.19562271,  -2.20087263,
         -2.20772907,  -2.20020498,  -2.16161404,  -2.21983492,
         -2.20567516,  -2.30103709,  -2.29675483,  -2.3241096 ,
         -2.24528041,  -2.27585565,  -2.29368062,  -2.28118656,
         -2.31360889]]).T

def get_constants():
    return [LANE_KEYPOINTS, LANE_SKELETON, HFLIP, LANE_SIGMAS,
            LANE_POSE, LANE_CATEGORIES, LANE_WEIGHTS]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_lane.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the lane skeleton")
    for j1, j2 in LANE_SKELETON:
        print(LANE_KEYPOINTS[j1 - 1], '-', LANE_KEYPOINTS[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(LANE_POSE, sigmas = LANE_SIGMAS, skel = LANE_SKELETON,
#                    kps = LANE_KEYPOINTS, scr_weights = LANE_WEIGHTS)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_66 = plot3d_red(ax_2D, LANE_POSE, LANE_SKELETON)
        anim_66.save('openpifpaf/plugins/openlane/docs/LANE_pose.gif', fps=30)


if __name__ == '__main__':
    main()
