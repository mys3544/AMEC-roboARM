"""Detection and grasp planning. No hardware, no camera -- runs anywhere.

Everything here is the part that decides WHETHER and WHERE to move, which is the
part worth catching in a test. The moving itself is covered by test_arm_safety.py.
"""

import math

import cv2
import numpy as np
import pytest

from roboarm import config as cfg
from roboarm import detect, grasp
from roboarm import kinematics as kin
from roboarm import workspace as ws

# One pixel per millimetre, so a 30 px square in a test image is a 30 mm object and
# the expected numbers can be read straight off. The real homography is a projection
# and is not uniform like this; that is exactly why detect.py measures widths AFTER
# mapping to the table rather than in pixels.
MM_PER_PIXEL = np.diag([0.001, 0.001, 1.0])

# Middle of the pick strip: reachable with the fingers open, and inside the survey view.
GOOD_X, GOOD_Y = 0.160, 0.020


def target(width_m=0.030, length_m=0.030, x=GOOD_X, y=GOOD_Y, label="test"):
    return detect.Target(x=x, y=y, width_m=width_m, length_m=length_m,
                         angle_deg=0.0, label=label)


def scene(boxes):
    """A black table with white boxes on it. Returns (frame, background)."""
    background = np.zeros((480, 640, 3), np.uint8)
    frame = background.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return frame, background


# ---------------------------------------------------------------- detection --
def test_a_box_on_the_table_is_found_at_the_right_size():
    frame, background = scene([(200, 150, 30, 40)])
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].width_m == pytest.approx(0.030, abs=0.003)
    assert found[0].length_m == pytest.approx(0.040, abs=0.003)
    # Centre of the drawn box, in "metres" under this deliberately trivial mapping.
    assert found[0].x == pytest.approx(0.215, abs=0.003)
    assert found[0].y == pytest.approx(0.170, abs=0.003)


def test_an_unchanged_table_yields_nothing():
    _frame, background = scene([])
    assert detect.changes(background, background, MM_PER_PIXEL) == []


def test_several_objects_are_reported_nearest_first():
    frame, background = scene([(400, 100, 30, 30), (100, 100, 30, 30), (250, 100, 30, 30)])
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 3
    assert [round(t.x, 3) for t in found] == sorted(round(t.x, 3) for t in found)


def test_speckle_is_not_reported_as_an_object():
    """Sensor noise survives thresholding; it must not survive the morphology."""
    frame, background = scene([])
    rng = np.random.default_rng(0)
    for x, y in rng.integers(20, 400, size=(60, 2)):
        frame[y, x] = 255
    assert detect.changes(frame, background, MM_PER_PIXEL) == []


def test_something_far_too_small_is_seen_but_refused():
    frame, background = scene([(200, 150, 16, 16)])
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 1, "it should be detected..."
    assert not found[0].graspable, "...and then refused on size"
    assert "too thin" in found[0].why_not()


def test_annotation_puts_the_marker_back_where_the_object_was():
    """Round-trips through the inverse homography, so a mapping error would show."""
    frame, background = scene([(200, 150, 30, 30)])
    found = detect.changes(frame, background, MM_PER_PIXEL)
    drawn = detect.annotate(frame, MM_PER_PIXEL, found)
    assert drawn.shape == frame.shape
    # The green circle lands on the box, not somewhere else in the image.
    patch = drawn[150:181, 200:231]
    assert (patch[:, :, 1] > 150).any()


# ----------------------------------------------------------------- planning --
def test_the_gripper_opens_fully_and_the_descent_allows_for_the_close():
    """Fully open for the approach (alignment error is a few mm; a tight opening
    left a finger landing ON the cube), and the grasp pose sits higher by the
    distance the tips travel down while shutting, so they finish at GRASP_HEIGHT_M."""
    step = grasp.plan(target(), reach_offset_m=0.0)
    assert step.opening == cfg.GRIPPER_OPEN
    travel = grasp.closing_travel(step.opening, target().width_m)
    assert travel > 0.010, "closing from fully open moves the tips down by centimetres"
    z = kin.forward(step.grasp)[2]
    assert z == pytest.approx(-cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M + travel, abs=0.013)

def test_hover_and_grasp_share_one_pitch():
    """Otherwise the tool swings through an arc instead of descending."""
    step = grasp.plan(target())
    assert kin.tool_pitch(step.hover) == pytest.approx(kin.tool_pitch(step.grasp), abs=1.5)


def test_hover_is_above_the_grasp_and_both_are_above_the_table():
    step = grasp.plan(target())
    assert kin.forward(step.hover)[2] > kin.forward(step.grasp)[2]
    assert kin.forward(step.grasp)[2] > -cfg.TABLE_BELOW_PLATE


def test_both_planned_poses_carry_the_opening_they_were_solved_for():
    step = grasp.plan(target())
    assert step.hover[cfg.GRIPPER_ID] == step.opening
    assert step.grasp[cfg.GRIPPER_ID] == step.opening


def test_the_grasp_lands_on_the_object():
    """The raw model, with no reach offset: the IK lands where it was sent."""
    x, y, z = kin.forward(grasp.plan(target(), reach_offset_m=0.0).grasp)
    assert x == pytest.approx(GOOD_X, abs=0.013)
    assert y == pytest.approx(GOOD_Y, abs=0.013)
    travel = grasp.closing_travel(cfg.GRIPPER_OPEN, target().width_m)
    assert z == pytest.approx(-cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M + travel, abs=0.013)


@pytest.mark.parametrize("width, why", [(0.010, "too thin"), (0.080, "wider than")])
def test_planning_refuses_objects_the_gripper_cannot_hold(width, why):
    with pytest.raises(grasp.GraspError, match=why):
        grasp.plan(target(width_m=width))


def test_planning_refuses_an_object_out_of_reach():
    with pytest.raises(grasp.GraspError, match="cannot be grasped"):
        grasp.plan(target(x=0.400))


def test_planning_happens_before_anything_moves():
    """plan() must be pure: it is what lets an unreachable object be a message
    instead of an arm that sets off and stops halfway."""
    assert grasp.plan(target()) is not None


# ------------------------------------------------------------ gripper rules --
def test_moves_never_command_the_gripper_once_something_is_held():
    """The rule that keeps a servo from stalling against a held object."""
    pose = {1: 90, 2: 45, 3: 90, 4: 90, 5: 90, 6: 120}
    assert cfg.GRIPPER_ID not in grasp._arm_only(pose)
    assert set(grasp._arm_only(pose)) == {1, 2, 3, 4, 5}


def test_the_verifiable_width_floor_matches_what_grasped_can_detect():
    """MIN_WIDTH_M exists so grasped() can tell a held object from an empty close.

    An object stops the fingers at whatever gap it is wide; grasped() only believes
    it below GRIPPER_CLOSED minus the readback tolerance. Check the smallest allowed
    object really does clear that threshold.
    """
    gaps = sorted(cfg.GRIPPER_GAP_MM.items())
    angle = max(a for a, gap in gaps if gap / 1000 >= detect.MIN_WIDTH_M)
    assert angle < cfg.GRIPPER_CLOSED - cfg.READBACK_TOLERANCE_DEG


def test_the_width_ceiling_leaves_room_for_the_approach_clearance():
    widest = max(cfg.GRIPPER_GAP_MM.values()) / 1000
    assert detect.MAX_WIDTH_M + grasp.FINGER_CLEARANCE_M <= widest


def test_the_plan_turns_the_fingers_across_the_narrow_way():
    """A 30 x 60 mm bar lying at +40 deg: the fingers must close along +40."""
    step = grasp.plan(detect.Target(x=GOOD_X, y=GOOD_Y, width_m=0.030, length_m=0.060,
                                    angle_deg=40.0, label="bar"))
    assert kin.finger_heading(step.grasp) == pytest.approx(40.0, abs=0.6)
    assert step.hover[5] == step.grasp[5] == step.roll


def test_the_plan_undoes_the_base_yaw_for_a_cube_off_to_the_side():
    """Square to the table at bearing -25: without the roll the fingers would
    close at -25 deg to its faces, corner first."""
    bearing = math.radians(-25)
    cube = detect.Target(x=0.150 * math.cos(bearing), y=0.150 * math.sin(bearing),
                         width_m=0.040, length_m=0.040, angle_deg=0.0, label="cube")
    step = grasp.plan(cube)
    assert step.grasp[1] != 90, "the base had to yaw to reach it"
    assert kin.finger_heading(step.grasp) % 90 == pytest.approx(0.0, abs=0.6)
    assert abs(step.roll - cfg.J5_FINGERS_ACROSS_ARM) <= 45, "a cube takes the small turn"


def test_the_aim_point_is_beyond_the_object_along_its_bearing():
    """The real tips land short along the reach, so the aim goes further out on
    the same bearing -- never sideways."""
    seen = target(x=0.120, y=0.090, width_m=0.080)  # bearing 36.9 deg, 150 mm out
    aimed = grasp.aim_for(seen, 0.020)
    assert math.hypot(aimed.x, aimed.y) == pytest.approx(0.170, abs=1e-6)
    assert math.atan2(aimed.y, aimed.x) == pytest.approx(math.atan2(seen.y, seen.x))
    assert aimed.width_m == seen.width_m and aimed.label == seen.label
    assert grasp.aim_for(seen, 0.0) == seen


def test_the_aim_never_goes_beyond_a_quarter_of_the_objects_width():
    """A 20 mm cube at 173 mm: the grown offset says 9 mm, which put the pads on
    its far edge and shoved it (2026-09-21). Capped at 5 mm; a 40 mm cube keeps 9."""
    small = target(x=0.168, y=0.040, width_m=0.020, length_m=0.020)
    assert grasp.reach_offset(small.x, small.y, 0.005) == pytest.approx(0.0088, abs=0.0005)
    assert grasp.aim_offset(small, 0.005) == pytest.approx(0.005)
    big = target(x=0.168, y=0.040, width_m=0.040, length_m=0.040)
    assert grasp.aim_offset(big, 0.005) == pytest.approx(0.0088, abs=0.0005)
    assert grasp.aim_offset(small, 0.0) == 0.0


def test_the_plan_aims_past_the_object_by_the_reach_offset():
    cube = target(width_m=0.040, length_m=0.040)   # a 30 mm one would cap the aim at 7.5
    raw = grasp.plan(cube, reach_offset_m=0.0)
    offset = grasp.plan(cube, reach_offset_m=0.010)
    tip_raw = kin.forward(raw.grasp)
    tip_off = kin.forward(offset.grasp)
    further = math.hypot(*tip_off[:2]) - math.hypot(*tip_raw[:2])
    assert further == pytest.approx(0.010, abs=0.004), "IK rounds to whole degrees"
    assert cfg.REACH_OFFSET_M == pytest.approx(0.005)   # touch-probed 2026-09-14


def test_the_lift_keeps_the_pitch_the_grasp_used():
    """Otherwise the tool swings and drags the object sideways before it clears."""
    step = grasp.plan(target())
    assert step.pitch == pytest.approx(kin.tool_pitch(step.grasp), abs=1.0)


def test_the_squeeze_is_relaxed_rather_than_left_at_full_travel():
    """A servo commanded far past a stop draws full current until it is released."""
    assert 0 < grasp.SQUEEZE_DEG < cfg.READBACK_TOLERANCE_DEG * 3


# ------------------------------------------------------------- shadows --
def shadowed_scene(box, shadow, shade=0.55):
    """White paper, a coloured box, and a soft shadow beside it.

    The shadow is the background SCALED DOWN in brightness, which is what a real one
    is -- it must not be a flat grey patch, or the test would pass on a filter that
    only rejects one specific colour.
    """
    background = np.full((480, 640, 3), 235, np.uint8)
    frame = background.copy()
    x, y, w, h = shadow
    frame[y:y + h, x:x + w] = (background[y:y + h, x:x + w] * shade).astype(np.uint8)
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 90, 30), -1)  # a blue-ish box
    return frame, background


def test_a_shadow_is_not_reported_as_part_of_the_object():
    """The failure that stopped a real pick: a 40 mm cube measured 55 x 113 mm and
    was refused as wider than the gripper opens -- rejected on its own shadow."""
    frame, background = shadowed_scene(box=(200, 150, 40, 40), shadow=(200, 190, 40, 70))
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].length_m == pytest.approx(0.040, abs=0.008), "the box, not box+shadow"
    assert found[0].graspable


def test_the_shadow_would_have_ruined_it_without_the_filter():
    """Guards the guard: if the scene were not actually shadowed, the test above
    would pass for the wrong reason."""
    frame, background = shadowed_scene(box=(200, 150, 40, 40), shadow=(200, 190, 40, 70))
    lit = detect.foreground(frame, background)
    naive = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
    assert (naive > detect.CHANGE_THRESHOLD).sum() > lit.sum() / 255 * 1.5


def test_a_dark_object_survives_the_shadow_filter():
    """Shadows dim; they do not blacken. An object darker than SHADOW_DARKEST is
    kept, which is the only thing protecting a colourless dark object."""
    background = np.full((480, 640, 3), 235, np.uint8)
    frame = background.copy()
    cv2.rectangle(frame, (200, 150), (240, 190), (20, 20, 20), -1)
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].graspable


def test_a_brighter_room_is_not_a_table_full_of_objects():
    """The reference photo ages. Measured: bare paper read 12% brighter than its own
    reference a few hours later, which widens every silhouette."""
    background = np.full((480, 640, 3), 190, np.uint8)
    brighter = np.full((480, 640, 3), 214, np.uint8)   # the 1.12 actually measured
    assert detect.light_scale(brighter, background) == pytest.approx(1.12, abs=0.02)
    assert detect.changes(brighter, background, MM_PER_PIXEL) == []


def test_brightness_matching_does_not_hide_a_real_object():
    background = np.full((480, 640, 3), 190, np.uint8)
    frame = np.full((480, 640, 3), 214, np.uint8)
    cv2.rectangle(frame, (200, 150), (240, 190), (200, 90, 30), -1)
    found = detect.changes(frame, background, MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].width_m == pytest.approx(0.040, abs=0.006)


def test_light_scale_ignores_the_objects_themselves():
    """Median, not mean: a big dark object must not drag the estimate down."""
    background = np.full((480, 640, 3), 200, np.uint8)
    frame = background.copy()
    cv2.rectangle(frame, (0, 0), (250, 250), (10, 10, 10), -1)   # a sixth of the frame
    assert detect.light_scale(frame, background) == pytest.approx(1.0, abs=0.02)


# ------------------------------------------------- reaching a point, not a pose --
# MEASURED on the robot 2026-09-10 at 176 mm reach, tool 10 degrees off vertical:
# the joints land within ONE degree of the commanded pose -- converged by any
# joint-space test -- and the fingertip is still 9.9 mm short of the point that
# was asked for. A joint-space droop loop recovered only 3.2 mm of that, because a
# 1 degree error is inside any honest deadband; correcting in millimetres recovered
# 6.8 mm and put the height right as well (3.0 mm above the table when 8 mm was
# asked, corrected to 8.8 mm).
#
# These tests use a stub arm that sags by a fixed amount, which is what the real one
# does, so they check the CORRECTION rather than the servo.

class SaggingArm:
    """An arm whose joints always settle a degree or two below what was asked."""

    def __init__(self, sag=None, start=None):
        self.sag = sag if sag is not None else {2: -1, 3: -1}
        self.pose = dict(start or cfg.HOME_POSE)
        self.commanded = []

    def move_to(self, targets, speed_dps=40.0):
        self.commanded.append(dict(targets))
        for joint, angle in targets.items():
            self.pose[joint] = angle + self.sag.get(joint, 0)
        return dict(self.pose)

    def read(self, attempts=4):
        return dict(self.pose)


def reached(arm, gripper):
    return kin.forward({**arm.read(), cfg.GRIPPER_ID: gripper})


def test_a_sagging_arm_lands_short_when_nobody_corrects_it():
    """The defect, stated as a test: every joint in tolerance, the tip still short."""
    arm = SaggingArm()
    opening = cfg.GRIPPER_CLOSED
    x, y, z = 0.176, 0.0, -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    pose, _pitch = kin.solve(x, y, z, gripper=opening)
    arm.move_to({j: a for j, a in pose.items() if j != cfg.GRIPPER_ID})
    got = reached(arm, opening)
    assert math.dist(got[:2], (x, y)) > 0.004, "a degree of sag really is millimetres"


def test_reaching_a_point_corrects_what_the_pose_alone_cannot(monkeypatch):
    monkeypatch.setattr(grasp.time, "sleep", lambda _s: None)
    opening = cfg.GRIPPER_CLOSED
    x, y, z = 0.176, 0.0, -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    _pose, pitch = kin.solve(x, y, z, gripper=opening)

    open_loop = SaggingArm()
    open_loop.move_to({j: a for j, a in _pose.items() if j != cfg.GRIPPER_ID})
    before = math.dist(reached(open_loop, opening)[:2], (x, y))

    corrected = SaggingArm()
    grasp._reach_to(corrected, x, y, z, pitch, opening, speed_dps=8)
    after = math.dist(reached(corrected, opening)[:2], (x, y))

    assert after < before, f"correction made it worse: {before:.4f} -> {after:.4f}"
    # Not zero, and cannot be: the IK rounds to whole servo degrees, and at this
    # pitch one degree is 3.2 mm of reach, so half a degree of residual is the
    # floor. The real arm measured 3.1 mm left after correcting, from 9.9 mm.
    assert after <= 0.004, f"still {after * 1000:.1f} mm out after correcting"


def test_reaching_a_point_fixes_the_height_too():
    """The sag costs height as well as reach, and both were measured wrong."""
    opening = cfg.GRIPPER_CLOSED
    x, y = 0.176, 0.0
    z = -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    _pose, pitch = kin.solve(x, y, z, gripper=opening)
    arm = SaggingArm()
    grasp._reach_to(arm, x, y, z, pitch, opening, speed_dps=8, passes=2)
    assert reached(arm, opening)[2] == pytest.approx(z, abs=0.003)


def test_reaching_a_point_never_commands_the_gripper():
    """It must be safe to run with something held: only the arm joints may move."""
    arm = SaggingArm()
    x, y = 0.170, 0.010
    z = -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    _pose, pitch = kin.solve(x, y, z, gripper=cfg.GRIPPER_CLOSED)
    grasp._reach_to(arm, x, y, z, pitch, cfg.GRIPPER_CLOSED, speed_dps=8)
    assert arm.commanded, "it did not move at all"
    for sent in arm.commanded:
        assert cfg.GRIPPER_ID not in sent


def test_an_arm_that_lands_perfectly_is_left_alone():
    """No sag, no correction: it must not fidget once it is already there."""
    arm = SaggingArm(sag={})
    x, y = 0.170, 0.010
    z = -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    _pose, pitch = kin.solve(x, y, z, gripper=cfg.GRIPPER_CLOSED)
    grasp._reach_to(arm, x, y, z, pitch, cfg.GRIPPER_CLOSED, speed_dps=8)
    assert len(arm.commanded) == 1


def test_correcting_off_the_edge_of_the_workspace_stops_rather_than_raising():
    """A correction that walks the aim out of reach leaves the arm where it is.

    Raising here would abandon the arm mid-pick, which is worse than an imperfect
    position -- the caller can still close the fingers and report an honest miss.
    """
    arm = SaggingArm(sag={2: -25, 3: -25})   # absurd sag, so the aim runs away
    x, y = 0.205, 0.0
    z = -cfg.TABLE_BELOW_PLATE + grasp.GRASP_HEIGHT_M
    _pose, pitch = kin.solve(x, y, z, gripper=cfg.GRIPPER_CLOSED)
    grasp._reach_to(arm, x, y, z, pitch, cfg.GRIPPER_CLOSED, speed_dps=8)


# ------------------------------------------------------------ the colour rung --
# The rung that needs neither an empty-table photograph nor the tag facing up. It
# earned its place on 2026-09-10, when the box was knocked onto its side mid-demo:
# the tag went face-down, and change detection cannot sweep a ring because its
# background is per-station. Colour could still find it from every station.

def colour_scene(boxes, bgr=(190, 60, 40)):
    """A near-neutral table with strongly coloured boxes on it.

    180-grey is what the real table reads as: bright, and almost unsaturated. The
    box colour is a saturated blue, as the lab's is.
    """
    frame = np.full((480, 640, 3), 180, np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, -1)
    return frame


def test_a_coloured_box_is_found_at_the_right_place_and_size():
    frame = colour_scene([(200, 150, 40, 50)])
    found = detect.coloured(frame, MM_PER_PIXEL)
    assert len(found) == 1
    assert found[0].width_m == pytest.approx(0.040, abs=0.003)
    assert found[0].length_m == pytest.approx(0.050, abs=0.003)
    assert found[0].x == pytest.approx(0.220, abs=0.003)
    assert found[0].y == pytest.approx(0.175, abs=0.003)


def test_a_grey_box_is_not_a_coloured_one():
    """The whole point: a shadow is a darker table, and has no colour of its own.

    A brightness test cannot tell those apart. This one does not even see it.
    """
    frame = colour_scene([(200, 150, 40, 50)], bgr=(90, 90, 90))
    assert detect.coloured(frame, MM_PER_PIXEL) == []


def test_the_table_alone_yields_nothing():
    assert detect.coloured(colour_scene([]), MM_PER_PIXEL) == []


def test_speckle_too_small_to_grasp_is_dropped():
    frame = colour_scene([(300, 200, 4, 4)])
    assert detect.coloured(frame, MM_PER_PIXEL) == []


def test_two_boxes_come_back_nearest_first():
    frame = colour_scene([(400, 300, 30, 30), (100, 80, 30, 30)])
    found = detect.coloured(frame, MM_PER_PIXEL)
    assert len(found) == 2
    first, second = found
    assert math.hypot(first.x, first.y) < math.hypot(second.x, second.y)


def test_standing_above_the_table_is_corrected_towards_the_nadir():
    """Same relation as _unlift, for an object that cannot measure its own size."""
    frame = colour_scene([(300, 200, 40, 40)])
    nadir = (0.100, 0.100)
    flat = detect.coloured(frame, MM_PER_PIXEL)[0]
    lifted = detect.coloured(frame, MM_PER_PIXEL, nadir=nadir, height_m=0.028)[0]
    assert (math.dist((lifted.x, lifted.y), nadir)
            < math.dist((flat.x, flat.y), nadir)), "must move TOWARDS the lens"
    # and stay on the line between the two, not wander off it
    along = (flat.x - nadir[0], flat.y - nadir[1])
    moved = (lifted.x - nadir[0], lifted.y - nadir[1])
    assert along[0] * moved[1] - along[1] * moved[0] == pytest.approx(0.0, abs=1e-9)


def test_correcting_for_height_without_a_nadir_is_refused():
    """Silently correcting about the middle of the picture is the 20 mm bug."""
    frame = colour_scene([(300, 200, 40, 40)])
    with pytest.raises(ValueError, match="nadir"):
        detect.coloured(frame, MM_PER_PIXEL, height_m=0.028)


def test_nothing_on_the_table_needs_no_nadir():
    frame = colour_scene([(300, 200, 40, 40)])
    assert len(detect.coloured(frame, MM_PER_PIXEL, height_m=0.0)) == 1


# ------------------------------------------------------------- cube ranging --
# A cube's outline on the table plane is its base plus its top magnified about
# the point under the lens, joined by the side faces the lens can see. Its height
# is its width, so the outline alone says how big and where it is -- no tag, no
# background, no declared size. These render exactly that outline and ask
# detect.range_block() to undo it.

NADIR = (0.050, 0.050)     # pixel (50, 50) under MM_PER_PIXEL
LENS_M = 0.212             # the calibrated survey pose's lens height


def cube_range(outline, nadir, lens_m):
    """(footprint quad, edge length, agreement) -- range_block without the height,
    which for a cube is the edge length again."""
    quad, width, _height, agreement = detect.range_block(outline, nadir, lens_m)
    return quad, width, agreement


def cube_outline(x, y, size, yaw_deg=0.0, nadir=NADIR, lens_m=LENS_M):
    """Table-plane outline of a cube: hull of its base and its magnified top."""
    half = size / 2
    yaw = math.radians(yaw_deg)
    spin = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    base = np.array([[x, y]]) + np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]]) @ spin.T
    grown = lens_m / (lens_m - size)
    top = np.asarray(nadir) + (base - np.asarray(nadir)) * grown
    both = np.vstack([base, top]).astype(np.float32)
    return cv2.convexHull(both).reshape(-1, 2).astype(float)


def hull_scene(cubes, bgr=(190, 60, 40), nadir=NADIR, lens_m=LENS_M):
    """A colourless table with cubes drawn the way the wrist camera sees them."""
    frame = np.full((480, 640, 3), 180, np.uint8)
    inverse = np.linalg.inv(MM_PER_PIXEL)
    for x, y, size, *rest in cubes:
        yaw = rest[0] if rest else 0.0
        pixels = ws.apply(inverse, cube_outline(x, y, size, yaw, nadir, lens_m))
        cv2.fillPoly(frame, [pixels.astype(np.int32)], bgr)
    return frame


@pytest.mark.parametrize("size", [0.022, 0.040, 0.055])
@pytest.mark.parametrize("x, y", [(0.300, 0.250), (0.150, 0.300), (0.400, 0.120)])
def test_cube_range_recovers_size_and_position(size, x, y):
    quad, found, agreement = cube_range(cube_outline(x, y, size), NADIR, LENS_M)
    centre = quad.mean(axis=0)
    assert found == pytest.approx(size, abs=0.0015)
    assert centre[0] == pytest.approx(x, abs=0.002)
    assert centre[1] == pytest.approx(y, abs=0.002)
    assert agreement > 0.9


@pytest.mark.parametrize("yaw", [0, 12, 30, 45, 77])
def test_cube_range_copes_with_a_turned_cube(yaw):
    quad, found, agreement = cube_range(
        cube_outline(0.300, 0.250, 0.040, yaw_deg=yaw), NADIR, LENS_M)
    centre = quad.mean(axis=0)
    assert found == pytest.approx(0.040, abs=0.0015)
    assert centre[0] == pytest.approx(0.300, abs=0.002)
    assert centre[1] == pytest.approx(0.250, abs=0.002)
    assert agreement > 0.9


def test_the_uncorrected_outline_is_bigger_and_further_out():
    """What cube_range is FOR: the raw outline of a 40 mm cube reads well over
    40 mm and sits further from the nadir than the cube does."""
    outline = cube_outline(0.300, 0.250, 0.040)
    raw = detect._target_from_quad(cv2.boxPoints(cv2.minAreaRect(outline.astype(np.float32))), "raw")
    assert raw.width_m > 0.046
    assert math.dist((raw.x, raw.y), NADIR) > math.dist((0.300, 0.250), NADIR) + 0.010


def test_a_shadow_off_the_far_side_does_not_change_the_size():
    """The size is read across the outline, so a shadow trailing away from the
    lens along the radial lengthens the outline without widening it. The
    footprint no longer explains the whole outline, and the agreement says so."""
    outline = cube_outline(0.300, 0.250, 0.040)
    nadir = np.asarray(NADIR)
    away = (np.array([0.300, 0.250]) - nadir)
    away /= np.linalg.norm(away)
    shadow = outline + away * 0.030
    smeared = cv2.convexHull(np.vstack([outline, shadow]).astype(np.float32)).reshape(-1, 2)
    _quad, found, agreement = cube_range(smeared, NADIR, LENS_M)
    _q, _f, clean = cube_range(outline, NADIR, LENS_M)
    assert found == pytest.approx(0.040, abs=0.003)
    assert agreement < clean - 0.1


def test_a_nadir_error_barely_moves_the_answer():
    """The reading depends on the nadir only through the final unlift: 10 mm of
    nadir error is about 2 mm of position and nothing in size. (The near-face
    closed form this replaced lost the whole height to such an error.)"""
    outline = cube_outline(0.300, 0.250, 0.040)
    _q, size, _a = cube_range(outline, NADIR, LENS_M)
    quad, size_off, _a = cube_range(outline, (NADIR[0] + 0.010, NADIR[1]), LENS_M)
    assert size_off == pytest.approx(size, abs=0.0005)
    assert math.dist(quad.mean(axis=0), (0.300, 0.250)) < 0.003


def test_cube_range_refuses_a_degenerate_outline():
    with pytest.raises(ValueError):
        cube_range(np.array([[0.1, 0.1], [0.2, 0.2]]), NADIR, LENS_M)


def test_coloured_with_the_lens_height_sizes_and_places_the_cube():
    """At this scene's 1 px = 1 mm the mask's opening chamfers corners by
    millimetres; the real camera has four pixels to the millimetre, see the
    next test for what that buys."""
    frame = hull_scene([(0.300, 0.250, 0.040)])
    found = detect.coloured(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M)
    assert len(found) == 1
    cube = found[0]
    assert cube.width_m == pytest.approx(0.040, abs=0.002)
    assert cube.height_m == pytest.approx(0.040, abs=0.002)
    assert cube.x == pytest.approx(0.300, abs=0.005)
    assert cube.y == pytest.approx(0.250, abs=0.005)
    assert cube.graspable


@pytest.mark.parametrize("yaw", [0, 30, 60])
def test_at_the_real_cameras_pixel_size_the_reading_is_sub_millimetre(yaw):
    fine = np.diag([0.00025, 0.00025, 1.0])   # ~4 px/mm, as the wrist camera
    frame = np.full((480, 640, 3), 180, np.uint8)
    pixels = ws.apply(np.linalg.inv(fine), cube_outline(0.100, 0.060, 0.040, yaw))
    cv2.fillPoly(frame, [pixels.astype(np.int32)], (190, 60, 40))
    cube = detect.coloured(frame, fine, nadir=NADIR, lens_m=LENS_M)[0]
    assert not cube.clipped
    assert cube.width_m == pytest.approx(0.040, abs=0.001)
    assert cube.x == pytest.approx(0.100, abs=0.001)
    assert cube.y == pytest.approx(0.060, abs=0.001)


def test_cubes_of_every_size_come_out_their_own_size():
    frame = hull_scene([(0.300, 0.250, 0.025), (0.150, 0.300, 0.055), (0.450, 0.100, 0.040)])
    found = detect.coloured(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M)
    sizes = sorted(round(t.width_m, 3) for t in found)
    assert sizes == pytest.approx([0.025, 0.040, 0.055], abs=0.0015)


def test_the_fixed_height_path_is_still_there_for_things_that_are_not_cubes():
    frame = hull_scene([(0.300, 0.250, 0.040)])
    ranged = detect.coloured(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M)[0]
    fixed = detect.coloured(frame, MM_PER_PIXEL, nadir=NADIR, height_m=0.040)[0]
    assert fixed.height_m is None
    # Both undo the parallax; the fixed path keeps the side faces in the width.
    assert fixed.width_m > ranged.width_m


def test_objects_ranges_a_neural_outline_when_told_the_lens(monkeypatch):
    outline = cube_outline(0.300, 0.250, 0.040)
    pixels = ws.apply(np.linalg.inv(MM_PER_PIXEL), outline)
    x0, y0 = pixels.min(axis=0)
    x1, y1 = pixels.max(axis=0)
    box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    reply = {"detections": [{"label": "cube", "confidence": 0.8, "box": box,
                             "polygon": pixels.tolist()}]}
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: reply)
    frame = np.zeros((480, 640, 3), np.uint8)
    raw = detect.objects(frame, MM_PER_PIXEL, url="http://x")[0]
    ranged = detect.objects(frame, MM_PER_PIXEL, url="http://x", nadir=NADIR, lens_m=LENS_M)[0]
    assert raw.width_m > 0.046
    assert ranged.width_m == pytest.approx(0.040, abs=0.002)
    assert ranged.height_m == pytest.approx(0.040, abs=0.002)
    assert ranged.x == pytest.approx(0.300, abs=0.002)
    assert ranged.confidence == pytest.approx(0.8, abs=0.01)
    # A box alone is read as a cube's top, which is only roughly right: the box
    # takes in the near face too, so it reads big and is unlifted too far. Good
    # enough to know something is there; the outline is what a pick needs, and
    # the served model is a segmentation model precisely so that it is sent.
    reply["detections"][0].pop("polygon")
    boxed = detect.objects(frame, MM_PER_PIXEL, url="http://x", nadir=NADIR, lens_m=LENS_M)[0]
    assert boxed.x == pytest.approx(0.300, abs=0.060)
    assert 0.040 <= boxed.width_m < 0.070


def test_distinct_keeps_the_surest_of_each_object():
    a = detect.Target(0.300, 0.250, 0.040, 0.040, 0.0, "colour 1", confidence=0.95)
    b = detect.Target(0.305, 0.252, 0.046, 0.046, 0.0, "cube", confidence=0.6)
    c = detect.Target(0.150, 0.350, 0.030, 0.030, 0.0, "colour 2", confidence=0.9)
    kept = detect.distinct([b, a, c])
    assert [t.label for t in kept] == ["colour 2", "colour 1"] or \
        [t.label for t in kept] == ["colour 1", "colour 2"]
    assert all(t.label != "cube" for t in kept)


def test_everything_takes_position_from_the_tag_and_size_from_the_silhouette(monkeypatch):
    tag = detect.Target(0.301, 0.249, 0.026, 0.026, 5.0, "tag 7")
    blob = detect.Target(0.300, 0.250, 0.040, 0.040, 0.0, "colour 1", height_m=0.040)
    loose = detect.Target(0.150, 0.350, 0.030, 0.030, 0.0, "colour 2", height_m=0.030)
    monkeypatch.setattr(detect, "markers", lambda *a, **k: [tag])
    monkeypatch.setattr(detect, "coloured", lambda *a, **k: [blob, loose])
    frame = np.zeros((480, 640, 3), np.uint8)
    found = detect.everything(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M, url=None)
    by_label = {t.label: t for t in found}
    assert set(by_label) == {"tag 7", "colour 2"}
    fused = by_label["tag 7"]
    assert (fused.x, fused.y) == (0.301, 0.249)
    assert fused.width_m == 0.040 and fused.height_m == 0.040


def test_everything_survives_a_down_vision_service(monkeypatch):
    def down(*a, **k):
        raise detect.DetectorOffline("nobody home")
    monkeypatch.setattr(detect, "objects", down)
    notes = []
    frame = hull_scene([(0.300, 0.250, 0.040)])
    found = detect.everything(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M,
                              url="http://x", note=notes.append)
    assert len(found) == 1 and found[0].width_m == pytest.approx(0.040, abs=0.0015)
    assert any("vision service down" in n for n in notes)


def test_ladder_auto_needs_no_tag_no_background_and_no_vision(monkeypatch, tmp_path):
    monkeypatch.setattr(detect, "BACKGROUND_PATH", tmp_path / "table_background.png")
    frame = hull_scene([(0.300, 0.250, 0.040)])
    found = detect.ladder(frame, "auto", MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M, url=None)
    assert len(found) == 1
    assert found[0].width_m == pytest.approx(0.040, abs=0.0015)


def test_a_declared_size_is_also_the_cube_height():
    tag = detect.Target(0.301, 0.249, 0.026, 0.026, 5.0, "tag 7")
    declared = detect.declared_size([tag], 0.040)[0]
    assert declared.width_m == declared.height_m == 0.040


def test_an_outline_cut_by_the_frame_edge_is_flagged_not_mis_sized():
    """A 60 mm cube half out of shot used to range as a small cube that then
    passed the in-frame test. Now the contour touching the border is the test."""
    frame = hull_scene([(0.620, 0.250, 0.060)])   # hangs off the right edge
    found = detect.coloured(frame, MM_PER_PIXEL, nadir=NADIR, lens_m=LENS_M)
    assert len(found) == 1
    cut = found[0]
    assert cut.clipped and not cut.graspable and "out of frame" in cut.why_not()
    assert cut.confidence == 0.0
    whole = detect.coloured(hull_scene([(0.400, 0.250, 0.060)]), MM_PER_PIXEL,
                            nadir=NADIR, lens_m=LENS_M)[0]
    assert not whole.clipped and "wider" in whole.why_not()


def test_a_clipped_silhouette_lends_no_size_to_a_tag():
    tag = detect.Target(0.300, 0.250, 0.026, 0.026, 0.0, "tag 1")
    cut = detect.Target(0.302, 0.252, 0.024, 0.024, 0.0, "colour 1", 0.0, 0.024, clipped=True)
    fused = detect.fuse([tag], [cut])[0]
    assert fused.width_m == 0.026


def test_a_neural_box_on_the_edge_is_clipped(monkeypatch):
    reply = {"detections": [{"label": "cube", "confidence": 0.9, "box": [600.0, 200.0, 40.0, 40.0]}]}
    monkeypatch.setattr(detect, "_vision_post", lambda *a, **k: reply)
    frame = np.zeros((480, 640, 3), np.uint8)
    found = detect.objects(frame, MM_PER_PIXEL, url="http://x", nadir=NADIR, lens_m=LENS_M)
    assert found and found[0].clipped and not found[0].graspable


def test_a_tag_measures_the_objects_height_and_so_a_cubes_width():
    """A 26 mm tag imaged at 32 mm sits H * (1 - 26/32) up: 40 mm of a 212 mm lens."""
    frame = np.full((480, 640, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.OBJECT_TAG_DICT))
    tag = cv2.aruco.generateImageMarker(dictionary, 3, 32)
    frame[200:232, 300:332] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    tagged = detect.markers(frame, MM_PER_PIXEL, tag_m=0.026, nadir=NADIR, lens_m=0.212)
    assert len(tagged) == 1
    cube = tagged[0]
    assert cube.height_m == pytest.approx(0.212 * (1 - 26 / 32), abs=0.006)
    assert cube.width_m == pytest.approx(cube.height_m)
    flat = detect.markers(frame, MM_PER_PIXEL, tag_m=0.026, nadir=NADIR)[0]
    assert flat.height_m is None and flat.width_m == pytest.approx(0.026, abs=0.001)
