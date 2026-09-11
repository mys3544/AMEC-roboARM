"""Kinematics tests. Pure geometry -- no hardware, runs anywhere."""

import itertools
import math

import pytest

from roboarm import config as cfg
from roboarm import kinematics as kin

# Gripper CLOSED, because that is the state the 575 mm table-to-fingertip figure
# below was measured in, and the state the whole geometry fit was calibrated at.
# The gripper angle is load-bearing here: open, the tool is 28 mm shorter.
STRAIGHT_UP = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: cfg.GRIPPER_CLOSED}

# IK returns whole servo degrees. At ~350 mm reach one degree is about 6 mm, and
# three joints each rounding by up to half a degree can compound past a centimetre.
# This is the arm's resolution, not slack in the maths.
ROUNDING_LIMIT = 0.013


# ---------------------------------------------------------------- forward --
def test_straight_up_is_very_nearly_vertical():
    """All joints at servo 90 leaves the arm all but upright.

    Not EXACTLY upright: the joints are mounted a few degrees off (see
    JOINT_OFFSET_DEG), and J2's +7.5 and J3's -10.5 lean opposite ways so they
    almost cancel. That near-cancellation is why the parked pose measured perfect
    while bent poses drifted by up to 20 mm.
    """
    x, y, z = kin.forward(STRAIGHT_UP)
    assert abs(x) < 0.010, "within a centimetre of the axis"
    assert abs(y) < 1e-9, "no sideways component at J1=90"
    # Slightly less than the full stacked length, because the joint offsets leave
    # the arm a few degrees off vertical rather than perfectly straight.
    stacked = cfg.BASE_PLATE_TO_J2 + kin.L1 + kin.L2 + kin.L3
    assert z == pytest.approx(stacked, abs=0.003)
    # Checked against the real arm: parked upright with the gripper CLOSED, the
    # table-to-fingertip distance measured 575 mm. The model puts it at
    # 190 + 30 + 83.5 + 83.5 + 80 + 110 = 577, so the whole chain is right to 2 mm.
    above_table = z + cfg.TABLE_BELOW_PLATE
    assert above_table == pytest.approx(0.575, abs=0.006)


def test_reach_grows_smoothly_as_the_shoulder_tips_forward():
    """Sanity on the shape of the workspace, not a fit to measured data.

    (The old version of this test compared against figures the forward-sweep tool
    printed -- but those were the model's own predictions, never measured, so it
    was checking the model against itself.)
    """
    reaches = [kin.forward({**STRAIGHT_UP, 2: j2})[0] for j2 in (90, 75, 60, 45, 30, 15)]
    assert reaches[0] == pytest.approx(0.0, abs=0.010)  # near, not exactly, the axis
    assert all(b > a for a, b in itertools.pairwise(reaches)), "reach must increase"
    # Slightly under the 344 mm measured in the forward sweep: that sweep ran
    # before calibration, and the offsets shorten the effective reach a little.
    assert reaches[-1] == pytest.approx(0.340, abs=0.006)


def test_rising_j2_reaches_backward_towards_the_mast():
    forward_x = kin.forward({**STRAIGHT_UP, 2: 60})[0]
    backward_x = kin.forward({**STRAIGHT_UP, 2: 105})[0]
    assert forward_x > 0, "J2 below 90 must reach forward"
    assert backward_x < 0, "J2 above 90 must reach backward, towards the mast"


def test_rising_j1_swings_right():
    # +y is left, so swinging right means y goes negative.
    assert kin.forward({**STRAIGHT_UP, 2: 45, 1: 120})[1] < 0


def test_tool_pitch_is_measured_from_vertical():
    # At all-90 the tool is within a few degrees of vertical -- not exactly zero,
    # because the joint offsets do not perfectly cancel.
    assert abs(kin.tool_pitch(STRAIGHT_UP)) < 6.0
    # Tipping J2 forward by 90 degrees must swing the tool by the same 90.
    swung = kin.tool_pitch({**STRAIGHT_UP, 2: 0}) - kin.tool_pitch(STRAIGHT_UP)
    assert swung == pytest.approx(90.0, abs=0.1)


# ---------------------------------------------------------------- inverse --
LEGAL_POSES = [
    {**STRAIGHT_UP, 2: j2, 3: j3, 4: j4}
    for j2, j3, j4 in [
        (45, 90, 90), (30, 90, 90), (60, 90, 90), (50, 100, 80),
        (40, 105, 95), (70, 100, 70), (55, 95, 100), (35, 95, 85),
    ]
]


@pytest.mark.parametrize("pose", LEGAL_POSES, ids=lambda p: f"J2={p[2]},J3={p[3]},J4={p[4]}")
def test_ik_round_trips_through_fk(pose):
    """Every target comes from FK, so it is reachable by construction.

    IK may legitimately return a DIFFERENT pose than the one we started from -- the
    same fingertip position and pitch can often be produced two ways -- so the check
    is on where the fingertip ends up, not on the joint angles.
    """
    target = kin.forward(pose)
    solved = kin.inverse(*target, pitch_deg=kin.tool_pitch(pose), gripper=pose[6])
    # Servo angles are whole degrees; a degree at the shoulder is ~6 mm at full
    # extension, so this is the achievable precision, not slop.
    assert math.dist(kin.forward(solved), target) < ROUNDING_LIMIT


def test_ik_round_trips_off_axis():
    pose = {**STRAIGHT_UP, 1: 115, 2: 45, 3: 95, 4: 85}
    target = kin.forward(pose)
    assert target[1] < 0, "J1 above 90 should put the arm to the right"
    solved = kin.inverse(*target, pitch_deg=kin.tool_pitch(pose), gripper=pose[6])
    assert math.dist(kin.forward(solved), target) < ROUNDING_LIMIT


def test_ik_refuses_a_point_beyond_reach():
    with pytest.raises(kin.Unreachable, match="mm from the shoulder"):
        kin.inverse(0.60, 0.0, 0.20)


def test_ik_refuses_a_pose_that_would_hit_the_mast():
    # Behind the arm, where the mast is.
    with pytest.raises(kin.Unreachable):
        kin.inverse(-0.20, 0.0, 0.25, pitch_deg=-60.0)


def test_ik_output_is_directly_commandable():
    """Every joint IK returns must already satisfy arm.py's own limits."""
    reference = {**STRAIGHT_UP, 2: 45}
    pose = kin.inverse(*kin.forward(reference), pitch_deg=kin.tool_pitch(reference))
    assert set(pose) == set(cfg.JOINT_IDS)
    for joint in (1, 2, 3, 4):
        lo, hi = cfg.SAFE_LIMITS[joint]
        assert lo <= pose[joint] <= hi
    assert cfg.mast_clearance(pose) >= cfg.MIN_MAST_CLEARANCE_M


def test_both_elbow_solutions_reach_the_same_point():
    reference = {**STRAIGHT_UP, 2: 50, 3: 100, 4: 80}
    target, pitch = kin.forward(reference), kin.tool_pitch(reference)
    solved = []
    for elbow_up in (True, False):
        try:
            solved.append(
                kin.inverse(*target, pitch_deg=pitch, elbow_up=elbow_up, gripper=reference[6])
            )
        except kin.Unreachable:
            continue
    assert solved, "at least one elbow branch must solve a point FK produced"
    for pose in solved:
        assert math.dist(kin.forward(pose), target) < ROUNDING_LIMIT


def test_reachable_agrees_with_inverse():
    reference = {**STRAIGHT_UP, 2: 45}
    assert kin.reachable(*kin.forward(reference), pitch_deg=kin.tool_pitch(reference))
    assert not kin.reachable(0.60, 0.0, 0.20)


# ------------------------------------------------------------------ solve --
def test_solve_finds_a_pitch_across_the_working_strip():
    z = -cfg.TABLE_BELOW_PLATE + 0.030
    for reach in (0.14, 0.17, 0.20, 0.23):
        pose, pitch = kin.solve(reach, 0.0, z)
        assert math.dist(kin.forward(pose), (reach, 0.0, z)) < ROUNDING_LIMIT
        assert 140.0 <= pitch <= 180.0


def test_solve_uses_a_steeper_pitch_for_nearer_targets():
    z = -cfg.TABLE_BELOW_PLATE + 0.030
    _, near = kin.solve(0.14, 0.0, z)
    _, far = kin.solve(0.23, 0.0, z)
    assert near > far, "near targets should be taken more vertically"


def test_solve_refuses_something_genuinely_out_of_reach():
    with pytest.raises(kin.Unreachable, match="not reachable at any pitch"):
        kin.solve(0.50, 0.0, -0.150)


def test_nothing_may_be_commanded_below_the_table():
    with pytest.raises(kin.Unreachable, match="below the table"):
        kin.inverse(0.15, 0.0, -cfg.TABLE_BELOW_PLATE - 0.010, 180.0)


# ------------------------------------------------------------ tool length --
def test_the_tool_gets_shorter_as_the_gripper_opens():
    """The measured table, straight out of config. Opening splays the fingers back."""
    assert cfg.tool_length(150) > cfg.tool_length(90) > cfg.tool_length(30)
    assert cfg.tool_length(30) == pytest.approx(0.162)
    assert cfg.tool_length(150) == pytest.approx(0.190)


def test_tool_length_is_flat_outside_the_measured_range():
    """Three points do not justify extrapolating a curve. Flat ends are honest."""
    assert cfg.tool_length(10) == cfg.tool_length(30)
    assert cfg.tool_length(cfg.GRIPPER_CLOSED) == cfg.tool_length(150)


def test_tool_length_interpolates_between_measured_points():
    middle = cfg.tool_length(120)
    assert cfg.tool_length(90) < middle < cfg.tool_length(150)
    assert middle == pytest.approx(0.185)


def test_opening_the_gripper_raises_the_fingertips():
    """The failure this whole model exists to prevent.

    Same joint angles, gripper opened: the fingertips end up 28 mm HIGHER, because
    the tool is 28 mm shorter. Solve for a grasp with the closed length and approach
    with the gripper open, and you close on empty air above the object.
    """
    down = {1: 90, 2: 26, 3: 31, 4: 29, 5: 90, 6: cfg.GRIPPER_CLOSED}
    closed_z = kin.forward(down)[2]
    open_z = kin.forward({**down, 6: cfg.GRIPPER_OPEN})[2]
    assert (open_z - closed_z) == pytest.approx(0.028, abs=0.001)


def test_ik_puts_the_fingertips_at_the_target_whatever_the_gripper_does():
    """The fix: tell solve() the opening and it compensates.

    Via solve() rather than inverse(), because a fixed pitch is not available to
    every opening -- an open gripper is short enough that a straight-down approach
    at 150 mm would need J2=11, below its safe limit. Letting the pitch follow is
    exactly what the grasp code does.
    """
    target = (0.150, 0.0, -cfg.TABLE_BELOW_PLATE + 0.020)
    for gripper in (cfg.GRIPPER_OPEN, 90, cfg.GRIPPER_CLOSED):
        pose, _pitch = kin.solve(*target, gripper=gripper)
        assert pose[6] == gripper, "must report the opening it solved for"
        assert math.dist(kin.forward(pose), target) < ROUNDING_LIMIT


def test_opening_the_gripper_costs_reach():
    """A shorter tool needs a lower shoulder, and the shoulder runs out first.

    Worth a test because it is counter-intuitive and it shapes the demo: objects
    have to be placed where the arm can still get to them with the fingers OPEN.
    """
    z = -cfg.TABLE_BELOW_PLATE + 0.020

    def furthest(gripper):
        return max(mm for mm in range(100, 300, 5)
                   if _solvable(mm / 1000, z, gripper))

    assert furthest(cfg.GRIPPER_OPEN) < furthest(cfg.GRIPPER_CLOSED)


def _solvable(x, z, gripper):
    try:
        kin.solve(x, 0.0, z, gripper=gripper)
    except kin.Unreachable:
        return False
    return True


def test_gripper_for_gap_picks_the_narrowest_opening_that_fits():
    """Widest servo angle == narrowest fingers, since a rising angle closes them."""
    assert cfg.GRIPPER_GAP_MM[cfg.gripper_for_gap(0.040)] >= 40
    assert cfg.GRIPPER_GAP_MM[cfg.gripper_for_gap(0.020)] >= 20
    # Not simply flung fully open: a 20 mm object does not need a 70 mm gap.
    assert cfg.gripper_for_gap(0.020) > cfg.GRIPPER_OPEN
    with pytest.raises(ValueError, match="nothing opens to"):
        cfg.gripper_for_gap(0.200)


def test_gripper_gap_answers_at_angles_nobody_measured():
    """GRIPPER_CLOSED is 170 and the table jumps 150 -> 177, so a direct lookup
    raises KeyError. tools/camera_offset.py crashed on exactly that."""
    assert cfg.GRIPPER_CLOSED not in cfg.GRIPPER_GAP_MM
    gap = cfg.gripper_gap(cfg.GRIPPER_CLOSED)
    assert 0.013 < gap < 0.025, "between the 177 and 150 entries"
    # Ends stay flat rather than extrapolating off the measured range.
    assert cfg.gripper_gap(0) == cfg.gripper_gap(30)
    assert cfg.gripper_gap(200) == cfg.gripper_gap(177)


def test_gripper_gap_and_gripper_for_gap_agree():
    """Round-trip: an angle wide enough for a gap must really be that wide."""
    for want in (0.020, 0.030, 0.045, 0.060):
        assert cfg.gripper_gap(cfg.gripper_for_gap(want)) >= want


# ------------------------------------------------------------ wrist lens --
# The pose the table homography was fitted at, from data/table_homography.json.
SURVEY_POSE = {1: 90, 2: 56, 3: 23, 4: 10, 5: 89, 6: 30}


def test_lens_sits_where_the_ruler_said():
    """MEASURED: 125 mm from the fingertips closed, 96 mm open, i.e. 65 mm from J4.

    Checked through forward(), so it also catches anyone re-swapping the two numbers:
    with the old value the lens computed 60 mm further out and every parallax
    correction was wrong.
    """
    for gripper in (cfg.GRIPPER_CLOSED, cfg.GRIPPER_OPEN):
        pose = {**STRAIGHT_UP, 6: gripper}
        lens_to_tip = cfg.tool_length(gripper) - cfg.CAMERA_FROM_J4
        expected = {cfg.GRIPPER_CLOSED: 0.125, cfg.GRIPPER_OPEN: 0.096}[gripper]
        assert lens_to_tip == pytest.approx(expected, abs=0.002)
        assert kin.forward(pose)[2] > 0


def test_lens_height_agrees_with_the_tag_magnification():
    """Independent cross-check. A 26 mm tag on a 40 mm cube came out 32 mm through
    the homography -- magnified 1.22 -- which puts the lens at h*m/(m-1) = 222 mm.
    The kinematics must land near that; with the swapped constant it gave 153 mm."""
    assert kin.camera_height(SURVEY_POSE) == pytest.approx(0.213, abs=0.010)


def test_the_lens_is_not_above_the_middle_of_the_picture():
    """The assumption that cost us a 20 mm reach error.

    The image centre maps to about (164, +19) mm; the lens is 50 mm off the forearm
    axis, so it is nowhere near there. If these two are ever within a few mm of each
    other, something has quietly reverted.
    """
    nadir = kin.camera_nadir(SURVEY_POSE)
    assert math.dist(nadir, (0.164, 0.019)) > 0.040


def test_the_lens_offset_follows_the_arm_around():
    """Off-axis means off-axis in the TABLE frame too, so it swings with J1."""
    straight = kin.camera_nadir(SURVEY_POSE)
    swung = kin.camera_nadir({**SURVEY_POSE, 1: 60})
    assert swung != straight
    # Same distance from the base whichever way the arm points.
    assert math.hypot(*swung) == pytest.approx(math.hypot(*straight), abs=1e-9)


# ----------------------------------------------------------- wrist roll ----
def test_fingers_close_along_the_arm_at_the_measured_roll():
    straight = {1: 90, 5: cfg.J5_FINGERS_ALONG_ARM}
    assert kin.finger_heading(straight) == pytest.approx(0.0)
    across = {1: 90, 5: cfg.J5_FINGERS_ALONG_ARM - 90}
    assert abs(kin.finger_heading(across)) == pytest.approx(90.0)


def test_rising_j5_turns_the_closing_line_clockwise():
    """Measured: a finger seen at the LEFT of the wrist picture at J5=110 had
    crossed to the RIGHT by J5=210."""
    left = kin.finger_heading({1: 90, 5: cfg.J5_FINGERS_ALONG_ARM - 40})
    right = kin.finger_heading({1: 90, 5: cfg.J5_FINGERS_ALONG_ARM + 40})
    assert left > 0 > right


def test_roll_for_undoes_the_base_yaw():
    """A cube square to the table, off to the right: the base yaws 30 deg right to
    reach it, so the wrist must turn 30 deg back to close square on its faces."""
    j1 = 90 + 30  # kinematics yaw = 90 - J1, so this is 30 deg to the right
    j5 = kin.roll_for(0.0, j1, square=True)
    assert kin.finger_heading({1: j1, 5: j5}) % 90 == pytest.approx(0.0, abs=0.6)
    assert abs(j5 - cfg.J5_FINGERS_ACROSS_ARM) == 30


def test_a_cube_square_in_front_needs_no_turn_at_all():
    """Faces at 0 and the arm straight ahead: rest (89) closes across it already,
    and the wrist must NOT swing 90 degrees to the equally good along-arm position."""
    assert kin.roll_for(0.0, 90, square=True) == cfg.J5_FINGERS_ACROSS_ARM
    assert kin.roll_for(90.0, 90, square=True) == cfg.J5_FINGERS_ACROSS_ARM
    for closing in range(-90, 91, 5):
        assert abs(kin.roll_for(float(closing), 90, square=True)
                   - cfg.J5_FINGERS_ACROSS_ARM) <= 45


def test_roll_for_matches_the_objects_own_angle():
    for closing in (-80.0, -30.0, 0.0, 25.0, 60.0, 89.0):
        j5 = kin.roll_for(closing, 90)
        assert kin.finger_heading({1: 90, 5: j5}) == pytest.approx(closing, abs=0.6)
        assert cfg.SAFE_LIMITS[5][0] <= j5 <= cfg.SAFE_LIMITS[5][1]


def test_a_square_takes_the_smaller_turn():
    assert abs(kin.roll_for(10.0, 90, square=True) - cfg.J5_FINGERS_ACROSS_ARM) == 10
    assert abs(kin.roll_for(10.0, 90, square=False) - cfg.J5_FINGERS_ACROSS_ARM) == 80


def test_roll_for_never_leaves_the_safe_range():
    for j1 in range(cfg.SAFE_LIMITS[1][0], cfg.SAFE_LIMITS[1][1] + 1, 5):
        for closing in range(-90, 91, 5):
            j5 = kin.roll_for(float(closing), j1)
            assert cfg.SAFE_LIMITS[5][0] <= j5 <= cfg.SAFE_LIMITS[5][1]
            assert kin.finger_heading({1: j1, 5: j5}) == pytest.approx(
                (closing + 90) % 180 - 90, abs=0.6) or abs(closing) == 90


def test_the_lens_does_not_move_when_the_gripper_does():
    """It is bolted to the forearm; the fingers are not. This is what makes the
    nadir usable at all -- it needs no gripper state."""
    opened = kin.camera_nadir({**SURVEY_POSE, 6: cfg.GRIPPER_OPEN})
    closed = kin.camera_nadir({**SURVEY_POSE, 6: cfg.GRIPPER_CLOSED})
    assert opened == closed
