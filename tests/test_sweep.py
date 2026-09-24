"""Sweep tests. Pure geometry against the REAL calibration -- no hardware.

The homography below is the one actually fitted on the robot on 2026-09-08 (the
fit that was then validated end to end to 2 mm), copied out of
data/table_homography.json. Using the real matrix rather than a synthetic one is
the point: the coverage numbers these tests assert are the coverage numbers the
robot actually gets, so if a future recalibration moves the camera enough to open
a hole in the ring, this suite says so instead of passing on a made-up matrix.
"""

import math

import numpy as np
import pytest

from roboarm import config as cfg
from roboarm import detect, sweep
from roboarm import kinematics as kin

# data/table_homography.json, fitted 2026-09-08, worst residual 0.48 mm over 8 points.
REAL_H = np.array([
    [1.5806220846933396e-05, -2.412854587985502e-04, 2.1607999921844648e-01],
    [-2.2744926875526548e-04, -3.5041628047172227e-07, 9.179117812253038e-02],
    [5.9373685046794684e-05, -8.77513698668719e-05, 1.0],
])
REAL_SURVEY = {1: 90, 2: 56, 3: 23, 4: 10, 5: 89, 6: 30}

# Enumerating the reachable set asks the IK about 20000 points, which is a second
# or so. Every coverage test wants the same answer, so it is computed once.
GRASPABLE = sweep.reachable_grasp_points()


# A second, further-reaching look for the hint tests: the real matrix with its
# table output scaled 1.25x outward from the base, and a different survey pose.
OUTER_H = np.diag([1.25, 1.25, 1.0]) @ REAL_H
OUTER_SURVEY = {1: 90, 2: 30, 3: 51, 4: 12, 5: 90, 6: 30}
# And a nearer-seeing one: the same picture landing 50 mm closer to the base.
NEAR_H = np.array([[1.0, 0.0, -0.05], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ REAL_H
NEAR_SURVEY = {1: 90, 2: 75, 3: 11, 4: 0, 5: 90, 6: 30}


def _clipped(x, y, edges, label="blob"):
    return detect.Target(x=x, y=y, width_m=0.02, length_m=0.02, angle_deg=0.0,
                         label=label, confidence=0.0, clipped=True,
                         edges=frozenset(edges))


def _station(dyaw=0.0):
    return sweep.Look(dyaw, sweep.pose_at(REAL_SURVEY, dyaw), sweep.rotate(REAL_H, dyaw))


# ---------------------------------------------------------------- hints ----
def test_frame_sides_names_the_far_edge_and_the_left_edge():
    sides = sweep.frame_sides(REAL_H)
    assert {sides["far"], sides["near"]} == {"top", "bottom"}
    assert {sides["left"], sides["right"]} == {"left", "right"}
    assert sides["far"] == "top"        # the primary look: far is the top row
    assert sides["left"] == "left"      # and the picture is not mirrored
    assert sweep.reach_of(OUTER_H) > sweep.reach_of(REAL_H)


def test_a_far_edge_clip_hints_at_the_further_look_on_that_bearing():
    look = _station(0.0)
    target = _clipped(0.19, 0.06, {"top"})
    got = sweep.hints(look, [target], [(REAL_H, REAL_SURVEY, "primary"),
                                       (OUTER_H, OUTER_SURVEY, "outer")])
    assert len(got) == 1
    hint = got[0]
    assert hint.look.name == "outer"
    assert all(hint.look.pose[j] == OUTER_SURVEY[j] for j in (2, 3, 4, 5, 6))
    wanted = sweep.yaw_to_centre(OUTER_H, target.x, target.y)
    assert hint.look.dyaw == pytest.approx(wanted, abs=0.5)
    assert "far edge" in hint.why and "outer" in hint.why


def test_a_far_edge_clip_with_no_further_look_is_no_hint():
    got = sweep.hints(_station(0.0), [_clipped(0.19, 0.0, {"top"})],
                      [(REAL_H, REAL_SURVEY, "primary")])
    assert got == []


def test_a_side_clip_turns_the_same_look_to_centre_it():
    look = _station(0.0)
    target = _clipped(0.16, 0.07, {"left"})
    got = sweep.hints(look, [target], [(REAL_H, REAL_SURVEY, "primary")])
    assert len(got) == 1
    assert got[0].look.name == "primary"
    assert got[0].look.dyaw == pytest.approx(
        sweep.yaw_to_centre(REAL_H, target.x, target.y), abs=0.5)
    assert got[0].look.dyaw > 0          # left of the picture is +yaw
    assert "left edge" in got[0].why


def test_a_near_edge_clip_or_an_unclipped_target_gives_nothing():
    look = _station(0.0)
    whole = detect.Target(x=0.16, y=0.0, width_m=0.04, length_m=0.04,
                          angle_deg=0.0, label="cube")
    got = sweep.hints(look, [_clipped(0.13, 0.0, {"bottom"}), whole],
                      [(REAL_H, REAL_SURVEY, "primary")])
    assert got == []


def test_a_near_edge_clip_hints_at_the_nearer_look_when_there_is_one():
    """The white cube 11 cm straight ahead (2026-09-22): cut off at the primary's
    near edge at every station, graspable since the J3 floor went to 0, and
    listed by nobody. With a near look calibrated, that clip points at it."""
    assert sweep.nearest_of(NEAR_H) < sweep.nearest_of(REAL_H) - 0.04
    look = _station(0.0)
    target = _clipped(0.125, 0.01, {"bottom"})
    got = sweep.hints(look, [target], [(REAL_H, REAL_SURVEY, "primary"),
                                       (OUTER_H, OUTER_SURVEY, "outer"),
                                       (NEAR_H, NEAR_SURVEY, "near")])
    assert len(got) == 1 and got[0].look.name == "near"
    assert all(got[0].look.pose[j] == NEAR_SURVEY[j] for j in (2, 3, 4, 5, 6))
    assert "near edge" in got[0].why and "sees nearer" in got[0].why


def test_hints_never_point_back_at_the_station_that_saw_them():
    look = _station(0.0)
    # Something cut off at the left edge but whose bearing already runs through
    # the centre: turning would change nothing, so no hint.
    bearing = math.radians(sweep.look_bearing(REAL_H))
    target = _clipped(0.16 * math.cos(bearing), 0.16 * math.sin(bearing), {"left"})
    assert sweep.hints(look, [target], [(REAL_H, REAL_SURVEY, "primary")]) == []


def test_a_far_clip_from_the_furthest_look_is_reported_beyond_reach():
    both = [(REAL_H, REAL_SURVEY, "primary"), (OUTER_H, OUTER_SURVEY, "outer")]
    outer = sweep.Look(0.0, dict(OUTER_SURVEY), OUTER_H, "outer")
    lost = _clipped(0.23, 0.0, {"top"}, "far thing")
    assert sweep.beyond_reach(outer, [lost, _clipped(0.2, 0.05, {"left"})], both) == [lost]
    # From the primary the outer look still reaches further, so nothing is lost yet.
    assert sweep.beyond_reach(_station(0.0), [lost], both) == []
    # ... unless the primary is the only look there is.
    assert sweep.beyond_reach(_station(0.0), [lost], both[:1]) == [lost]


def test_two_clips_of_one_object_give_one_hint():
    look = _station(0.0)
    both = [_clipped(0.19, 0.06, {"top", "left"}, "a"), _clipped(0.19, 0.061, {"top"}, "b")]
    got = sweep.hints(look, both, [(REAL_H, REAL_SURVEY, "primary"),
                                   (OUTER_H, OUTER_SURVEY, "outer")])
    assert len(got) == 1


def test_a_pooled_fit_ignores_one_bad_corner():
    """12+ points: a corner 15 mm off is left out of the fit, not averaged in."""
    from roboarm import workspace as ws
    rng = np.random.default_rng(1)
    pixels = rng.uniform([20, 20], [620, 460], size=(16, 2))
    table = ws.apply(REAL_H, pixels)
    table[3] += (0.015, 0.0)
    matrix, _worst, n = ws.fit_points(pixels, table)
    assert n == 16
    good = np.delete(np.arange(16), 3)
    err = np.linalg.norm(ws.apply(matrix, pixels[good]) - table[good], axis=1)
    assert err.max() < 0.001, "the 15 good corners still fit to under a millimetre"


def _rotate2(point, degrees):
    """An independent rotation, written out longhand to check `rotate` against."""
    turn = math.radians(degrees)
    x, y = point
    return (x * math.cos(turn) - y * math.sin(turn),
            x * math.sin(turn) + y * math.cos(turn))


# --------------------------------------------------------------- rotate ----
def test_rotate_by_nothing_changes_nothing():
    assert sweep.rotate(REAL_H, 0.0) == pytest.approx(REAL_H)


def test_rotate_turns_the_mapped_point_about_the_base():
    """The rotated homography must agree with rotating its answer by hand."""
    from roboarm import workspace as ws
    for pixel in [(10, 10), (320, 240), (630, 470), (100, 400)]:
        plain = ws.apply(REAL_H, [pixel])[0]
        for dyaw in (-75, -25, 0, 25, 75):
            spun = ws.apply(sweep.rotate(REAL_H, dyaw), [pixel])[0]
            assert spun == pytest.approx(_rotate2(plain, dyaw), abs=1e-12)


def test_rotate_preserves_distance_from_the_base():
    """A yaw cannot change how far away something is -- only its bearing."""
    from roboarm import workspace as ws
    pixel = [(320, 240)]
    reference = np.linalg.norm(ws.apply(REAL_H, pixel)[0])
    for dyaw in range(-80, 81, 10):
        moved = np.linalg.norm(ws.apply(sweep.rotate(REAL_H, dyaw), pixel)[0])
        assert moved == pytest.approx(reference, abs=1e-12)


def test_rotations_compose():
    assert sweep.rotate(sweep.rotate(REAL_H, 20), 15) == pytest.approx(
        sweep.rotate(REAL_H, 35))


# --------------------------------------------------------------- pose_at ----
def test_pose_at_agrees_with_the_kinematics_own_yaw():
    """THE sign test.

    `rotate` turns the map one way and `pose_at` turns the arm the other; if the
    two conventions disagree the whole module is a mirror image of itself and every
    pick off-centre goes to the wrong side of the table. kinematics.camera_nadir()
    is an independent witness -- it derives the lens position from the joint angles
    without going anywhere near a homography -- so the two must land together.
    """
    reference = kin.camera_nadir(REAL_SURVEY)
    for dyaw in (-80, -55, -25, -5, 5, 25, 55, 80):
        moved = kin.camera_nadir(sweep.pose_at(REAL_SURVEY, dyaw))
        assert moved == pytest.approx(_rotate2(reference, dyaw), abs=1e-9)


def test_pose_at_moves_only_the_base():
    pose = sweep.pose_at(REAL_SURVEY, 40)
    assert pose[1] == 50, "yawing 40 degrees LEFT takes J1 down by 40"
    for joint in (2, 3, 4, 5, 6):
        assert pose[joint] == REAL_SURVEY[joint], f"J{joint} must not move"


def test_pose_at_refuses_to_leave_the_safe_range():
    low, high = cfg.SAFE_LIMITS[1]
    with pytest.raises(sweep.NoLook):
        sweep.pose_at(REAL_SURVEY, REAL_SURVEY[1] - low + 1)
    with pytest.raises(sweep.NoLook):
        sweep.pose_at(REAL_SURVEY, REAL_SURVEY[1] - high - 1)


def test_positive_yaw_goes_left():
    """Sanity in the direction a human would check it: +y is left."""
    _x, y = kin.camera_nadir(sweep.pose_at(REAL_SURVEY, 60))
    _x0, y0 = kin.camera_nadir(REAL_SURVEY)
    assert y > y0


# ------------------------------------------------------------------ ring ----
def test_ring_stations_are_all_legal_and_share_every_other_joint():
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    assert len(looks) >= 5
    low, high = cfg.SAFE_LIMITS[1]
    for look in looks:
        assert low <= look.pose[1] <= high
        for joint in (2, 3, 4, 5, 6):
            assert look.pose[joint] == REAL_SURVEY[joint]


def test_ring_includes_the_calibrated_look_itself():
    """The one look whose accuracy was actually measured must be in the set."""
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    assert any(look.dyaw == 0 for look in looks)
    home = next(look for look in looks if look.dyaw == 0)
    assert home.pose == REAL_SURVEY
    assert home.matrix == pytest.approx(REAL_H)


def test_every_ring_station_clears_the_mast():
    """Inherited from the calibrated pose, but assert it rather than assume it."""
    for look in sweep.ring(REAL_SURVEY, REAL_H):
        assert cfg.mast_clearance(look.pose) >= cfg.MIN_MAST_CLEARANCE_M


def test_the_fingers_sweep_well_above_the_table():
    """The base may turn through the ring without raking objects off the table."""
    for look in sweep.ring(REAL_SURVEY, REAL_H):
        _x, _y, z = kin.forward(look.pose)
        above_table = z + cfg.TABLE_BELOW_PLATE
        assert above_table > 0.075, "must clear a 40 mm cube with room to spare"


def test_ring_rejects_a_nonsense_step():
    for step in (0, -5):
        with pytest.raises(ValueError):
            sweep.ring(REAL_SURVEY, REAL_H, step_deg=step)


# -------------------------------------------------------------- coverage ----
def test_one_look_alone_covers_only_a_corner_of_the_workspace():
    """The problem this module exists to solve, pinned as a number."""
    only = [sweep.Look(0.0, REAL_SURVEY, REAL_H)]
    fraction, _missed = sweep.coverage(only, GRASPABLE)
    # 5.9 % since J2 may go below zero (2026-09-23): the envelope grew to 262 mm
    assert 0.05 < fraction < 0.17, f"one pose can measure {fraction:.1%}"


def test_the_ring_multiplies_what_one_look_can_measure():
    one, _ = sweep.coverage([sweep.Look(0.0, REAL_SURVEY, REAL_H)], GRASPABLE)
    many, _ = sweep.coverage(sweep.ring(REAL_SURVEY, REAL_H), GRASPABLE)
    assert many > 4 * one, f"{one:.1%} -> {many:.1%} is not worth the sweep"
    # 40 % since the back-tilted grasp pitches (nearest point 90 mm, a band the
    # primary ring cannot see and the near look covers); 46 % before them.
    # 30.5 % since the J2 floor went to -14: a 238..262 mm rim no look sees yet.
    assert many > 0.29


def test_the_ring_fixes_bearing_and_leaves_radius_alone():
    """The honest shape of what a yaw ring can and cannot do.

    Turning the base sweeps the camera round in BEARING, so every heading the arm
    can reach gets looked at. It cannot change how FAR the camera looks, because a
    yaw maps the table to itself and a tilt does not. So the blind spot must be an
    outer RIM -- never a missing wedge, which would mean the ring had a hole in it.
    """
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    _fraction, missed = sweep.coverage(looks, GRASPABLE)
    assert len(missed), "if this ever covers everything, tighten the claim"
    # The sweep spans cfg.SWEEP_SPAN_DEG (160): the wedge past +-80 deg is
    # graspable (J1 goes to 180) but never looked at, by choice (2026-09-22).
    bearings = np.degrees(np.arctan2(missed[:, 1], missed[:, 0]))
    missed = missed[np.abs(bearings) <= sweep.cfg.SWEEP_SPAN_DEG / 2 - 2]
    radii = np.hypot(missed[:, 0], missed[:, 1])
    # Two rims, never a wedge. The J3 floor of 0 (2026-09-22) added a band at
    # 122..129 mm the arm can grasp but no station sees: the primary look's
    # near edge is 129 mm, and nothing at survey height looks nearer.
    inner, outer = radii[radii < 0.150], radii[radii >= 0.150]
    assert len(outer) and outer.min() > 0.180, "the far blind spot must be the outer rim"
    assert (inner < 0.135).all(), "the near blind spot must hug the primary near edge"

    # And inside the band it does cover, EVERY bearing must work -- that is the
    # whole point of the ring, and the thing the old single look could not do.
    for bearing in range(-78, 79, 6):
        for radius in (0.135, 0.155, 0.175):
            x = radius * math.cos(math.radians(bearing))
            y = radius * math.sin(math.radians(bearing))
            assert any(sweep.sees(look, x, y) for look in looks), (
                f"nothing looks at {radius * 1000:.0f} mm, {bearing} deg")


def test_a_finer_ring_only_ever_helps():
    coarse, _ = sweep.coverage(sweep.ring(REAL_SURVEY, REAL_H, 25), GRASPABLE)
    fine, _ = sweep.coverage(sweep.ring(REAL_SURVEY, REAL_H, 15), GRASPABLE)
    assert fine >= coarse


# ------------------------------------------------ standing above the table ----
def test_magnification_matches_the_cube_measured_on_the_robot():
    """212 mm lens, 40 mm cube -> 1.23. The real tag measured 1.22 on 2026-09-10."""
    grown = sweep.magnification(REAL_SURVEY, sweep.OBJECT_HEIGHT_M)
    assert grown == pytest.approx(1.23, abs=0.02)


def test_nothing_on_the_table_is_magnified():
    assert sweep.magnification(REAL_SURVEY, 0.0) == 1.0


def test_an_object_taller_than_the_lens_is_refused():
    with pytest.raises(ValueError):
        sweep.magnification(REAL_SURVEY, 0.5)


def test_height_pushes_an_object_away_from_the_nadir():
    """The direction matters: parallax throws things OUTWARD from under the lens."""
    look = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    nadir = np.array(look.nadir)
    point = np.array([0.200, 0.060])
    shown = np.array(sweep.apparent(look, *point, sweep.OBJECT_HEIGHT_M))
    assert np.linalg.norm(shown - nadir) > np.linalg.norm(point - nadir)
    # and straight out along the same line, not off at an angle
    along, out = point - nadir, shown - nadir
    assert along[0] * out[1] - along[1] * out[0] == pytest.approx(0.0, abs=1e-12)


def test_a_point_at_the_nadir_does_not_move():
    look = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    shown = sweep.apparent(look, *look.nadir, sweep.OBJECT_HEIGHT_M)
    assert shown == pytest.approx(look.nadir, abs=1e-12)


def test_apparent_is_the_inverse_of_the_detector_unlift():
    """sweep predicts where a raised object appears; detect._unlift undoes it.

    They are the same relation read in opposite directions, so a round trip has to
    come back to where it started -- if these two ever drift apart, every tagged
    pick acquires a silent offset.
    """
    look = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    nadir = np.array(look.nadir)
    truth = np.array([0.175, 0.030])
    grown = sweep.magnification(REAL_SURVEY, sweep.OBJECT_HEIGHT_M)
    shown = np.array(sweep.apparent(look, *truth, sweep.OBJECT_HEIGHT_M))
    # _unlift measures the magnification from the tag's apparent size; feed it a
    # quad of exactly that size so the two describe the same situation.
    quad = shown + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (
        0.5 * sweep.TAG_SPAN_M * grown)
    back = detect._unlift(quad, nadir, sweep.TAG_SPAN_M)
    assert back.mean(axis=0) == pytest.approx(truth, abs=1e-9)


def test_pixels_per_metre_is_about_four_and_a_half_per_mm():
    for x, y in [(0.130, 0.0), (0.164, 0.019), (0.200, 0.050)]:
        scale = sweep.pixels_per_metre(REAL_H, x, y)
        assert 3500 < scale < 5500, f"{scale:.0f} px/m at {x}, {y}"


# ------------------------------------------------------------------ sees ----
def test_sees_reproduces_what_the_robot_actually_did():
    """The cube sat at (165, +5) mm on 2026-09-10.

    Driving the ring, the tag decoded ONLY from the calibrated station: at -25 deg
    the cube was clipped by the left edge of the frame, exactly as the photograph
    in docs shows. If this model ever stops agreeing with that run, it has stopped
    describing this camera.
    """
    x, y = 0.165, 0.005
    decoded = {look.dyaw: sweep.sees(look, x, y, sweep.TAG_SPAN_M)
               for look in sweep.ring(REAL_SURVEY, REAL_H)}
    assert decoded[0.0] is True
    assert not any(seen for dyaw, seen in decoded.items() if dyaw != 0.0)


def test_a_bigger_object_is_harder_to_see_than_a_smaller_one():
    """Never the other way round -- more of it has to fit in the same frame."""
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    for look in looks:
        for radius in (0.140, 0.170, 0.195):
            for bearing in (-40, 0, 40):
                x = radius * math.cos(math.radians(bearing))
                y = radius * math.sin(math.radians(bearing))
                if sweep.sees(look, x, y, sweep.OBJECT_SPAN_M):
                    assert sweep.sees(look, x, y, sweep.TAG_SPAN_M)


def test_sees_implies_the_point_itself_is_in_the_picture():
    """sees asks about an object; the point under it must then be in shot too."""
    width, height = cfg.WRIST_CAM_SIZE
    look = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    for radius in (0.130, 0.150, 0.170, 0.190, 0.205):
        for bearing in (-30, -10, 10, 30):
            x = radius * math.cos(math.radians(bearing))
            y = radius * math.sin(math.radians(bearing))
            if sweep.sees(look, x, y):
                px, py = sweep.to_pixel(look.matrix, x, y)
                assert 0 < px < width and 0 < py < height


# -------------------------------------------------------------- stations ----
def test_stations_rings_every_calibrated_look():
    calibrated = [(REAL_H, REAL_SURVEY, "primary"),
                  (REAL_H, {**REAL_SURVEY, 2: 50}, "outer")]
    one = sweep.ring(REAL_SURVEY, REAL_H)
    both = sweep.stations(calibrated)
    assert len(both) == 2 * len(one)


def test_stations_of_one_look_is_just_its_ring():
    only = sweep.stations([(REAL_H, REAL_SURVEY, "primary")])
    assert [look.dyaw for look in only] == [
        look.dyaw for look in sweep.ring(REAL_SURVEY, REAL_H)]


def test_best_refine_prefers_the_primary_calibration():
    """The primary is the only look validated end to end; it gets first refusal."""
    calibrated = [(REAL_H, REAL_SURVEY, "primary"),
                  (REAL_H, REAL_SURVEY, "outer")]
    look = sweep.best_refine(calibrated, 0.170, 0.020)
    assert look.pose[1] == sweep.refine_look(REAL_SURVEY, REAL_H, 0.170, 0.020).pose[1]


def test_best_refine_reports_every_reason_when_none_will_do():
    calibrated = [(REAL_H, REAL_SURVEY, "primary")]
    with pytest.raises(sweep.NoLook, match="primary"):
        sweep.best_refine(calibrated, 0.320, 0.0)


# ------------------------------------------------------------ refine look ----
def test_refine_puts_the_object_in_the_middle_of_the_picture():
    """Whatever bearing an object sits at, the refine look must centre it.

    Radii are the measured working band, 129..189 mm: past that the cube is thrown
    off the edge by its own parallax however the base is turned, which
    test_refine_refuses_what_it_cannot_centre pins from the other side.
    """
    width, height = cfg.WRIST_CAM_SIZE
    middle = np.array([width / 2, height / 2])
    for bearing in range(-70, 71, 10):
        for radius in (0.135, 0.160, 0.185):
            x = radius * math.cos(math.radians(bearing))
            y = radius * math.sin(math.radians(bearing))
            look = sweep.refine_look(REAL_SURVEY, REAL_H, x, y)
            pixel = sweep.to_pixel(look.matrix, x, y)
            assert sweep.sees(look, x, y)
            # Only the BEARING can be steered, so the object lands on the centre
            # line at its own radius -- across the frame it must be dead centre,
            # along it wherever the distance puts it.
            assert abs(pixel[0] - middle[0]) < 40, (
                f"{radius * 1000:.0f} mm at {bearing} deg landed at x={pixel[0]:.0f}")


def test_refine_only_turns_the_base():
    look = sweep.refine_look(REAL_SURVEY, REAL_H, 0.06, -0.16)
    for joint in (2, 3, 4, 5, 6):
        assert look.pose[joint] == REAL_SURVEY[joint]


def test_refine_reports_the_yaw_it_actually_commanded():
    """J1 is whole degrees, so the matrix must describe the pose as ROUNDED."""
    for bearing in range(-70, 71, 7):
        x = 0.165 * math.cos(math.radians(bearing))
        y = 0.165 * math.sin(math.radians(bearing))
        look = sweep.refine_look(REAL_SURVEY, REAL_H, x, y)
        assert look.dyaw == float(REAL_SURVEY[1] - look.pose[1])
        assert look.matrix == pytest.approx(sweep.rotate(REAL_H, look.dyaw))


def test_refine_refuses_what_it_cannot_centre():
    """Beyond the middle of the frame it must raise, not quietly do worse."""
    with pytest.raises(sweep.NoLook):
        sweep.refine_look(REAL_SURVEY, REAL_H, 0.320, 0.0)


def test_yaw_to_centre_is_the_inverse_of_the_look_bearing():
    for bearing in (-60.0, -12.5, 0.0, 33.0, 75.0):
        radius = 0.17
        x = radius * math.cos(math.radians(bearing))
        y = radius * math.sin(math.radians(bearing))
        dyaw = sweep.yaw_to_centre(REAL_H, x, y)
        assert dyaw + sweep.look_bearing(REAL_H) == pytest.approx(bearing)


def test_the_look_bearing_is_not_the_nadir():
    """Pinned because conflating the two is exactly the 20 mm bug detect.py hit."""
    centre = sweep.look_bearing(REAL_H)
    nadir = kin.camera_nadir(REAL_SURVEY)
    assert centre == pytest.approx(6.6, abs=1.0)
    assert math.degrees(math.atan2(nadir[1], nadir[0])) == pytest.approx(-20.0, abs=1.0)


# ----------------------------------------------------------------- merge ----
def _target(x, y, label="cube"):
    return detect.Target(x=x, y=y, width_m=0.040, length_m=0.040, angle_deg=0.0,
                         label=label)


def test_merge_collapses_one_object_seen_from_two_stations():
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    left = next(look for look in looks if look.dyaw == 25)
    home = next(look for look in looks if look.dyaw == 0)
    merged = sweep.merge([(home, _target(0.170, 0.030)),
                          (left, _target(0.171, 0.031))])
    assert len(merged) == 1


def test_merge_keeps_genuinely_separate_objects():
    home = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    merged = sweep.merge([(home, _target(0.150, 0.000)),
                          (home, _target(0.150, 0.060))])
    assert len(merged) == 2


def test_merge_prefers_the_view_that_saw_it_nearest_the_middle():
    """The tie-break that makes overlapping stations an advantage, not a wobble."""
    looks = sweep.ring(REAL_SURVEY, REAL_H)
    home = next(look for look in looks if look.dyaw == 0)
    far = next(look for look in looks if look.dyaw == 50)
    centred = _target(0.164, 0.019, label="good")     # the middle of home's frame
    edge = _target(0.166, 0.021, label="edgy")
    for order in ([(home, centred), (far, edge)], [(far, edge), (home, centred)]):
        merged = sweep.merge(order)
        assert len(merged) == 1
        assert merged[0].label == "good"


def test_merge_of_nothing_is_nothing():
    assert sweep.merge([]) == []


def test_merge_returns_nearest_first():
    home = sweep.Look(0.0, REAL_SURVEY, REAL_H)
    merged = sweep.merge([(home, _target(0.200, 0.0)), (home, _target(0.140, 0.0))])
    assert [round(t.x, 3) for t in merged] == [0.140, 0.200]


# ------------------------------------------------- the reachable envelope ----
def test_the_graspable_envelope_is_an_annulus_we_can_state():
    """Numbers quoted in the module docstring."""
    radii = np.hypot(GRASPABLE[:, 0], GRASPABLE[:, 1])
    bearings = np.degrees(np.arctan2(GRASPABLE[:, 1], GRASPABLE[:, 0]))
    # 0.131 with the J3 floor at 10, 0.122 at 0, 0.090 with pitches to 190
    assert radii.min() == pytest.approx(0.090, abs=0.006)
    # 0.210 with the old J2 floor of 15, 0.238 with 5
    assert radii.max() == pytest.approx(0.262, abs=0.006)
    assert bearings.min() == pytest.approx(-90, abs=2)   # -80 until J1's ceiling went to 180
    assert bearings.max() == pytest.approx(80, abs=2)
