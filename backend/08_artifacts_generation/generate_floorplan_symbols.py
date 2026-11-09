#!/usr/bin/env python3
"""Parametric floorplan SVGs: class ID, width (depth), length (horizontal), in metres."""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = 'http://www.w3.org/2000/svg'
# IDs match CATEGORY_GROUPS in category similarity and the numbered SVG files.
CLASS_NAMES = {
    1: 'door', 2: 'window', 3: 'stairs', 4: 'ramp', 5: 'elevator',
    6: 'column', 7: 'radiator', 8: 'gate', 9: 'fountain', 10: 'gazebo',
    11: 'table', 12: 'cabinet', 13: 'chair', 14: 'rug', 15: 'bookshelf',
    16: 'sofa', 17: 'television', 18: 'bed', 19: 'cooktop',
    20: 'refrigerator', 21: 'oven', 22: 'dishwasher', 23: 'sink',
    24: 'wc', 25: 'bidet', 26: 'shower', 27: 'washing machine', 28: 'car',
}


# Broad household/architectural footprint envelopes, not building-code limits.
# Each entry is (min depth/Y, max depth/Y, min horizontal length/X, max length/X), metres.
# Use the union of class synonyms: e.g. chair includes bench, shower includes tub/jacuzzi.
DIMENSION_LIMITS = {
    1:  (0.04, 3.5, 0.35, 3.5),   # Door, door-frame, doorway; canvas may include swing.
    2:  (0.03, 1.2, 0.15, 10.0),  # Window, curtain.
    3:  (0.50, 20.0, 0.45, 8.0),  # Stairs, staircase, steps, stairway.
    4:  (0.50, 30.0, 0.50, 8.0),  # Ramp.
    5:  (0.60, 5.0, 0.60, 5.0),   # Elevator, lift.
    6:  (0.08, 3.0, 0.08, 3.0),   # Column, pillar.
    7:  (0.08, 2.0, 0.15, 4.0),   # Radiator, convector, heater, fireplace, chimney.
    8:  (0.03, 5.0, 0.45, 12.0),  # Gate; allow shallow closed/near-closed footprints.
    9:  (0.20, 12.0, 0.20, 12.0), # Fountain.
    10: (0.80, 15.0, 0.80, 15.0), # Gazebo.
    11: (0.20, 2.5, 0.30, 8.0),   # Table, desk, dining/coffee table, worktop.
    12: (0.15, 1.3, 0.15, 8.0),   # Cabinet, closet, wardrobe, nightstand, cupboard, drawer.
    13: (0.25, 1.2, 0.25, 4.0),   # Chair, office chair, bench: long, but not deep.
    14: (0.20, 15.0, 0.20, 20.0), # Rug, carpet, mat.
    15: (0.10, 1.2, 0.20, 10.0),  # Bookshelf, shelf, ledge, rack.
    16: (0.50, 3.5, 0.50, 8.0),   # Sofa, couch, armchair; includes chaise/corner footprints.
    17: (0.02, 0.8, 0.18, 3.5),   # Television, monitor, including stand depth.
    18: (0.65, 2.8, 0.40, 3.2),   # Bed, crib.
    19: (0.25, 1.1, 0.25, 1.6),   # Cooktop.
    20: (0.30, 1.2, 0.30, 1.8),   # Refrigerator, fridge; compact to double-door.
    21: (0.30, 1.0, 0.30, 1.2),   # Oven, including compact/countertop.
    22: (0.45, 0.9, 0.45, 0.9),   # Dishwasher: compact, slimline and full-size.
    23: (0.20, 1.2, 0.20, 3.0),   # Sink: small basin to long multi-bowl unit.
    24: (0.40, 1.0, 0.25, 0.75),  # WC, toilette.
    25: (0.35, 0.85, 0.25, 0.65), # Bidet.
    26: (0.60, 3.5, 0.60, 4.0),   # Shower, jacuzzi, tub.
    27: (0.35, 1.1, 0.40, 1.0),   # Washing machine, dryer; shallow/top-loading models.
    28: (0.40, 3.0, 0.90, 7.0),   # Car, motorbike.
}


class Drawing:
    """Coordinates are millimetres; details have physical sizes, not scale transforms."""

    def __init__(self, length: float, width: float):
        self.root = ET.Element('svg', xmlns=SVG_NS, width=f'{length:g}mm',
                               height=f'{width:g}mm', viewBox=f'0 0 {length:g} {width:g}')
        ET.SubElement(self.root, 'rect', width=f'{length:g}', height=f'{width:g}', fill='white')
        stroke_width = min(16, max(8, length / 100, width / 100))
        self.group = ET.SubElement(self.root, 'g', fill='white', stroke='black',
                                   **{'stroke-width': f'{stroke_width:g}',
                                      'stroke-linejoin': 'round', 'stroke-linecap': 'round'})

    def element(self, tag, **attrs):
        return ET.SubElement(self.group, tag, {k.replace('_', '-'): f'{v:g}' if isinstance(v, (float, int)) else v for k, v in attrs.items()})

    def rect(self, x, y, w, h, r=12, **attrs):
        return self.element('rect', x=x, y=y, width=w, height=h, rx=min(r, w / 2, h / 2), **attrs)

    def circle(self, x, y, r, **attrs):
        return self.element('circle', cx=x, cy=y, r=r, **attrs)

    def path(self, d, **attrs):
        return self.element('path', d=d, fill='none', **attrs)

    def line(self, x1, y1, x2, y2, **attrs):
        return self.element('line', x1=x1, y1=y1, x2=x2, y2=y2, **attrs)


def positions(start, end, spacing):
    # Bound output size for unusually large footprints.
    count = min(200, max(1, math.ceil((end - start) / spacing)))
    return [start + (end - start) * i / count for i in range(1, count)]


def sized_svg(
    class_id: int,
    width_m: float,
    length_m: float,
    *,
    validate_dimensions: bool = True,
) -> str:
    """Return a standalone SVG. Length runs along X; width/depth runs along Y.

    Out-of-range footprints raise ValueError by default; bounds in DIMENSION_LIMITS are inclusive.
    Set validate_dimensions=False only when an upstream step already accepted the instance.
    Fixed details shrink only when a valid footprint is too small to contain them.
    All classes use explicit geometry rules; no anisotropic SVG scaling is used.
    """
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in CLASS_NAMES:
        raise ValueError('class ID must be an integer from 1 to 28')
    for name, value in (('width', width_m), ('length', length_m)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number of metres')
    if validate_dimensions:
        min_width, max_width, min_length, max_length = DIMENSION_LIMITS[class_id]
        for name, value, lower, upper in (
            ('width/depth (Y)', width_m, min_width, max_width),
            ('length (X)', length_m, min_length, max_length),
        ):
            if not lower <= value <= upper:
                raise ValueError(
                    f'class {class_id} ({CLASS_NAMES[class_id]}): {name} must be '
                    f'between {lower:g} and {upper:g} m inclusive; got {value:g} m'
                )
    w, h = length_m * 1000, width_m * 1000
    d = Drawing(w, h)
    # Small uniform inset keeps strokes within the exact requested canvas.
    m = min(20, w * .04, h * .04)
    x, y, a, b = m, m, w - 2*m, h - 2*m
    cx, cy = w/2, h/2
    t = min(45, a/12, b/12)
    d.root.set('data-class-id', str(class_id))
    ET.SubElement(d.root, 'title').text = CLASS_NAMES[class_id]

    if class_id == 1:  # Door: leaf and swing arc share one radius.
        r = min(a, b) * .9
        d.line(x, y, x, y+r)
        d.path(f'M{x} {y+r} A{r} {r} 0 0 0 {x+r} {y}')
        d.line(x, y, x+r, y, stroke_dasharray='12 12')
        d.circle(x, y, min(9, t/3))
    elif class_id == 2:  # Window: frame depth stays fixed.
        depth = min(100, b)
        d.rect(x, cy-depth/2, a, depth, 0)
        d.line(x, cy, x+a, cy)
        for px in positions(x, x+a, 450):
            d.line(px, cy-depth/2, px, cy+depth/2)
    elif class_id == 3:  # Stairs: add treads rather than stretching their spacing.
        d.rect(x, y, a, b, 0)
        for py in positions(y, y+b, 220):
            d.line(x, py, x+a, py)
        d.line(cx, y+b*.85, cx, y+b*.15)
        d.path(f'M{cx-t} {y+b*.15+t} L{cx} {y+b*.15} L{cx+t} {y+b*.15+t}')
    elif class_id == 4:  # Ramp: border and arrowhead retain thickness.
        d.rect(x, y, a, b)
        for f in (.25, .5, .75):
            d.line(x+t, y+b*f, x+a-t, y+b*f)
        d.line(cx, y+b*.85, cx, y+b*.15)
        d.path(f'M{cx-t} {y+b*.15+t} L{cx} {y+b*.15} L{cx+t} {y+b*.15+t}')
    elif class_id == 5:  # Elevator: centered doors and fixed-size inward arrows.
        d.rect(x, y, a, b)
        d.line(cx, y, cx, y+b)
        for sign in (-1, 1):
            px = cx+sign*a*.25
            d.path(f'M{px+sign*t} {cy-t} L{px} {cy} L{px+sign*t} {cy+t}')
    elif class_id == 6:  # Column: round even for a rectangular footprint.
        r = min(a, b)/2
        d.circle(cx, cy, r)
        d.circle(cx, cy, r-min(45, r*.2))
        d.line(cx-r*.4, cy-r*.4, cx+r*.4, cy+r*.4)
        d.line(cx-r*.4, cy+r*.4, cx+r*.4, cy-r*.4)
    elif class_id == 7:  # Radiator: fixed fin pitch and round valves.
        d.rect(x, y, a, b)
        for px in positions(x, x+a, 90):
            d.line(px, y+t, px, y+b-t)
        for px in (x+t, x+a-t):
            d.circle(px, y+b-t, min(12, t/3))
    elif class_id == 8:  # Gate: two equal leaves; only the left opens.
        post = min(35, a/20, b/10)
        left, right = x+post, x+a-post
        leaf = (right-left)/2
        thick = min(40, b/8, leaf/8)
        # Limit opening on shallow footprints without distorting the leaves.
        angle = min(math.radians(18), math.asin(min(1, (b-2*post-thick)/leaf)))
        angle = max(0, angle)
        baseline = y+b-post-thick
        for px in (left, right):
            d.circle(px, baseline, post)
        for hinge, direction, rotation in ((left, 1, -math.degrees(angle)), (right, -1, 0)):
            group = d.group
            d.group = ET.SubElement(group, 'g', transform=f'rotate({rotation:g} {hinge:g} {baseline:g})')
            start = hinge if direction == 1 else hinge-leaf
            d.rect(start, baseline-thick/2, leaf, thick, 4)
            for px in positions(start, start+leaf, 90):
                d.line(px, baseline-thick/2, px, baseline+thick/2)
            d.group = group
            d.circle(hinge, baseline, post*.3)
        ex, ey = left+leaf*math.cos(angle), baseline-leaf*math.sin(angle)
        d.path(f'M{ex} {ey} A{leaf} {leaf} 0 0 1 {cx} {baseline}', stroke_dasharray='12 12')
    elif class_id == 9:  # Fountain: concentric bowls, not stretched circles.
        r = min(a, b)/2
        for radius in (r, r*.75, r*.3):
            d.circle(cx, cy, radius)
        for sign in (-1, 1):
            d.path(f'M{cx} {cy} Q{cx+sign*r*.8} {cy-r*.8} {cx+sign*r*.65} {cy}')
    elif class_id == 10:  # Gazebo: roof outline grows; corner posts stay round.
        d.rect(x, y, a, b, 0)
        for px in (x+t, x+a-t):
            for py in (y+t, y+b-t):
                d.line(cx, cy, px, py)
                d.circle(px, py, min(22, t/2))
    elif class_id == 11:  # Table: fixed rim width and corner radius.
        d.rect(x, y, a, b)
        d.rect(x+t, y+t, a-2*t, b-2*t, 6, fill='none')
    elif class_id == 12:  # Cabinet: panel sizes grow, handles do not.
        d.rect(x, y, a, b)
        d.line(cx, y, cx, y+b)
        d.line(x, cy, x+a, cy)
        for sign in (-1, 1):
            d.circle(cx+sign*t*.6, cy, min(12, t/4))
    elif class_id == 13:  # Chair: back depth and leg size stay fixed.
        back = min(120, b*.22)
        d.rect(x, y, a, back)
        d.rect(x+t, y+back+t/2, a-2*t, b-back-t/2, 24)
        for px in (x+t, x+a-2*t):
            for py in (y+back+t/2, y+b-t):
                d.rect(px, py, t, t, 4)
    elif class_id == 14:  # Rug: fixed border and fringe length.
        fringe = min(35, b/10)
        d.rect(x, y+fringe, a, b-2*fringe)
        d.rect(x+t, y+fringe+t, a-2*t, b-2*fringe-2*t, 6, fill='none')
        for px in positions(x, x+a, 65):
            d.line(px, y, px, y+fringe)
            d.line(px, y+b-fringe, px, y+b)
    elif class_id == 15:  # Bookshelf: add bays instead of widening every bay.
        d.rect(x, y, a, b)
        for px in positions(x, x+a, 350):
            d.line(px, y, px, y+b)
    elif class_id == 16:  # Sofa: arm thickness and back depth independent of seat size.
        arm, back = min(170, a*.2), min(190, b*.25)
        d.rect(x, y, arm, b, 35, data_part='left-arm')
        d.rect(x+a-arm, y, arm, b, 35, data_part='right-arm')
        d.rect(x+arm, y, a-2*arm, back, 30, data_part='back')
        gap = min(30, b*.04)
        d.rect(x+arm, y+back+gap, a-2*arm, b-back-gap, 20, data_part='seat')
    elif class_id == 17:  # TV: bezel and stand stay fixed in depth.
        stand = min(120, b*.2)
        d.rect(x, y, a, b-stand, 22)
        d.rect(x+t, y+t, a-2*t, b-stand-2*t, 12)
        d.line(cx, y+b-stand, cx, y+b)
        d.line(cx-a*.2, y+b, cx+a*.2, y+b)
    elif class_id == 18:  # Bed: pillow depth and blanket folds stay fixed.
        pillow = min(215, b*.3)
        d.rect(x, y, a, b, 35)
        d.rect(x+t, y+t, a-2*t, pillow, 24)
        d.line(cx, y+t, cx, y+t+pillow)
        d.rect(x+t, y+pillow+2*t, a-2*t, b-pillow-3*t, 6)
        d.line(x+2*t, y+b-2*t, x+a-2*t, y+b-2*t)
    elif class_id == 19:  # Cooktop: circular burners, spaced by the footprint.
        d.rect(x, y, a, b, 20)
        r = min(145, a*.2, b*.2)
        for px in (x+a*.25, x+a*.75):
            for py in (y+b*.25, y+b*.75):
                d.circle(px, py, r)
                d.circle(px, py, r*.48)
    elif class_id == 20:  # Refrigerator: divider grows; handles remain a fixed length.
        d.rect(x, y, a, b, 20)
        split = y+b*.425
        d.line(x, split, x+a, split)
        for start, end in ((y, split), (split, y+b)):
            handle = min(175, (end-start)*.65)
            d.line(x+a-t, (start+end-handle)/2, x+a-t, (start+end+handle)/2)
    elif class_id == 21:  # Oven: fixed control strip and round knobs.
        panel = min(170, b*.25)
        d.rect(x, y, a, b)
        d.line(x, y+panel, x+a, y+panel)
        for fraction in (.2, .4, .6, .8):
            d.circle(x+a*fraction, y+panel/2, min(24, panel/5, a/15))
        d.rect(x+t, y+panel+t, a-2*t, b-panel-2*t)
        d.line(cx-a*.15, y+panel+2*t, cx+a*.15, y+panel+2*t)
    elif class_id == 22:  # Dishwasher: panel depth and plate circles preserved.
        panel = min(130, b*.2)
        d.rect(x, y, a, b)
        d.line(x+t, y+panel, x+a-t, y+panel)
        r = min(90, a/7, (b-panel)/4)
        for fraction in (.3, .5, .7):
            d.circle(x+a*fraction, y+panel+(b-panel)/2, r)
    elif class_id == 23:  # Sink: rim/faucet fixed, basin grows and drain stays circular.
        deck = min(160, b*.3)
        d.rect(x, y, a, b, 35)
        d.rect(x+t, y+deck, a-2*t, b-deck-t, 95)
        d.circle(cx, y+deck+(b-deck)*.6, min(30, t*.6))
        for px in (cx-a*.2, cx+a*.2):
            d.circle(px, y+deck/2, min(24, deck/5, a/20))
        d.rect(cx-t/2, y+deck*.3, t, deck, 16)
    elif class_id == 24:  # WC: tank depth fixed; bowl remains an oval by design.
        tank = min(190, b*.28)
        d.rect(x+a*.15, y, a*.7, tank, 28)
        bowl_y, bowl_h = y+tank+t/2, b-tank-t/2
        d.rect(x, bowl_y, a, bowl_h, min(a/2, bowl_h/2))
        d.element('ellipse', cx=cx, cy=bowl_y+bowl_h/2, rx=a*.36, ry=bowl_h*.4, fill='none')
        d.line(cx-t, y+tank/2, cx+t, y+tank/2)
    elif class_id == 25:  # Bidet: bowl inset and round drain, fixed faucet.
        d.rect(x, y, a, b, min(a/2, b/3))
        d.element('ellipse', cx=cx, cy=y+b*.58, rx=a*.33, ry=b*.3, fill='none')
        d.circle(cx, y+b*.7, min(18, t/2))
        d.line(cx-t, y+t, cx+t, y+t)
        d.line(cx, y+t/2, cx, y+2*t)
    elif class_id == 26:  # Shower: tray grows; head, drain and rim do not stretch.
        d.rect(x, y, a, b, 28)
        d.rect(x+t, y+t, a-2*t, b-2*t, 20, fill='none')
        drain_y = y+b*.7
        r = min(75, a*.15, b*.15)
        for px in (x+t, x+a-t):
            for py in (y+t, y+b-t):
                d.line(px, py, cx, drain_y)
        d.circle(cx, drain_y, min(35, r/2))
        d.line(cx, y, cx, y+t+r)
        d.circle(cx, y+t+2*r, r)
        d.circle(cx, y+t+2*r, r*.73)
        for dx, dy in ((-.25, -.25), (.25, -.25), (0, 0), (-.25, .25), (.25, .25)):
            d.circle(cx+r*dx, y+t+2*r+r*dy, r*.06, fill='black', stroke='none')
    elif class_id == 27:  # Washer: circular door fits below a fixed control panel.
        panel = min(150, b*.25)
        d.rect(x, y, a, b, 28)
        d.rect(x+t, y+t, a-2*t, panel-t)
        r = min(a*.4, (b-panel)*.42)
        d.circle(cx, y+panel+(b-panel)/2, r)
        d.circle(cx, y+panel+(b-panel)/2, r*.78)
        d.circle(x+2*t, y+panel/2, min(12, t/4))
    elif class_id == 28:  # Car: body grows; wheels retain their own proportions.
        wheel_w, wheel_h = min(100, a*.15), min(200, b*.2)
        for px in (x, x+a-wheel_w):
            for py in (y+b*.18, y+b*.7):
                d.rect(px, py, wheel_w, wheel_h, 18)
        d.rect(x+wheel_w/2, y, a-wheel_w, b, min(100, a*.2))
        d.rect(x+wheel_w, y+b*.22, a-2*wheel_w, b*.5, 30)
        d.line(x+wheel_w, y+b*.36, x+a-wheel_w, y+b*.36)
    return ET.tostring(d.root, encoding='unicode')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('class_id', type=int, help='class number, 1–28 (matches classifier and SVG filenames)')
    parser.add_argument('width', type=float, help='front-to-back depth in metres (SVG Y)')
    parser.add_argument('length', type=float, help='horizontal length in metres (SVG X)')
    parser.add_argument('-o', '--output', type=Path, help='write SVG here instead of stdout')
    args = parser.parse_args(argv)
    try:
        svg = sized_svg(args.class_id, args.width, args.length)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(svg + '\n', encoding='utf-8')
        else:
            print(svg)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
