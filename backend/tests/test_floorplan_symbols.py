import ast
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / '08_artifacts_generation/generate_floorplan_symbols.py'
CATEGORY_MODULE_PATH = Path(__file__).resolve().parents[1] / '05_semantic_segmentation/run_non_structural_category_similarity.py'
spec = importlib.util.spec_from_file_location('floorplan_symbols', MODULE_PATH)
symbols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbols)
NS = {'s': symbols.SVG_NS}


@pytest.mark.parametrize('class_id', symbols.CLASS_NAMES)
@pytest.mark.parametrize('width_end,length_end', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_all_classes_have_exact_canvas_and_valid_geometry(class_id, width_end, length_end):
    min_width, max_width, min_length, max_length = symbols.DIMENSION_LIMITS[class_id]
    width = (min_width, max_width)[width_end]
    length = (min_length, max_length)[length_end]
    root = ET.fromstring(symbols.sized_svg(class_id, width, length))
    assert root.attrib['width'] == f'{length*1000:g}mm'
    assert root.attrib['height'] == f'{width*1000:g}mm'
    assert root.attrib['viewBox'] == f'0 0 {length*1000:g} {width*1000:g}'
    assert root.attrib.get('preserveAspectRatio') != 'none'
    assert root.find('s:title', NS).text == symbols.CLASS_NAMES[class_id]
    assert len(root.find('s:g', NS)) > 0
    for node in root.iter():
        assert 'scale' not in node.attrib.get('transform', '')
        for key in ('width', 'height', 'r', 'rx', 'ry'):
            if key in node.attrib and not node.attrib[key].endswith('mm'):
                assert math.isfinite(float(node.attrib[key]))
                assert float(node.attrib[key]) >= 0 if key in ('rx', 'ry') else float(node.attrib[key]) > 0


def part(width, length, name):
    root = ET.fromstring(symbols.sized_svg(16, width, length))
    return root.find(f'.//s:rect[@data-part="{name}"]', NS).attrib


def test_sofa_keeps_arm_thickness_and_back_depth():
    for width, length in [(1, 2), (1, 4), (2, 2), (2, 4)]:
        assert part(width, length, 'left-arm')['width'] == '170'
        assert part(width, length, 'right-arm')['width'] == '170'
        assert part(width, length, 'back')['height'] == '190'
    assert part(1, 4, 'right-arm')['x'] != part(1, 2, 'right-arm')['x']
    assert float(part(1, 4, 'seat')['width']) > float(part(1, 2, 'seat')['width'])
    assert float(part(2, 2, 'seat')['height']) > float(part(1, 2, 'seat')['height'])


def test_gate_has_two_leaves_only_one_open_and_uniform_arc():
    root = ET.fromstring(symbols.sized_svg(8, 1, 2))
    leaves = root.findall('.//s:g[@transform]', NS)
    assert len(leaves) == 2
    assert leaves[0].attrib['transform'].startswith('rotate(-18 ')
    assert leaves[1].attrib['transform'].startswith('rotate(0 ')
    assert leaves[0].find('s:rect', NS).attrib['width'] == leaves[1].find('s:rect', NS).attrib['width']


def test_circular_details_keep_radius_when_canvas_grows():
    for class_id, sizes in ((7, ((1, 1), (2, 3))),
                            (19, ((1, 1), (1.1, 1.6))),
                            (23, ((1, 1), (1.2, 3))),
                            (26, ((1, 1), (2, 3))),
                            (27, ((.6, .6), (1, 1)))):
        roots = [ET.fromstring(symbols.sized_svg(class_id, width, length)) for width, length in sizes]
        radii = [[node.attrib['r'] for node in root.findall('.//s:circle', NS)] for root in roots]
        if class_id != 27:  # Washer door grows uniformly to fit its enclosure.
            assert radii[0] == radii[1]
        assert all('transform' not in node.attrib for root in roots for node in root.findall('.//s:circle', NS))


@pytest.mark.parametrize('class_id', [0, 29, 30, -1, True, 1.5, '16'])
def test_invalid_class_rejected(class_id):
    with pytest.raises(ValueError):
        symbols.sized_svg(class_id, 1, 2)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), 1.79e308])
def test_invalid_dimensions_rejected(bad):
    with pytest.raises(ValueError):
        symbols.sized_svg(16, bad, 2)
    with pytest.raises(ValueError):
        symbols.sized_svg(16, 1, bad)


def test_cli_writes_numbered_class(tmp_path):
    output = tmp_path / 'sofa.svg'
    assert symbols.main(['16', '.9', '2.4', '-o', str(output)]) == 0
    assert ET.parse(output).getroot().attrib['viewBox'] == '0 0 2400 900'


def test_numbered_class_names_match_current_ontology():
    source = CATEGORY_MODULE_PATH.read_text()
    tree = ast.parse(source)
    groups = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'CATEGORY_GROUPS'
                          for target in node.targets))
    assert all(symbols.CLASS_NAMES[category_id] == words[0]
               for category_id, words in groups.items())
    assert symbols.CLASS_NAMES[18] == 'bed'
    assert symbols.CLASS_NAMES[22] == 'dishwasher'
    assert symbols.CLASS_NAMES[28] == 'car'
    assert tuple(symbols.CLASS_NAMES) == tuple(range(1, 29))
    assert symbols.DIMENSION_LIMITS.keys() == symbols.CLASS_NAMES.keys()


@pytest.mark.parametrize('class_id', symbols.CLASS_NAMES)
@pytest.mark.parametrize('axis,upper', [(0, False), (0, True), (1, False), (1, True)])
def test_dimensions_just_outside_each_limit_are_rejected(class_id, axis, upper):
    min_w, max_w, min_l, max_l = symbols.DIMENSION_LIMITS[class_id]
    bounds = ((min_w, max_w), (min_l, max_l))
    dimensions = [(min_w+max_w)/2, (min_l+max_l)/2]
    dimensions[axis] = math.nextafter(bounds[axis][upper], math.inf if upper else -math.inf)
    with pytest.raises(ValueError, match=f'class {class_id} .*between .* inclusive; got'):
        symbols.sized_svg(class_id, *dimensions)


@pytest.mark.parametrize('size', [.4, 2])
def test_impossible_dishwasher_sizes_rejected(size):
    with pytest.raises(ValueError, match='dishwasher'):
        symbols.sized_svg(22, size, size)


def test_compact_and_slimline_dishwashers_accepted():
    for width, length in ((.5, .551), (.6, .45), (.6, .6)):
        assert ET.fromstring(symbols.sized_svg(22, width, length)).attrib['data-class-id'] == '22'


def test_final_association_render_can_skip_symbol_footprint_limits():
    with pytest.raises(ValueError, match='dishwasher'):
        symbols.sized_svg(22, .3, .3)
    root = ET.fromstring(symbols.sized_svg(22, .3, .3, validate_dimensions=False))
    assert root.attrib['viewBox'] == '0 0 300 300'
    with pytest.raises(ValueError, match='finite positive'):
        symbols.sized_svg(22, 0, .3, validate_dimensions=False)


def test_class_synonyms_have_room_without_allowing_deep_benches():
    for class_id, width, length in ((13, .45, 3.5), (16, .75, .75),
                                   (12, .35, .35), (26, 2.5, 3),
                                   (28, .6, 2), (20, .4, .4)):
        symbols.sized_svg(class_id, width, length)
    with pytest.raises(ValueError, match='chair'):
        symbols.sized_svg(13, 2, 3.5)


def test_cli_rejects_invalid_size_without_overwriting_output(tmp_path, capsys):
    output = tmp_path / 'dishwasher.svg'
    output.write_text('previous valid output')
    with pytest.raises(SystemExit) as error:
        symbols.main(['22', '2', '2', '-o', str(output)])
    assert error.value.code == 2
    assert output.read_text() == 'previous valid output'
    message = capsys.readouterr().err
    assert 'dishwasher' in message and '0.45 and 0.9 m inclusive' in message
