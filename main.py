import os
import subprocess
import platform
import itertools
from dataclasses import dataclass

dot_positions = {
    'V': [(0, 0), (1, 0), (1, 1), (0, 1)],
    'F': [(0.5, 0.5)],
    'F!': [(0.5, -0.2), (1.2, 0.5), (0.5, 1.2), (-0.2, 0.5)],
    'E': [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],
    'vf': [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
    'vf!': [(0.25, -0.2), (0.75, -0.2), (1.2, 0.25), (1.2, 0.75), (0.75, 1.2), (0.25, 1.2), (-0.2, 0.75), (-0.2, 0.25)],
    've': [(0.25, 0), (0.75, 0), (1, 0.25), (1, 0.75), (0.75, 1), (0.25, 1), (0, 0.75), (0, 0.25)],
    've_e': [(0.25, 0), (0.75, 0), (1, 0.25), (1, 0.75), (0.75, 1), (0.25, 1), (0, 0.75), (0, 0.25)],
    've_c': [(0.25, 0), (0.75, 0), (1, 0.25), (1, 0.75), (0.75, 1), (0.25, 1), (0, 0.75), (0, 0.25)],
    'fe': [(0.25, 0.5), (0.75, 0.5), (0.5, 0.75), (0.5, 0.25)],
    'fe!': [(-0.2, 0.5), (1.2, 0.5), (0.5, 1.22), (0.5, -0.2)]
}

# Interior point classes require degree >= MIN_DEGREE_INTERIOR unless a continuation edge is present
MIN_DEGREE_INTERIOR = 3
interior_point_classes = {'F', 'vf', 'fe'}
interior_positions = set(
    p for cls in interior_point_classes for p in dot_positions[cls]
)


atoms = {
    ('E', 'E'): [(0, [1]), (1, [2]), (2, [3]), (3, [0])],
    ('E', 'F'): [(0, [0]), (1, [0]), (2, [0]), (3, [0])],
    ('E', 'V'): [(0, [0, 1]), (1, [1, 2]), (2, [2, 3]), (3, [3, 0])],
    ('E', 've'): [(0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (3, [6, 7])],
    ('E', 'vf'): [(0, [0, 1]), (1, [1, 2]), (2, [2, 3]), (3, [3, 0])],
    ('E', 'fe'): [(0, [3]), (1, [1]), (2, [2]), (3, [0])],
    ('F', 'F!'): [(0, [0, 1, 2, 3])],
    ('F', 'V'): [(0, [0, 1, 2, 3])],
    ('F', 've'): [(0, [0, 1, 2, 3, 4, 5, 6, 7])],
    ('F', 'vf'): [(0, [0, 1, 2, 3])],
    ('F', 'fe'): [(0, [0, 1, 2, 3])],
    ('V', 'V'): [(0, [1, 3]), (1, [0, 2]), (2, [1, 3]), (3, [0, 2])],
    ('V', 've'): [(0, [0, 7]), (1, [1, 2]), (2, [3, 4]), (3, [5, 6])],
    ('V', 'vf'): [(0, [0]), (1, [1]), (2, [2]), (3, [3])],
    ('fe', 'V'): [(0, [3]), (0, [0]), (1, [2]), (1, [1]), (2, [2]), (2, [3]), (3, [0]), (3, [1])],
    # ve-ve: split into edge-adjacent (ve_e) and corner-adjacent (ve_c)
    ('ve_e', 've_e'): [(0, [1]), (1, [0]), (2, [3]), (3, [2]), (4, [5]), (5, [4]), (6, [7]), (7, [6])],
    ('ve_c', 've_c'): [(0, [7]), (1, [2]), (2, [1]), (3, [4]), (4, [3]), (5, [6]), (6, [5]), (7, [0])],
    ('ve', 'vf'): [(0, [0]), (1, [1]), (2, [1]), (3, [2]), (4, [2]), (5, [3]), (6, [3]), (7, [0])],
    ('fe', 've'): [(0, [6, 7]), (1, [2, 3]), (2, [4, 5]), (3, [0, 1])],
    ('vf', 'vf'): [(0, [1, 3]), (1, [0, 2]), (2, [1, 3]), (3, [0, 2])],
    ('vf', 'vf!'): [(0, [7, 0]), (1, [1, 2]), (2, [3, 4]), (3, [5, 6])],
    ('fe', 'vf'): [(0, [3, 0]), (1, [1, 2]), (2, [2, 3]), (3, [0, 1])],
    ('fe', 'fe'): [(0, [3]), (1, [2]), (2, [0]), (3, [1])],
    ('fe', 'fe!'): [(0, [0]), (1, [1]), (2, [2]), (3, [3])]
}

EPSILON = 1e-9


@dataclass(frozen=True)
class Segment:
    a: tuple
    b: tuple

    def __post_init__(self):
        # Canonical order so (a,b) == (b,a)
        if self.a > self.b:
            a, b = self.b, self.a
            object.__setattr__(self, 'a', a)
            object.__setattr__(self, 'b', b)


def cross2d(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    """Returns True if s1 and s2 intersect in their interiors (not at shared endpoints)."""
    p1, p2 = s1.a, s1.b
    p3, p4 = s2.a, s2.b

    # Shared endpoint — touching is allowed
    shared = {p1, p2} & {p3, p4}
    if len(shared) == 2:
        return True  # duplicate segment
    # For shared endpoint: non-collinear touching is fine, but collinear overlap is not.
    # Fall through — the parametric check returns False for non-collinear shared endpoints
    # (t or u lands on 0 or 1, outside the open interval), and the collinear path
    # correctly detects one segment containing the other.

    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]

    denom = d1x * d2y - d1y * d2x

    if abs(denom) < EPSILON:
        # Parallel — check if actually collinear before testing overlap
        cross = (p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x
        if abs(cross) > EPSILON:
            return False  # parallel but not collinear
        return collinear_overlap(s1, s2)

    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / denom

    return EPSILON < t < 1 - EPSILON and EPSILON < u < 1 - EPSILON


def collinear_overlap(s1: Segment, s2: Segment) -> bool:
    """Returns True if two collinear segments overlap (share more than a point)."""
    # Project onto whichever axis has more spread
    dx = s1.b[0] - s1.a[0]
    dy = s1.b[1] - s1.a[1]

    if abs(dx) > abs(dy):
        a1, b1 = s1.a[0], s1.b[0]
        a2, b2 = s2.a[0], s2.b[0]
    else:
        a1, b1 = s1.a[1], s1.b[1]
        a2, b2 = s2.a[1], s2.b[1]

    lo1, hi1 = min(a1, b1), max(a1, b1)
    lo2, hi2 = min(a2, b2), max(a2, b2)

    overlap = min(hi1, hi2) - max(lo1, lo2)
    return overlap > EPSILON


def _base_class(cls: str) -> str:
    """Normalise a point class name to its base class for rank calculation."""
    cls = cls.rstrip('!')
    for suffix in ('_e', '_c'):
        if cls.endswith(suffix):
            return cls[:-len(suffix)]
    return cls


def combination_rank(combo) -> int:
    """Number of distinct base point classes in an atom combination."""
    return len({_base_class(cls) for atom in combo for cls in atom})


def connectivity_check(segments: list) -> bool:
    """Returns True if the segment graph is connected (single component)."""
    if not segments:
        return False
    adj = {}
    for seg in segments:
        adj.setdefault(seg.a, set()).add(seg.b)
        adj.setdefault(seg.b, set()).add(seg.a)
    nodes = list(adj)
    visited = {nodes[0]}
    queue = [nodes[0]]
    while queue:
        node = queue.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited == set(nodes)


def has_crossing_or_duplicate(segments: list) -> bool:
    """Returns True if any two segments cross or are duplicates."""
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i], segments[j]):
                return True
    return False


def degree_check(segments: list, points: list) -> bool:
    """Returns True if all interior vertices (F, vf, fe) have degree >= MIN_DEGREE_INTERIOR."""
    degree = {}
    for seg in segments:
        degree[seg.a] = degree.get(seg.a, 0) + 1
        degree[seg.b] = degree.get(seg.b, 0) + 1

    for pt in set(points):
        if pt not in interior_positions:
            continue
        if degree.get(pt, 0) < MIN_DEGREE_INTERIOR:
            return False
    return True


# Corner positions — shared by 4 cells, too complex to analyse locally
corner_positions = set(map(tuple, dot_positions['V']))


def _boundary_edge(pt):
    """Returns the edge index (0=bottom,1=right,2=top,3=left) for a non-corner boundary point,
    or None if the point is a corner, interior, or outside the cell."""
    if pt in corner_positions:
        return None
    x, y = pt
    if abs(y) < EPSILON:       return 0
    if abs(x - 1) < EPSILON:   return 1
    if abs(y - 1) < EPSILON:   return 2
    if abs(x) < EPSILON:       return 3
    return None


def _reflect(pt, edge_idx):
    x, y = pt
    if edge_idx == 0: return (x, -y)
    if edge_idx == 1: return (2 - x, y)
    if edge_idx == 2: return (x, 2 - y)
    if edge_idx == 3: return (-x, y)


def boundary_check(segments: list, points: list) -> bool:
    """For each non-corner boundary point (E, ve), simulate the adjacent cell's contribution
    by reflecting neighbours across the shared edge. If the resulting global degree is still
    below MIN_DEGREE_INTERIOR, the point is unresolvable."""
    degree = {}
    incident = {}
    for seg in segments:
        for pt, other in ((seg.a, seg.b), (seg.b, seg.a)):
            degree[pt] = degree.get(pt, 0) + 1
            incident.setdefault(pt, set()).add(other)

    for pt in set(points):
        local_deg = degree.get(pt, 0)
        if local_deg >= MIN_DEGREE_INTERIOR:
            continue
        edge_idx = _boundary_edge(pt)
        if edge_idx is None:
            continue  # corner, interior, or outside — not handled here
        neighbors = frozenset(incident.get(pt, set()))
        reflected = frozenset(_reflect(n, edge_idx) for n in neighbors)
        new_connections = reflected - neighbors - {pt}
        global_deg = local_deg + len(new_connections)
        if global_deg < MIN_DEGREE_INTERIOR:
            return False
    return True


def create_data(line_config):
    line_config = line_config.split(',')
    points = []
    segments = []

    for line in line_config:
        group_a, group_b = line.split('-')
        if group_a.upper() > group_b.upper():
            start_group, end_group = group_b, group_a
        else:
            start_group, end_group = group_a, group_b
        key = (start_group, end_group)

        start_points = dot_positions[start_group]
        end_points = dot_positions[end_group]
        all_indices = atoms[key]
        omit_end = key[1].endswith("!")

        points += list(start_points)
        if not omit_end:
            points += list(end_points)

        atom_segments = []
        for connection_tuple in all_indices:
            start = start_points[connection_tuple[0]]
            for i in connection_tuple[1]:
                end = end_points[i]
                atom_segments.append(Segment(start, end))
        # Deduplicate within atom (atom definitions list each edge from both ends)
        segments.extend(dict.fromkeys(atom_segments))

    return segments, points


RENDER_MODE = 'debug'  # 'debug' or 'final'

colors = ['red', 'green', 'blue']
widths = ['0.1', '0.05', '0.025']
opacities = ['0.3', '0.5', '0.7']


def create_svg(all_segs, all_pts, indices, mode=RENDER_MODE):
    unique_dots = list(set(all_pts))
    padding = 0.1
    page_size = 200
    viewbox_size = 2 + 2 * padding
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-{padding} -{padding} {viewbox_size} {viewbox_size}" width="{page_size}" height="{page_size}">'
    svg_content = ""

    if mode == 'final':
        for seg in all_segs:
            svg_content += f'<line x1="{seg.a[0]}" y1="{seg.a[1]}" x2="{seg.b[0]}" y2="{seg.b[1]}" stroke="black" stroke-width="0.02"/>'
        for x, y in unique_dots:
            svg_content += f'<circle cx="{x}" cy="{y}" r="0.05" fill="red" stroke="black" stroke-width="0.01"/>'
    else:
        for count, seg in enumerate(all_segs):
            idx = indices[count]
            svg_content += f'<line x1="{seg.a[0]}" y1="{seg.a[1]}" x2="{seg.b[0]}" y2="{seg.b[1]}" stroke="{colors[idx]}" opacity="{opacities[idx]}" stroke-width="{widths[idx]}"/>'
        for x, y in unique_dots:
            svg_content += f'<circle cx="{x}" cy="{y}" r="0.05" fill="purple" stroke="black" stroke-width="0.01"/>'

    svg_footer = '</svg>'
    return f"{svg_header}{svg_content}{svg_footer}"


def open_svg(filename):
    os_name = platform.system()
    if os_name == 'Darwin':
        subprocess.run(['open', filename], check=True)
    elif os_name == 'Windows':
        subprocess.run(['start', filename], shell=True, check=True)


if __name__ == '__main__':
    seen_combos = set()
    for target_rank in range(1, 3):
        for n_atoms in range(1, 4):
            for combo in itertools.combinations(atoms.keys(), r=n_atoms):
                if combination_rank(combo) != target_rank:
                    continue
                if combo in seen_combos:
                    continue
                seen_combos.add(combo)

                pair = combo
                all_segments = []
                pair_indices = []
                all_dots = []
                pair_index = 0

                for atom in pair:
                    label = '-'.join(atom)
                    segments, dots = create_data(label)
                    all_segments += segments
                    pair_indices += [pair_index] * len(segments)
                    pair_index += 1
                    all_dots += dots

                readable_name = str(pair).replace(' ', '').replace('\',\'', '-').replace('(', '').replace(')', '').replace('\'', '')
                if readable_name.endswith(','): readable_name = readable_name[:-1]
                svg_filename = f'{readable_name}.svg'

                if has_crossing_or_duplicate(all_segments):
                    folder = 'svgs_bad'
                elif not degree_check(all_segments, all_dots) or not boundary_check(all_segments, all_dots) or not connectivity_check(all_segments):
                    folder = 'svgs_low_degree'
                else:
                    folder = 'svgs_good'

                class_subdir = '.'.join(sorted({_base_class(cls) for atom in pair for cls in atom}))
                svg_filename = f'{folder}/{class_subdir}/{svg_filename}'
                os.makedirs(f'{folder}/{class_subdir}', exist_ok=True)
                svg = create_svg(all_segments, all_dots, pair_indices)
                with open(svg_filename, 'w') as file:
                    file.write(svg)