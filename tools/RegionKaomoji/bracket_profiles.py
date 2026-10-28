"""Conservative, auditable bracket profiles for the opt-in research path.

Input is one tightly bounded connected ink component, never recognized text.
Every row must contain a single interval. Square caps need an actual horizontal
run on one row; round curves need both monotone halves and a stable center;
braces need two stable shoulders and a localized central point. Conflicting
families produce no shape. These fixed pixel rules are independent of labels,
models, datasets and the legacy profile implementation.
"""
from __future__ import annotations

import math

import numpy as np


def _square(mask, widths):
    """Return straight-stem/cap witnesses and all tested stem measurements."""
    height, width = mask.shape
    if width < 3:
        return None, []
    band = max(1, math.ceil(height * .20))
    middle = mask[band:height-band]
    if not len(middle) or int(max(widths[band:height-band])) > max(1, math.ceil(width * .35)):
        return None, []
    support = middle.sum(axis=0)
    tied = np.flatnonzero(support == support.max())
    required = max(2, math.ceil(width * .50))
    edge_band = math.floor(width * .25)
    observations, matches = [], []
    # Both outer tied columns are necessary for thick mirrored square stems.
    for stem in dict.fromkeys((int(tied[0]), int(tied[-1]))):
        coverage = float(np.count_nonzero(mask[:, stem]) / height)
        if coverage < .85:
            continue
        if stem <= edge_band:
            direction, step = 'left', 1
        elif stem >= width - 1 - edge_band:
            direction, step = 'right', -1
        else:
            continue
        facts = {'stem_column': stem, 'stem_coverage': coverage,
                 'required_cap_extension': required,
                 'maximum_middle_ink_width': int(max(widths[band:height-band]))}
        for name, rows in (('top', range(band)), ('bottom', range(height-band, height))):
            extension, witness = 0, None
            for row in rows:
                if not mask[row, stem]:
                    continue
                end = stem
                while 0 <= end + step < width and mask[row, end + step]:
                    end += step
                length = abs(end - stem)
                if length > extension:
                    extension, witness = length, row
            facts[f'{name}_cap_extension'] = extension
            facts[f'{name}_cap_witness_row'] = witness
        observations.append({'direction': direction, **facts})
        if min(facts['top_cap_extension'], facts['bottom_cap_extension']) >= required:
            matches.append({'kind': 'right-angle', 'direction': direction, 'measurements': facts})
    directions = {item['direction'] for item in matches}
    # Several columns can belong to the same thick stem. They are one family,
    # not competing observations; choose its outer column by the longer cap.
    selected = max(matches, key=lambda item: min(item['measurements']['top_cap_extension'],
                                                item['measurements']['bottom_cap_extension'])) if matches else None
    return (selected if len(directions) == 1 else None), observations


def _round(profile, widths, width):
    """Require a narrow stroke through the full U-shaped center-line profile."""
    height = len(profile)
    # A horizontal cap that spans the component cannot also prove a round arc.
    if int(max(widths)) > max(2, math.ceil(width * .55)):
        return None
    middle = profile[height//3:max(height//3+1, 2*height//3)]
    middle_range = float(np.ptp(middle))
    if middle_range > .75:
        return None
    center = float(np.median(middle))
    end_band = max(1, height//5)
    midpoint = height//2
    for direction, sign in (('left', 1), ('right', -1)):
        oriented = sign * (profile - center)
        upper = float(np.max(oriented[:end_band]))
        lower = float(np.max(oriented[-end_band:]))
        if upper < .75 or lower < .75:
            continue
        # Half-pixel center quantization is permitted, but a second bend or
        # reversal elsewhere on either half is not silently averaged away.
        upper_reversal = float(max(0.0, np.max(np.diff(oriented[:midpoint+1]))))
        lower_reversal = float(max(0.0, -np.min(np.diff(oriented[midpoint:]))))
        if max(upper_reversal, lower_reversal) > .5 or float(np.min(oriented)) < -.25:
            continue
        average = float((np.mean(oriented[:end_band]) + np.mean(oriented[-end_band:])) / 2)
        return {'kind': 'small-curve' if average < 1.0 else 'curve', 'direction': direction,
                'measurements': {'top_endpoint_displacement': upper,
                                 'bottom_endpoint_displacement': lower,
                                 'middle_profile_range': middle_range,
                                 'upper_profile_reversal': upper_reversal,
                                 'lower_profile_reversal': lower_reversal}}
    return None


def _brace(profile, widths, width):
    """Require stable upper/lower shoulders on both sides of a narrow point."""
    height = len(profile)
    if height < 12 or int(max(widths)) > max(2, math.ceil(width * .55)):
        return None
    upper = profile[math.ceil(height*.15):math.floor(height*.40)]
    lower = profile[math.ceil(height*.60):math.floor(height*.85)]
    if min(len(upper), len(lower)) < 2:
        return None
    upper_range, lower_range = float(np.ptp(upper)), float(np.ptp(lower))
    if max(upper_range, lower_range) > .5:
        return None
    shoulder_top, shoulder_bottom = float(np.median(upper)), float(np.median(lower))
    if abs(shoulder_top - shoulder_bottom) > .5:
        return None
    shoulder = (shoulder_top + shoulder_bottom) / 2
    band = max(1, height//6)
    for direction, sign in (('left', 1), ('right', -1)):
        oriented = sign * (profile - shoulder)
        deep = np.flatnonzero(oriented <= -1.0)
        if not 1 <= len(deep) <= max(3, math.ceil(height*.25)):
            continue
        if deep[0] < height*.35 or deep[-1] > height*.65:
            continue
        top, bottom = float(np.max(oriented[:band])), float(np.max(oriented[-band:]))
        if min(top, bottom) < .75:
            continue
        point = float(-np.min(oriented))
        # A one-pixel rounding spur on an otherwise smooth parenthesis is not
        # an independent brace point. The central excursion must be at least
        # as pronounced as the smaller endpoint-to-shoulder excursion.
        if point < min(top, bottom):
            continue
        # Endpoints, shoulders and the central point must join with no extra
        # oscillations; narrow random zigzags cannot supply brace evidence.
        midpoint = height//2
        if np.max(np.diff(oriented[:midpoint+1])) > .5 or np.min(np.diff(oriented[midpoint:])) < -.5:
            continue
        return {'kind': 'brace', 'direction': direction,
                'measurements': {'central_point_displacement': point,
                                 'central_point_rows': [int(row) for row in deep],
                                 'upper_shoulder_range': upper_range,
                                 'lower_shoulder_range': lower_range,
                                 'top_endpoint_displacement': top,
                                 'bottom_endpoint_displacement': bottom}}
    return None


def classify(local_mask):
    """Classify one tight binary ink component, or return ``None`` on ambiguity.

    Coordinates in measurements refer to this component's unchanged pixel
    grid. The caller owns the mask; no mutation, thresholding or resizing occurs.
    Invalid dimensions, missing rows, separate rails and dense blocks fail
    closed. Successful observations include the complete row-center profile
    and every square-stem test so cap/curve decisions can be audited in pixels.
    """
    mask = np.asarray(local_mask)
    if mask.ndim != 2 or min(mask.shape, default=0) == 0:
        return None
    height, width = mask.shape
    if height < 6 or width < 2 or width > height*.55:
        return None
    mask = mask != 0
    if float(np.mean(mask)) > .78 or not np.all(mask.any(axis=1)):
        return None
    starts = np.argmax(mask, axis=1)
    ends = width - 1 - np.argmax(mask[:, ::-1], axis=1)
    widths = ends - starts + 1
    if not np.all(mask.sum(axis=1) == widths):
        return None
    profile = (starts + ends) / 2.0
    square, cap_tests = _square(mask, widths)
    matches = [item for item in (square, _round(profile, widths, width), _brace(profile, widths, width))
               if item is not None]
    if len(matches) != 1:
        return None
    result = matches[0]
    result['measurements'].update({'row_center_profile': [float(value) for value in profile],
                                   'row_ink_widths': [int(value) for value in widths],
                                   'right_angle_cap_tests': cap_tests})
    return result
