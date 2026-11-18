"""Readable names of the measures in the statistics tables, shared by the reports."""
import re


def name(m):
    pre = ""
    if m.startswith("p10 "): pre, m = "10th percentile of ", m[4:]
    if m.startswith("p90 "): pre, m = "90th percentile of ", m[4:]
    fixed = {"area_um2": "bundle area (µm²)", "perimeter_um": "bundle perimeter (µm)", "solidity": "bundle solidity", "eccentricity": "bundle eccentricity",
             "major_um": "bundle long axis (µm)", "minor_um": "bundle short axis (µm)", "skeleton_um": "bundle skeleton length (µm)", "branch_points": "bundle branch points",
             "thickness_um": "bundle thickness (µm)", "thickness_cv": "bundle thickness variation (CV)", "pieces": "bright pieces per bundle", "largest_piece_share": "largest bright piece share",
             "tip_height_um": "tip height (µm)", "tip_height_sd_um": "tip height spread across the bundle (µm)", "top_p90_um": "90th percentile top height (µm)", "volume_um3": "bundle volume (µm³)",
             "intensity_cv": "phalloidin variation within the bundle (CV)", "intensity_skew": "phalloidin skewness", "intensity_kurtosis": "phalloidin kurtosis", "histogram_entropy": "phalloidin histogram entropy",
             "bundle_to_surface": "bundle brightness relative to the cell surface", "lbp_uniform_share": "local binary pattern uniform share", "lbp_entropy": "local binary pattern entropy",
             "coherence": "stereocilia orientation coherence",
             "surface_area_um2": "apical surface area (µm²)", "surface_perimeter_um": "apical surface perimeter (µm)", "surface_eccentricity": "apical surface eccentricity",
             "surface_solidity": "apical surface solidity", "surface_elongation": "apical surface elongation (long/short axis)", "surface_axis_vs_row_deg": "apical surface long axis to the row (°)",
             "surface_area_cv": "apical surface area variation (CV)", "nearest_cell_um": "distance to the nearest hair cell (µm)", "gaps_per_100um": "gaps per 100 µm",
             "spacing_cv": "spacing variation along the row (CV)", "polarity_abs_dev_deg": "bundle direction from the row normal (°, median)",
             "polarity_share_off30": "share of bundles more than 30° off", "polarity_signed_mean_deg": "mean bundle rotation (°)", "polarity_circ_sd_deg": "bundle direction spread (circular SD, °)",
             "polarity_neighbor_diff_deg": "direction difference to the next bundle (°)", "nucleus_area_um2": "nucleus area (µm²)", "nucleus_volume_um3": "nucleus volume (µm³)",
             "nucleus_depth_um": "nucleus depth below the surface (µm)", "front view width_along_um": "bundle width along the row (µm)", "front view depth_across_um": "bundle depth across the row (µm)",
             "front view volume_um3": "bundle volume, front view (µm³)", "front view length_um": "bundle length base to tip (µm)", "front view share_lying_flat": "share of bundles lying flat",
             "front view surface_um2": "bundle surface area, front view (µm²)", "front view compactness": "bundle compactness, front view (1 = sphere)",
             "front view solidity_3d": "bundle 3D solidity, front view (volume / convex hull)", "front view thickness_um": "bundle thickness, front view (µm)",
             "ihc_bundle_height_standing_um": "bundle height, standing bundles (µm)", "ihc_bundle_lean_standing_deg": "bundle lean, standing bundles (°)",
             "org_score": "organization score (1 orderly to 3 disorganized)", "disorganized_share": "share scored disorganized", "per_100um": "cells per 100 µm",
             "outer_hair_cells_per_100um": "outer hair cells per 100 µm", "ihc_per_100um": "inner hair cells per 100 µm",
             "bundle height, standing bundles pooled per mouse (um)": "bundle height, standing bundles pooled per mouse (µm)",
             "bundle height, hand clicks pooled per mouse (um)": "bundle height, hand clicks pooled per mouse (µm)",
             "regularity_index": "spacing regularity (mean spacing / SD)", "gaps": "gaps in the row (per field)", "doubles": "spacings under half the local median (per field)",
             "ohc_interior_share_6_neighbors": "share of interior outer hair cells with 6 neighbors", "row_distance_IHC_OHC1_um": "distance inner row to outer row 1 (µm)",
             "row_distance_OHC1_OHC2_um": "distance outer row 1 to row 2 (µm)", "row_distance_OHC2_OHC3_um": "distance outer row 2 to row 3 (µm)",
             "stagger_OHC2_vs_OHC1": "stagger of row 2 against row 1 (fraction of spacing)", "stagger_OHC3_vs_OHC2": "stagger of row 3 against row 2 (fraction of spacing)",
             "notch_R": "notch direction alignment (mean resultant length)", "notch_circ_sd_deg": "notch direction spread (circular SD, °)",
             "notch_drift_deg_per_100um": "notch direction drift along the row (° per 100 µm)", "notch_share_pointing_inward": "share of notches pointing inward",
             "axis_R": "bundle axis alignment (mean resultant length)", "axis_circ_sd_deg": "bundle axis spread (circular SD, °)",
             "local_misorient_median_deg": "local misorientation against neighbors (°, median)", "orientation_corr_length_um": "orientation correlation length (µm)",
             "flag_misoriented_share": "share of cells misoriented (over 45°)", "flag_misoriented_ratio": "clustering of misoriented cells (observed/expected neighbor pairs)",
             "flag_shape_share": "share of bundles with doubtful shape", "flag_shape_ratio": "clustering of doubtful bundle shapes (observed/expected neighbor pairs)",
             "flag_no_bundle_share": "share of cells with no bundle found", "flag_no_bundle_ratio": "clustering of cells with no bundle (observed/expected neighbor pairs)",
             "flag_any_share": "share of cells with any flag", "flag_any_ratio": "clustering of flagged cells (observed/expected neighbor pairs)"}
    if m in fixed: return pre + fixed[m]
    g = re.match(r"(strip_)?glcm_(\w+)_d(\d)", m)
    if g:
        where = " along the bundle skeleton" if g.group(1) else ""
        return pre + f"GLCM {g.group(2)}{where}, {int(g.group(3)) * 0.11:.2f} µm"
    return pre + m


NEW = {"row_line_deviation_um": "distance of cells from the row line (µm, median)", "row_line_deviation_p90_um": "distance of cells from the row line (µm, 90th percentile)",
       "share_off_line": "share of cells more than 2 µm off the row line", "share_out_of_order": "share of cells closer to another row",
       "row_zigzag_um": "zigzag along the row (µm)", "front view surface_um2": "bundle surface area, front view (µm²)",
       "front view compactness": "bundle compactness, front view (1 = sphere)", "front view solidity_3d": "bundle 3D solidity, front view (volume / convex hull)",
       "front view thickness_um": "bundle thickness, front view (µm)", "nucleus volume_um3": "nucleus volume (µm³)", "nucleus equivalent_diameter_um": "nucleus diameter (µm)",
       "nucleus elongation": "nucleus elongation (long / short axis)", "nucleus sphericity": "nucleus sphericity (1 = sphere)",
       "nucleus dapi_total_rel": "nucleus DAPI content (relative to the field's nuclei)", "nucleus dapi_mean_rel": "nucleus DAPI brightness (relative to the field's nuclei)",
       "nucleus dapi_cv": "DAPI variation within the nucleus (CV)", "nucleus chromocenter_count": "chromocenters per nucleus",
       "nucleus chromocenter_share": "share of the nucleus in chromocenters", "nucleus chromatin_glcm_contrast": "chromatin GLCM contrast at 0.22 µm",
       "nucleus chromatin_glcm_homogeneity": "chromatin GLCM homogeneity at 0.22 µm", "nucleus depth_um": "nucleus depth below the surface (µm)",
       "nucleus_matched_share": "share of cells with a matched nucleus", "inner_like_outer_cells_per_100um": "inner-like cells in the outer rows per 100 µm",
       "inner_like_outer_share": "share of outer-row cells that are inner-like", "outer_like_inner_share": "share of inner-row cells that are outer-like",
       "ihc_over_ohc_calb2": "Calb2 of inner over outer hair cells"}


def readable(m):
    """Readable name of a measure in the statistics tables."""
    base = m[4:] if m[:4] in ("p10 ", "p90 ") else m
    if base in NEW: return {"p10 ": "10th percentile of ", "p90 ": "90th percentile of "}.get(m[:4], "") + NEW[base]
    return name(m)
