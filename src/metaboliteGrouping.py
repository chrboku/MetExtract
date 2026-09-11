import numpy as np

# Relative/absolute tolerances used when comparing normalized (shape-only) abundance
# profiles in zero-variance fallback cases.
NORMALIZED_PROFILE_RTOL = 1e-6
NORMALIZED_PROFILE_ATOL = 1e-9


def _coerce_abundance_profile(values):
    profile = []
    for value in values:
        try:
            numeric = float(value)
            if np.isnan(numeric):
                numeric = 0.0
        except (TypeError, ValueError):
            numeric = 0.0
        profile.append(max(0.0, numeric))
    return profile


def _pearson(values_a, values_b):
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    if np.std(values_a) == 0 or np.std(values_b) == 0:
        max_a = max(values_a)
        max_b = max(values_b)
        if max_a > 0 and max_b > 0:
            norm_a = [a / max_a for a in values_a]
            norm_b = [b / max_b for b in values_b]
            if np.allclose(norm_a, norm_b, rtol=NORMALIZED_PROFILE_RTOL, atol=NORMALIZED_PROFILE_ATOL):
                return 1.0
        return 0.0
    corr = np.corrcoef(values_a, values_b)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def _presence_aware_profile_similarity(profile_a, profile_b, both_missing_weight=0.25):
    if profile_a is None or profile_b is None:
        return 0.0
    if len(profile_a) < 2 or len(profile_b) < 2 or len(profile_a) != len(profile_b):
        return 0.0

    quantified_a = []
    quantified_b = []
    both_missing_count = 0

    for value_a, value_b in zip(profile_a, profile_b):
        has_a = value_a > 0.0
        has_b = value_b > 0.0
        if has_a and has_b:
            quantified_a.append(value_a)
            quantified_b.append(value_b)
        elif (not has_a) and (not has_b):
            both_missing_count += 1
        # samples where only one feature has abundance are omitted completely

    quantified_count = len(quantified_a)
    if quantified_count == 0 and both_missing_count == 0:
        return 0.0

    quantified_corr = _pearson(quantified_a, quantified_b)
    if quantified_corr is None:
        quantified_corr = 1.0 if quantified_count == 1 else 0.0

    weighted_sum = quantified_count * quantified_corr + both_missing_weight * both_missing_count
    total_weight = quantified_count + both_missing_weight * both_missing_count
    if total_weight <= 0:
        return 0.0

    return float(weighted_sum / total_weight)


def _connected_components(group_ids, adjacency):
    remaining = set(group_ids)
    components = []
    while remaining:
        root = remaining.pop()
        stack = [root]
        component = {root}
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _avg_similarity_to_group(node, group, similarities):
    """Return the mean similarity from a node to all other members of a group."""
    vals = []
    for other in group:
        if node == other:
            continue
        vals.append(similarities.get(node, {}).get(other, 0.0))
    if len(vals) == 0:
        return 0.0
    return float(sum(vals) / len(vals))


def _connection_rate_to_group(node, group, adjacency):
    """Return the fraction of group members that are directly connected to node."""
    if len(group) == 0:
        return 0.0
    degree = sum(1 for neighbor in adjacency.get(node, set()) if neighbor in group)
    return float(degree) / float(len(group))


def _split_component_by_dense_subclusters(component_ids, adjacency, similarities, min_connection_rate):
    """Split a connected component into dense subclusters using connection-rate cores.

    A dense core contains nodes that satisfy the minimum connection-rate criterion
    within the full component. Remaining nodes are then assigned to the best-matching
    core (if their connection rate to that core is sufficient), otherwise kept as
    separate groups.
    """
    if len(component_ids) <= 2:
        return [sorted(component_ids)]

    component_set = set(component_ids)
    min_required_connections = max(0.0, min_connection_rate) * (len(component_set) - 1)
    dense_nodes = set()
    for node in component_set:
        degree = sum(1 for neighbor in adjacency.get(node, set()) if neighbor in component_set)
        if degree >= min_required_connections:
            dense_nodes.add(node)

    # avoid over-splitting: if no node reaches the connection-rate threshold, keep this
    # component intact rather than fragmenting nearly all nodes into singleton groups
    if len(dense_nodes) == 0:
        return [sorted(component_ids)]

    dense_groups = _connected_components(dense_nodes, adjacency)
    groups = [set(group) for group in dense_groups]

    remaining = sorted(component_set - dense_nodes)
    for node in remaining:
        best_group = None
        best_score = None
        for group in groups:
            if _connection_rate_to_group(node, group, adjacency) >= min_connection_rate:
                score = _avg_similarity_to_group(node, group, similarities)
                if best_score is None or score > best_score:
                    best_group = group
                    best_score = score
        if best_group is not None:
            best_group.add(node)
        else:
            groups.append({node})

    return [sorted(group) for group in groups]


def split_group_by_relative_abundance(group_ids, abundance_vectors, min_peak_correlation, min_connection_rate):
    if len(group_ids) <= 2:
        return [group_ids]

    profiles = {}
    for feature_id in group_ids:
        if feature_id in abundance_vectors:
            profiles[feature_id] = _coerce_abundance_profile(abundance_vectors[feature_id])

    if len(profiles) < len(group_ids):
        return [group_ids]
    profile_lengths = {len(profile) for profile in profiles.values()}
    if len(profile_lengths) != 1:
        return [group_ids]

    similarities = {feature_id: {} for feature_id in group_ids}
    adjacency = {feature_id: set() for feature_id in group_ids}
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            feature_a = group_ids[i]
            feature_b = group_ids[j]
            similarities[feature_a][feature_b] = _presence_aware_profile_similarity(profiles[feature_a], profiles[feature_b])
            similarities[feature_b][feature_a] = similarities[feature_a][feature_b]
            if similarities[feature_a][feature_b] >= min_peak_correlation:
                adjacency[feature_a].add(feature_b)
                adjacency[feature_b].add(feature_a)

    components = _connected_components(group_ids, adjacency)
    refined_groups = []
    for component in components:
        refined_groups.extend(_split_component_by_dense_subclusters(sorted(component), adjacency, similarities, min_connection_rate))

    return sorted([sorted(group) for group in refined_groups], key=lambda group: (len(group), group), reverse=True)
