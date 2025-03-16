import numpy as np


def ransac(world_points, bearing_vectors, solver, max_iterations, inlier_dist_threshold, probability):
    iterations = 0
    best_inliers_count = 0
    k = 1.0
    skipped_count = 0
    max_skip = max_iterations * 10

    indices = list(range(world_points.shape[1]))
    sampled_indices = list(np.random.choice(indices, 4))
    best_model = None

    while iterations < k and iterations < max_iterations and skipped_count < max_skip:
        p_w = world_points[:, sampled_indices[:3]]
        p_i = bearing_vectors[:, sampled_indices[:3]]

        models = solver(p_w, p_i)
        if not models:
            skipped_count += 1
            continue

        selected_model_idx = 0
        min_cost = float('inf')
        pt_idx = sampled_indices[3]
        for i, model in enumerate(models):
            cost = model.reprojection_dist_of(world_points[:, pt_idx], bearing_vectors[:, pt_idx])
            if cost < min_cost:
                selected_model_idx = i
                min_cost = cost

        dists_to_all = models[selected_model_idx].reprojection_dists_of(world_points, bearing_vectors)
        inlier_count = np.sum(dists_to_all < inlier_dist_threshold)

        if inlier_count > best_inliers_count:
            best_inliers_count = inlier_count
            best_model = models[selected_model_idx]

            w = best_inliers_count / world_points.shape[1]
            p_no_outliers = max(np.finfo(float).eps, min(1.0 - np.finfo(float).eps, 1.0 - w**len(sampled_indices)))
            k = np.log(1.0 - probability) / np.log(p_no_outliers)

        sampled_indices = list(np.random.choice(indices, 4))
        iterations += 1

    return best_model

    