"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    #print(faces.shape)
    #print(vertices.shape)

    # v1_indices = faces[:, 0]
    # v1 = vertices[v1_indices]

    vertices = vertices.squeeze()
    vec1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    vec2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cross_product = np.cross(vec1, vec2)

    print(cross_product.shape)
    triangle_areas = np.linalg.norm(cross_product, axis=1) * 0.5

    print(triangle_areas.shape)

    print(triangle_areas)

    print(triangle_areas.shape)
    area_probabilities = triangle_areas / np.sum(triangle_areas)

    print(area_probabilities.shape)

    chosen_faces = np.random.choice(len(faces), size=n_points, p=area_probabilities)
    sampled_points = []
    for face_idx in chosen_faces:
        v0, v1, v2 = vertices[faces[face_idx]]
        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        point = u * v0 + v * v1 + w * v2
        sampled_points.append(point)

    return np.array(sampled_points)


    # ###############
    # TODO: Implement
    #raise NotImplementedError
    # ###############
