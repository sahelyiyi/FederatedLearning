import numpy as np
from collections import defaultdict, Counter
from math import sqrt


def load_data(data_path='regression_lasso/reg_3d_road/3D_spatial_network.txt'):
    with open(data_path, 'r') as f:
        data = f.read().split('\n')
    data = data[:-1]
    data = data[:500]
    fixed_data = []
    for item in data:
        item = item.split(',')
        item0, item1, item2, item3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
        fixed_data.append([item0, item1, item2, item3])
    return fixed_data


def get_graph_data(raw_data):
    data = []
    for item in raw_data:
        data.append(((item[1], item[2]), item[3]))

    data = list(set(data))

    E = 0
    MAX_DIST = 0.05
    neighbours = defaultdict(list)
    degrees = Counter()
    for i in range(len(data)):
        # if i % 1000 == 0:
        #     print (i)
        lat1, long1 = data[i][0]
        for j in range(i + 1, len(data)):
            lat2, long2 = data[j][0]
            dist = sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)
            if dist >= MAX_DIST:
                continue
            if dist == 0:
                continue
            if ((lat2, long2), dist) in neighbours[(lat1, long1)]:
                continue
            dist *= 100
            neighbours[(lat1, long1)].append(((lat2, long2), 1/dist))
            degrees[(lat1, long1)] += 1
            degrees[(lat2, long2)] += 1
            E += 1

    for item in neighbours:
        neighbours[item] = sorted(neighbours[item], key=lambda x: x[1], reverse=True)

    E = 0
    degrees = Counter()
    for item1, _ in data:
        neighbours[item1] = neighbours[item1][:10]
        for item2, dist in neighbours[item1]:
            degrees[item1] += 1
            degrees[item2] += 1
            E += 1

    cnt = 0
    node_indices = {}
    for item, _ in data:
        lat, log = item[0], item[1]
        if degrees[(lat, log)] == 0:
            continue
        if (lat, log) in node_indices:
            continue
        node_indices[(lat, log)] = cnt
        cnt += 1

    N = len(node_indices)
    X = np.zeros((N, 1, 2))
    Y = np.zeros((N, 1))
    for i, item in enumerate(data):
        lat, log = item[0]
        if (lat, log) not in node_indices:
            continue

        idx = node_indices[(lat, log)]
        X[idx] = np.array([lat, log]).T
        Y[idx] = np.array([item[1]])

    m, n = X[0].shape

    B = np.zeros((E, N))
    weight_vec = np.zeros(E)
    cnt = 0
    for item1 in neighbours:
        if item1 not in node_indices:
            continue
        idx1 = node_indices[item1]
        for item2, dist in neighbours[item1]:
            if item2 not in node_indices:
                continue
            idx2 = node_indices[item2]
            if idx1 < idx2:
                B[cnt, idx1] = 1
                # D[cnt, idx1] = dist

                B[cnt, idx2] = -1
                # D[cnt, idx2] = -dist
            else:
                B[cnt, idx1] = -1
                # D[cnt, idx1] = -dist

                B[cnt, idx2] = 1
                # D[cnt, idx2] = dist
            weight_vec[cnt] = dist
            cnt += 1

    B = B[:cnt, :]
    weight_vec = weight_vec[:cnt]

    return B, weight_vec, Y, X

