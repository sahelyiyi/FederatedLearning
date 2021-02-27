import numpy as np
from collections import defaultdict, Counter
from math import sqrt


def load_data(data_path='3D_spatial_network.txt'):
    with open(data_path, 'r') as f:
        data = f.read().split('\n')
    data = data[:-1]
    data = data[:200000]
    fixed_data = []
    for item in data:
        item = item.split(',')
        item0, item1, item2, item3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
        fixed_data.append([item0, item1, item2, item3])
    return fixed_data


def merge_locations(data):
    sorted_lats = sorted(data, key=lambda x: x[3])

    merged_data = []
    for i in range(int(len(sorted_lats) / 5)):
        merge_cell = []
        for j in range(5):
            merge_cell.append(sorted_lats[i * 5 + j])
        merged_data.append(merge_cell)
    merged_data = np.array(merged_data)

    return merged_data

def get_graph_data(data):
    merged_data = merge_locations(data)

    sorted_merged = sorted(merged_data, key=lambda x: (np.mean(x[:, 1]), np.mean(x[:, 2])))

    mean_merged = []
    for merge_cell in sorted_merged:
        mean_merged.append((np.mean(merge_cell[:, 1]), np.mean(merge_cell[:, 2])))

    E = 0
    MAX_DIST = 0.01
    neighbours = defaultdict(list)
    degrees = Counter()
    for i in range(len(mean_merged)):
        # if i % 1000 == 0:
        #     print (i)
        lat1, long1 = mean_merged[i][0], mean_merged[i][1]
        for j in range(i + 1, len(mean_merged)):
            lat2, long2 = mean_merged[j][0], mean_merged[j][1]
            dist = sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)
            if dist >= MAX_DIST:
                break
            if dist == 0:
                continue
            if ((lat2, long2), dist) in neighbours[(lat1, long1)]:
                continue
            neighbours[(lat1, long1)].append(((lat2, long2), MAX_DIST - dist))
            degrees[(lat1, long1)] += 1
            degrees[(lat2, long2)] += 1
            E += 1

    sorted_merged = sorted(merged_data, key=lambda x: (np.mean(x[:, 2]), np.mean(x[:, 1])))

    mean_merged = []
    for merge_cell in sorted_merged:
        mean_merged.append((np.mean(merge_cell[:, 1]), np.mean(merge_cell[:, 2])))

    for i in range(len(mean_merged)):
        lat1, long1 = mean_merged[i][0], mean_merged[i][1]
        for j in range(i + 1, len(mean_merged)):
            lat2, long2 = mean_merged[j][0], mean_merged[j][1]
            dist = sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)
            if dist >= MAX_DIST:
                break
            if dist == 0:
                continue
            if ((lat2, long2), dist) in neighbours[(lat1, long1)]:
                continue
            neighbours[(lat1, long1)].append(((lat2, long2), MAX_DIST - dist))
            degrees[(lat1, long1)] += 1
            degrees[(lat2, long2)] += 1
            E += 1

    cnt = 0
    node_indices = {}
    for item in mean_merged:
        lat, log = item[0], item[1]
        if degrees[(lat, log)] == 0:
            continue
        if (lat, log) in node_indices:
            continue
        node_indices[(lat, log)] = cnt
        cnt += 1

    N = len(node_indices)
    X = np.zeros((N, 5, 2))
    Y = np.zeros((N, 5))
    for i, item in enumerate(mean_merged):
        lat, log = item[0], item[1]
        if (lat, log) not in node_indices:
            continue

        idx = node_indices[(lat, log)]
        X[idx] = np.array([sorted_merged[i][:, 1], sorted_merged[i][:, 2]]).T
        Y[idx] = np.array([sorted_merged[i][:, 3]])

    m, n = X[0].shape

    B = np.zeros((E, N))
    weight_vec = np.zeros(E)
    cnt = 0
    for item1 in neighbours:
        idx1 = node_indices[item1]
        for item2, dist in neighbours[item1]:
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
    return B, weight_vec, Y, X
