import scipy.io as sio
from numpy.random import permutation
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score


def to_onehot(label):
    m = len(set(label))
    n = len(label)
    onehot_matrix = np.zeros([n, m])
    for i in range(n):
        onehot_matrix[i, label[i]] = 1
    return onehot_matrix


def load_USPS_data_instace(path, sample_per_class):
    data = sio.loadmat(path)
    img = data['data']
    img_data = []
    label = []
    for i in range(img.shape[-1]):
        temp_sample_list = []
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j, i], [16, 16])
            temp_sample_list.append(temp)

        idx = permutation(len(temp_sample_list))
        idx = idx[:sample_per_class]
        selected_samples = [temp_sample_list[x] for x in idx]
        selected_labels = [i for _ in idx]
        img_data.extend(selected_samples)
        label.extend(selected_labels)

    img_data = np.array(img_data)
    img_data = img_data.astype('float')
    label = np.array(label[:])
    lb = np.array(label[:])
    label = to_onehot(label)
    return img_data, label, lb


if __name__ == '__main__':
    img, label, lb = load_USPS_data_instace("USPSdata/usps_all.mat", 100)
    img = np.reshape(img, [img.shape[0], -1])

    A = kneighbors_graph(img, 10, mode='distance')
    b = A.toarray()

    W = []
    Su = 0
    for item in b:
        Su += sum(item)
    avg = Su / 30000
    phi = 2 * avg ** 2
    for item in b:
        W1 = []
        for it in item:
            if it == 0:
                W1.append(0)
            else:
                W1.append(np.exp((it ** 2) / phi))
        W.append(W1)

    D = 10 * np.identity(1000)
    L = D - W

    sz_arr = [100, 300, 400, 600, 800, 900]
    for sz in sz_arr:
        x_values = np.random.randint(1000, size=sz)
        S = []
        y = []
        for item in x_values:
            n = np.zeros(1000)
            n[item] = 1
            y.append(label[item])
            S.append(n)
        S = np.array(S)
        y = np.array(y)
        j = np.matmul(S, np.linalg.inv(L))
        v = np.matmul(j, S.T)
        st1 = np.linalg.pinv(v)

        st2 = np.matmul(np.matmul(np.linalg.inv(L), S.T), st1)

        x = np.matmul(st2, y)

        pred = []
        for item in x:
            pred.append(np.argmax(item))

        print("acc for ", sz, " is :", accuracy_score(lb, pred))
