import numpy as np
import pandas as pd
import os
import os.path as p

dataset_path = "/mnt/data1/yl241/datasets/Pawpularity/"


def main():
    df = pd.read_csv(p.join(dataset_path, "train.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    b = df["Subject Focus"].values.reshape(-1, 1)
    c = df["Eyes"].values.reshape(-1, 1)
    d = df["Face"].values.reshape(-1, 1)
    e = df["Near"].values.reshape(-1, 1)
    f = df["Action"].values.reshape(-1, 1)
    g = df["Accessory"].values.reshape(-1, 1)
    h = df["Group"].values.reshape(-1, 1)
    i = df["Collage"].values.reshape(-1, 1)
    j = df["Human"].values.reshape(-1, 1)
    k = df["Occlusion"].values.reshape(-1, 1)
    l = df["Info"].values.reshape(-1, 1)
    m = df["Blur"].values.reshape(-1, 1)
    # print(e.shape)
    # print(f.shape)
    # combined = np.hstack((e, f))
    # print(combined.shape)
    meta = np.hstack((b, c, d, e, f, g, h, i, j, k, l, m))


if __name__ == "__main__":
    main()
