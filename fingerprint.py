import librosa, librosa.display
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# packages for multiprocessing
from tqdm import tqdm
import concurrent.futures
import time

import logging

# multiprocessing
process_num = 16
timeout = 360

# Audio Fingerprint Class
class AudioFingerprint:

    def __init__(self, file_path, params, plot=False):
        self.file_path = file_path
        self.params = params

        self.name = os.path.basename(file_path)

        self.compute_fingerprint(audio_path=file_path, plot=plot)

    def compute_fingerprint(self, audio_path, plot=False):
        tau = self.params["tau"]
        kappa = self.params["kappa"]
        n_target_dist = self.params["n_target_dist"]
        f_target_dist = self.params["f_target_dist"]

        # compute stft
        y, _ = librosa.load(audio_path, sr=self.params["sr"])
        S = np.abs(librosa.stft(y, n_fft=self.params["n_fft"], hop_length=self.params["n_hop"]))  # n_freq, time
        S = S[:self.params["n_freq"], :]
        n_freq, n_time = S.shape

        windows = self.get_window_size(S)

        # constellation map
        C = np.zeros_like(S.T)
        for i in np.arange(0, self.params["n_freq"]):

            for j in np.arange(n_time):

                kappa = windows[1][j]
                tau = windows[0][j]

                x_st = np.max([0, i - kappa])
                x_ed = np.min([n_freq, i + kappa + 1])

                y_st = np.max([0, j - tau])
                y_ed = np.min([n_time, j + tau + 1])

                if S[i, j] == np.max(S[x_st:x_ed, y_st:y_ed]):
                    C[j, i] = 1
        n, k = np.argwhere(C == 1).T

        if plot:
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111)
            im = ax.imshow(np.log(1 + 1 * S), origin='lower', aspect='auto', cmap='gray_r')
            ax.scatter(n, k, color='r', s=15, marker='o')

            _ = plt.xticks(np.arange(0, n_time, self.params["sr"] / self.params["n_hop"]), np.arange(0, len(y) // self.params["sr"]))
            _ = plt.yticks(np.arange(0, n_freq, 500 / self.params["sr"] * self.params["n_fft"]), np.arange(0, self.params["sr"] // 2, 500))
            _ = plt.xlabel("time (s)", fontsize=18)
            _ = plt.ylabel("frequency (Hz)",fontsize=18)

        # pair-wise indexing
        self.L = dict()
        for i in np.arange(len(k)):
            t_cur = n[i]
            f_cur = k[i]

            # define target zone
            t_st = t_cur + 1
            t_ed = np.min([n_time - 1, t_cur + n_target_dist + 1])
            f_st = np.max([0, f_cur - f_target_dist])
            f_ed = np.min([self.params["n_freq"] - 1, f_cur + f_target_dist + 1])

            # find target points
            n_target, k_target = np.argwhere(C[t_st:t_ed, f_st:f_ed] == 1).T

            # print("target zone peaks:", len(n_target))

            for j in np.arange(len(k_target)):
                t_target = n_target[j] + t_st
                f_target = k_target[j] + f_st

                # indexing
                hash_key = (f_cur, f_target, t_target - t_cur)
                time_stamp = [t_cur]
                self.L[hash_key] = self.L.get(hash_key, []) + time_stamp

    def get_length(self):
        return len(self.L)

    def search(self, FP_q, keys=None):
        if keys is None:
            keys = FP_q.L.keys()

        M = dict()
        for hash_key_q in keys:
            if hash_key_q not in self.L.keys():
                continue

            # found a hit
            # query timestamp(s)
            ns_q = FP_q.L[hash_key_q]
            ns_ref = self.L[hash_key_q]

            for n_q in ns_q:

                # get shifting times L(h) - n
                Lh_n = ns_ref - n_q
                for n in Lh_n:
                    M[n] = M.get(n, 0) + 1

        best_offset = max(M, key=M.get)
        return best_offset, M[best_offset]

    def get_window_size(self, S, context=5, window_base=np.array([[3],[20]])):
        n_freq, n_time = S.shape
        e = np.log(1+np.sqrt(np.sum(S ** 2, axis=0)))
        e = ndimage.convolve(e, weights=np.ones(context)/context)
        windows = np.matmul(window_base, np.ones((1, len(e))))
        windows[:, e >= 3.5] += np.array([[2], [10]])
        return windows.astype(int)

# Hash Table Class
class HashTable:
    def __init__(self):
        self.table = dict()

    def add(self, FP):
        name = FP.name
        for hash_key, timestamps in FP.L.items():
            self.table[hash_key] = self.table.get(hash_key, []) + [name]

    def search(self, FP):
        search_list = dict()
        for hash_key, timestamps in FP.L.items():
            hit_list = self.table.get(hash_key, [])
            for hit_name in hit_list:
                search_list[hit_name] = search_list.get(hit_name, []) + [hash_key]
        return search_list

    def __length__(self):
        return len(self.table)

    def print(self):
        logging.debug(msg=self.table)

# Fingerprint Database Class
class FingerprintDB:
    def __init__(self, database_path, fingerprint_path, params):
        self.HT = HashTable()
        self.database_path = database_path

        if os.path.exists(fingerprint_path) == False:
            os.mkdir(fingerprint_path)

        fingerprint_pkl = os.path.join(fingerprint_path, "database.pkl")
        self.params = params

        self.FP_Dict = dict()

        if os.path.exists(fingerprint_pkl):
            with open(fingerprint_pkl, 'rb') as f:
                self.HT, self.FP_Dict = pickle.load(f)
        else:
            self.build()
            with open(fingerprint_pkl, 'wb') as f:
                pickle.dump([self.HT, self.FP_Dict], f)

    def build(self):

        # build database
        database_list = os.listdir(self.database_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as pool:
            results = list(tqdm(pool.map(self.add, database_list), total=len(database_list)))

        for res in results:
            FP_ref = res
            self.HT.add(FP_ref)
            self.FP_Dict[FP_ref.name] = FP_ref

    def add(self, name_ref):
        # add one audio to database
        FP_ref = AudioFingerprint(os.path.join(self.database_path, name_ref), self.params)
        return FP_ref

    def search(self, FP_q, report=True, return_first=3):
        t = time.time()

        search_list = self.HT.search(FP_q)
        assert(len(search_list) > 0)

        scores = {}

        items = list(search_list.items())

        for item in items:
            name, keys = item
            FP_ref = self.FP_Dict[name]
            best_offset, best_score = self.search_ref(FP_ref, FP_q, keys)
            scores[name] = (best_score, best_offset)

        ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_ref = ranked[0][0]
        best_score, best_offset = ranked[0][1]

        t = time.time() - t
        if report:
            print("Query: " + FP_q.name + " Best hit: " + best_ref + " at offset " + \
                                   str(FP_q.params["n_hop"] / FP_q.params["sr"] * best_offset) + " secs.")

        # find the rank of the reference
        ranked_clean = []
        for i in np.arange(len(ranked)):
            if ranked[i][0] not in ranked_clean:
                ranked_clean.append(ranked[i][0])

        hit = np.Inf
        for i in np.arange(len(ranked_clean)):
            if ranked_clean[i] == q2ref(FP_q.name):
                hit = i
                break

        return_first = np.min([return_first, len(ranked)])

        return best_ref, hit, ranked_clean[:return_first]

    def search_ref(self, FP_ref, FP_q, keys):
        best_offset, best_score = FP_ref.search(FP_q, keys)

        return best_offset, best_score


def q2ref(query_file):
    query_name = os.path.basename(query_file)
    names = query_name.split('-')[0]
    ref_name = "{}.wav".format(names)
    return ref_name