import librosa, librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

# packages for multiprocessing
from tqdm import tqdm
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import time

import logging

# multiprocessing
process_num = 32
timeout = 360

# Audio Fingerprint Class
class AudioFingerprint:

    def __init__(self, file_path, fingerprint_path, params, save=True):
        self.file_path = file_path
        self.params = params

        self.name = os.path.basename(file_path)
        if os.path.exists(fingerprint_path) == False:
            os.mkdir(fingerprint_path)
        self.fingerprint_path = os.path.join(fingerprint_path, self.name + ".npy")

        if save:
            # check if fingerprint already saved
            if os.path.exists(self.fingerprint_path):
                logging.debug(msg="Loading " + self.fingerprint_path)
                with open(self.fingerprint_path, 'rb') as f:
                    self.L = np.load(f, allow_pickle=True).item()
            else:
                self.compute_fingerprint(audio_path=file_path, plot=False)
                logging.debug(msg="Saving " + self.fingerprint_path)
                with open(self.fingerprint_path, 'wb') as f:
                    np.save(f, self.L)
        else:
            self.compute_fingerprint(audio_path=file_path, plot=False)

    def compute_fingerprint(self, audio_path, plot=False):
        tau = self.params["tau"]
        kappa = self.params["kappa"]
        n_target_dist = self.params["n_target_dist"]
        f_target_dist = self.params["f_target_dist"]

        # compute stft
        y, _ = librosa.load(audio_path, sr=self.params["sr"])
        S = np.abs(librosa.stft(y, n_fft=self.params["n_fft"], hop_length=self.params["n_hop"]))  # n_freq, time
        S = S[:self.params["n_freq"], :]
        _, n_time = S.shape

        # constellation map
        C = np.zeros_like(S.T)
        for i in np.arange(self.params["n_freq"]):

            x_st = np.max([0, i - tau])
            x_ed = np.min([self.params["n_freq"] - 1, i + tau + 1])

            for j in np.arange(n_time):

                y_st = np.max([0, j - kappa])
                y_ed = np.min([n_time - 1, j + kappa + 1])

                if S[i, j] == np.max(S[x_st:x_ed, y_st:y_ed]):
                    C[j, i] = 1
        n, k = np.argwhere(C == 1).T

        if plot:
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(np.log(1 + 1 * S), origin='lower', aspect='auto', cmap='gray_r')
            ax.scatter(n, k, color='r', s=10, marker='o')

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


class FingerprintDB:
    def __init__(self, database_path, fingerprint_path, params):
        self.HT = HashTable()
        self.database_path = database_path
        self.fingerprint_path = fingerprint_path
        self.params = params

        self.build()

    def build(self):

        t = time.time()

        # build database
        database_list = os.listdir(self.database_path)

        with tqdm(total=len(database_list)) as pbar:
            with ProcessPool(max_workers=process_num) as pool:
                future = pool.map(self.add, database_list, timeout=timeout)
                iterator = future.result()
                while True:
                    try:
                        FP_ref = next(iterator)
                        self.HT.add(FP_ref)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logging.debug("function took longer than %d seconds" % error.args[1], iterator)
                    except Exception as error:
                        logging.debug("function raised %s" % error)
                    finally:
                        pbar.update(1)

        t = time.time() - t
        print("Building time: {} secs".format(t))

    def add(self, name_ref):
        # add one audio to database
        FP_ref = AudioFingerprint(os.path.join(self.database_path, name_ref), self.fingerprint_path, self.params)
        return FP_ref

    def search(self, FP_q, report=True):
        t = time.time()

        search_list = self.HT.search(FP_q)

        scores = {}
        offsets = {}

        items = list(search_list.items())
        fp_qs = [FP_q] * len(items)
        with tqdm(total=len(items)) as pbar:
            with ProcessPool(max_workers=process_num) as pool:
                future = pool.map(self.search_ref, items, fp_qs, timeout=timeout)
                iterator = future.result()
                while True:
                    try:
                        name, best_offset, best_score = next(iterator)
                        scores[name] = best_score
                        offsets[name] = best_offset
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logging.debug("function took longer than %d seconds" % error.args[1], iterator)
                    except Exception as error:
                        logging.debug("function raised %s" % error)
                    finally:
                        pbar.update(1)

        best_ref = max(scores, key=scores.get)

        t = time.time() - t
        if report:
            print("Time elapsed: {} secs".format(t))
            print("Query: " + FP_q.name + " Best hit: " + best_ref + " at offset " + \
                                   str(FP_q.params["n_hop"] / FP_q.params["sr"] * offsets[best_ref]) + " secs.")

        hit = 0
        if best_ref == q2ref(FP_q.name):
            hit = 1

        return best_ref, t, hit

    def search_ref(self, item, FP_q):
        name, keys = item
        FP_ref = AudioFingerprint(os.path.join(self.database_path, name), self.fingerprint_path, self.params)
        best_offset, best_score = FP_ref.search(FP_q, keys)

        return name, best_offset, best_score


def q2ref(query_file):
    query_name = os.path.basename(query_file)
    names = query_name.split('-')[0]
    ref_name = "{}.wav".format(names)
    return ref_name