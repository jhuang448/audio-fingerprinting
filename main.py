import os, logging
from fingerprint import FingerprintDB, AudioFingerprint
logging.basicConfig(level=logging.WARNING)

# paths
database_path = "../fp_data/database_recordings/"
query_path = "../fp_data/query_recordings/"
fingerprint_path = "./fingerprints/"

# params
sr = 8000
n_fft = 2048
n_hop = 1024
n_freq = 256

# constellation map
tau = 3
kappa = 8

# target zone
n_target_dist = 25
f_target_dist = 30


def run(params):

    FDB = FingerprintDB(database_path, fingerprint_path, params)

    query_list = os.listdir(query_path)
    query_list = [query_list[0]]

    t_cum = 0
    hit_cum = 0

    for query_file in query_list:

        FP_q = AudioFingerprint(os.path.join(query_path, query_file), fingerprint_path, params, save=False)
        best_ref, t, hit = FDB.search(FP_q, report=False)

        t_cum += t
        hit_cum += hit

    acc = hit_cum / len(query_list)
    avg_t = t_cum / len(query_list)

    print("Acc: {} Time: {}".format(acc, avg_t))

if __name__ == "__main__":
    params = {
        "sr": sr,
        "n_fft": n_fft,
        "n_hop": n_hop,
        "n_freq": n_freq,
        "tau": tau,
        "kappa": kappa,
        "n_target_dist": n_target_dist,
        "f_target_dist": f_target_dist
    }

    run(params)

