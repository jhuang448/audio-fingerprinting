import os, logging, time
from fingerprint import FingerprintDB, AudioFingerprint
logging.basicConfig(level=logging.WARNING)

# paths
# database_path = "../fp_data_dummy/db/" # "../fp_data/database_recordings/"
# query_path = "../fp_data_dummy/query/" # "../fp_data/query_recordings/"
# fingerprint_path = "./fingerprints_dummy/"
# output_file = "./output/output_dummy.txt"

database_path = "../fp_data/database_recordings/"
query_path = "../fp_data/query_recordings/"
fingerprint_path = "./fingerprints/"
output_file = "./output/output.txt"

# params
sr = 8000
n_fft = 2048
n_hop = 512
n_freq = 1025

# constellation map
tau = 3
kappa = 20

# target zone
n_target_dist = 50
f_target_dist = 100

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

def fingerprintBuilder(database_path, fingerprint_path, params=params):
    t = time.time()
    FDB = FingerprintDB(database_path, fingerprint_path, params)
    t = time.time() - t
    print("Building time: {} secs".format(t))

def audioIdentification(query_path, fingerprint_path, output_file, params=params):
    t = time.time()

    FDB = FingerprintDB(None, fingerprint_path, params)

    t = time.time() - t
    print("Re-Building time: {} secs".format(t))

    query_list = os.listdir(query_path)
    # query_list = query_list[:10]

    t_cum = 0
    top1 = 0
    top3 = 0
    prec = 0

    with open(output_file, "w") as f:
        for query_file in query_list:
            t = time.time()

            FP_q = AudioFingerprint(os.path.join(query_path, query_file), params)

            best_ref, hit, ranked = FDB.search(FP_q, report=True)

            t_cum += time.time() - t
            prec += 1/(hit+1)
            if hit == 0:
                top1 += 1
            if hit <= 2:
                top3 += 1

            f.write("{}\t{}\t{}\t{}\n".format(query_file, ranked[0], ranked[1], ranked[2]))

    acc_top1 = top1 / len(query_list)
    acc_top3 = top3 / len(query_list)
    avg_prec = prec / len(query_list)
    avg_t = t_cum / len(query_list)

    print("Top1 acc: {} Top3 acc: {} MAP: {} Time: {}".format(acc_top1, acc_top3, avg_prec, avg_t))
    return acc_top1, acc_top3, avg_prec, avg_t

if __name__ == "__main__":

    fingerprintBuilder(database_path, fingerprint_path, params)
    audioIdentification(query_path, fingerprint_path, output_file, params)

