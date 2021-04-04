import os, logging, time
from fingerprint import FingerprintDB, AudioFingerprint
logging.basicConfig(level=logging.WARNING)

# paths
# database_path = "../fp_data_dummy/db/" # "../fp_data/database_recordings/"
# query_path = "../fp_data_dummy/query/" # "../fp_data/query_recordings/"
# fingerprint_path = "./fingerprints_dummy/"

database_path = "../fp_data/database_recordings/"
query_parent = "../fp_data/query_recordings/"
query_path = query_parent
fingerprint_path = "./fingerprints/"

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


def run(params):
    t = time.time()

    FDB = FingerprintDB(database_path, fingerprint_path, params)

    t = time.time() - t
    print("Building time: {} secs".format(t))

    query_list = os.listdir(query_path)
    # query_list = query_list[:10]
    # query_list = ["classical.00055-snippet-10-0.wav", "classical.00055-snippet-10-10.wav", "classical.00055-snippet-10-20.wav",
    #               "classical.00039-snippet-10-0.wav", "classical.00039-snippet-10-10.wav", "classical.00039-snippet-10-20.wav",
    #               "classical.00064-snippet-10-20.wav", "classical.00079-snippet-10-0.wav"]

    t_cum = 0
    hit_cum = 0

    with open("./output/output-{}-{}.txt".format(params["noise_type"], params["snr"]), "w") as f:
        for query_file in query_list:
            t = time.time()

            FP_q = AudioFingerprint(os.path.join(query_path, query_file), params)

            best_ref, hit, ranked = FDB.search(FP_q, report=True)

            t_cum += time.time() - t
            hit_cum += hit

            f.write("{}\t{}\t{}\t{}\n".format(query_file, ranked[0][0], ranked[1][0], ranked[2][0]))

    acc = hit_cum / len(query_list)
    avg_t = t_cum / len(query_list)

    print("Acc: {} Time: {}".format(acc, avg_t))
    return acc, avg_t

if __name__ == "__main__":

    params = {
        "sr": sr,
        "n_fft": n_fft,
        "n_hop": n_hop,
        "n_freq": n_freq,
        "tau": tau,
        "kappa": kappa,
        "n_target_dist": n_target_dist,
        "f_target_dist": f_target_dist,
        "noise_type": "given",
        "snr": "given"
    }

    run(params)

    # for noise_type in ["white", "scene"]:
    #     for snr in [-5, -10, -15]:
    #
    #         params["noise_type"] = noise_type
    #         params["snr"] = snr
    #
    #         query_path = os.path.join(query_parent, "noise={}/snr={}/".format(noise_type, snr))
    #
    #         run(params)

