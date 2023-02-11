import pickle

import tensorflow as tf

print(tf.io.gfile.listdir("gs://rl-infra-public/multi_game_dt"))
file = "checkpoint_38274228.pkl"
file_path = f"gs://rl-infra-public/multi_game_dt/{file}"

with tf.io.gfile.GFile(file_path, "rb") as f:
    model_params, model_state = pickle.load(f)

pickle.dump((model_params, model_state), open(file, "wb"))
