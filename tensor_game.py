import numpy as np
import pickle


def load_reconstruction():
    with open('reconstructed_tensors.pickle', 'rb') as handle:
        tensors_rec = pickle.load(handle)
    return tensors_rec

def load_true_tensors(subject):
    tensor_file_hr = np.load("preprocessed_data/" + subject + "tensors_hr.npz")
    tensors_hr = tensor_file_hr['tensors_hr']
    return tensors_hr

ten_rec = load_reconstruction()
ten_true = load_true_tensors("175136")

new_size = ten_rec.shape
if new_size != ten_true.shape:
    ten_true = ten_true[:new_size[0], :new_size[1], :new_size[2]]

print(ten_rec[69, 82, 109])
print(ten_true[69, 82, 109])

diff = np.subtract(ten_rec, ten_true)
max_arg = np.argmax(diff)
index = np.unravel_index(max_arg, diff.shape)

print(max_arg)
print(index)

print(ten_rec[index])
print(ten_true[index])