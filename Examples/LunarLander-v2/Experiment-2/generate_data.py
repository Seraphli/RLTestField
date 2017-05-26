import pickle

with open('memory.pkl', 'rb') as f:
    memory = pickle.load(f)

train_data = []
hist_len = 4
for seq in memory:
    state = []
    seq = seq[1]
    for i in range(len(seq)):
        s, a, r, t = seq[i]
        if i < hist_len - 1:
            state.append(s)
            continue
        state.append(s)
        train_data.append((state, a, r, t))
        del state[0]
with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
