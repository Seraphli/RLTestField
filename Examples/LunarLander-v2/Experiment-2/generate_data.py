import pickle, numpy as np

with open('memory.pkl', 'rb') as f:
    memory = pickle.load(f)

train_data = []
hist_len = 4
for seq in memory:
    hist = []
    seq = seq[1]
    for i in range(len(seq)):
        s, a, r, t = seq[i]
        if i < hist_len:
            hist.append(s)
            if i == hist_len - 1:
                state = np.array(hist).flatten()
            continue
        del hist[0]
        hist.append(s)
        state_ = np.array(hist).flatten()
        train_data.append((state, a, r, t, state_))
        state = state_

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
