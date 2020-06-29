from data.loader import get_fold, get_fold_idx


n_fold = 5
fold_idx = get_fold_idx(n_fold)

train_data, test_data = get_fold(fold_idx)

for fold_index in range(n_fold):
    fold_train_data, fold_dev_data = get_fold(fold_idx, fold_index)

    x_emb, x_pssm, x_tfidf, x_cp, y = fold_train_data