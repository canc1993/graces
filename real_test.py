from GRACES import GRACES
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")

# grid search range on key hyperparameters
dropout_prob = [0.1, 0.5, 0.75]
f_correct = [0, 0.1, 0.5, 0.9]


def main(name, n_features, n_iters, n_repeats):
    np.random.seed(0) # for reproducibility
    data = scipy.io.loadmat('data/' + name)
    x = data['X'].astype(float)
    if name == 'colon' or name == 'leukemia':
        y = np.int64(data['Y'])
        y[y == -1] = 0
    else:
        y = np.int64(data['Y']) - 1
    y = y.reshape(-1)
    auc_test = np.zeros(n_iters)
    seeds = np.random.choice(range(100), n_iters, replace=False) # for reproducibility
    for iter in tqdm(range(n_iters)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=seeds[iter], stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=2/7, random_state=seeds[iter], stratify=y_train)
        auc_grid = np.zeros((len(dropout_prob), len(f_correct)))
        loss_grid = np.zeros((len(dropout_prob), len(f_correct)))
        for i in range(len(dropout_prob)):
            for j in range(len(f_correct)):
                for r in range(n_repeats):
                    slc_g = GRACES(n_features=n_features, dropout_prob=dropout_prob[i], f_correct=f_correct[j])
                    selection_g = slc_g.select(x_train, y_train)
                    x_train_red_g = x_train[:, selection_g]
                    x_val_red = x_val[:, selection_g]
                    clf_g = svm.SVC(probability=True)
                    clf_g.fit(x_train_red_g, y_train)
                    y_val_pred = clf_g.predict_proba(x_val_red)
                    auc_grid[i, j] += roc_auc_score(y_val, y_val_pred[:, 1])
                    loss_grid += -np.sum(y_val * np.log(y_val_pred[:, 1]))
        index_i, index_j = np.where(auc_grid == np.max(auc_grid))
        best_index = np.argmin(loss_grid[index_i, index_j]) # break tie based on cross-entropy loss
        best_prob, best_f_correct = dropout_prob[int(index_i[best_index])], f_correct[int(index_j[best_index])]
        for r in range(n_repeats):
            slc = GRACES(n_features=n_features, dropout_prob=best_prob, f_correct=best_f_correct)
            selection = slc.select(x_train, y_train)
            x_train_red = x_train[:, selection]
            x_test_red = x_test[:, selection]
            clf = svm.SVC(probability=True)
            clf.fit(x_train_red, y_train)
            y_test_pred = clf.predict_proba(x_test_red)
            auc_test[iter] += roc_auc_score(y_test, y_test_pred[:, 1])
    return auc_test / n_repeats


if __name__ == "__main__":
    name = 'Prostate_GE'
    max_features = 10
    n_iters = 20
    n_repeats = 3
    results = np.zeros((max_features, n_iters))
    for p in range(max_features):
        results[p, :] = main(name=name, n_features=p+1, n_iters=n_iters, n_repeats=n_repeats)
    print('average test AUROC are', np.mean(results, axis=1))
