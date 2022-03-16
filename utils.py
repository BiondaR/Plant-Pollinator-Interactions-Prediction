#######################
# Trabalho 2          #
# Grafos e Aplicações #
# Bionda Rozin        #
#######################

# Imports
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold as k
import matplotlib.pyplot as plt
import numpy as np

def select_columns(data_frame, column_names, drop_name):
    new_frame = data_frame[column_names]
    new_frame = new_frame.groupby(drop_name).mean().reset_index()
    new_frame.drop([drop_name], axis=1, inplace=True)
    return new_frame

def fold_split(features, labels, n_folds=10):
    # Split in folds
    kf = k(n_splits=n_folds, shuffle=False)
    res = kf.split(features, labels)
    return list(res)

def fold_eval(clf, X, Y, folds, n_folds):
    eval = np.zeros((2,n_folds))

    for fold_id in range(n_folds):
        train_index = folds[fold_id][0]  # 80% of data
        test_index = folds[fold_id][1]  # 20% of data

        train_features = np.array([X[i] for i in train_index], dtype=np.float32)
        train_labels = [Y[i] for i in train_index]

        test_features = np.array([X[i] for i in test_index], dtype=np.float32)
        test_labels = [Y[i] for i in test_index]

        mean_acc = np.zeros(10)
        mean_bl = np.zeros(10)

        clf.fit(train_features, train_labels)
        for i in range(10):
            pred_labels = clf.predict(test_features)
            mean_acc[i] = round(accuracy_score(pred_labels, test_labels)*100,2)
            mean_bl[i] = round(brier_score_loss(pred_labels, test_labels)*100,2)

        eval[0][fold_id] = np.mean(mean_acc)
        eval[1][fold_id] = np.mean(mean_bl)

    return eval

def plot_bar(values, models, title):
    plt.clf()
    plt.figure(figsize=(10, 6))
    # Plot the bar graph
    plot = plt.bar(models, values)
    
    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2.,
                1.002*height,'%d' % int(height), ha='center', va='bottom')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel("Classificadores")
    plt.ylabel("Valores")
    
    # Display the graph on the screen
    plt.savefig("Figs\\"+title+".pdf")