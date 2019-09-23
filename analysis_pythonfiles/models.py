import pandas as pd
import numpy as np
import utils
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from tensorflow import keras


def simple_ann_network(data, y, n_neurons, nepochs, lr):
    X, shape = utils.extract_values(data)
    N, P = shape
    X, Y = X.astype(np.float32), y[:, None].astype(np.float32)
    XY = np.hstack((X, Y))
    np.random.shuffle(XY)
    X, Y = XY[:, :-1], XY[:, -1]
    if len(Y.shape) == 1:
        Y = Y[:, None]
    # data placeholders
    tf_X = tf.placeholder(tf.float32, [None, P])
    tf_Y = tf.placeholder(tf.float32, [None, Y.shape[1]])
    # network layers
    hl0 = tf.layers.dense(tf_X, n_neurons, activation=tf.nn.relu)
    hl1 = tf.layers.dense(hl0, Y.shape[1], activation=None)
    # for computing cost
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_Y, logits=hl1)
    cost = tf.reduce_mean(cross_ent)
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    pred_label = tf.nn.sigmoid(hl1)
    comp_label = tf.equal(tf.round(pred_label), tf_Y)
    acc = tf.reduce_mean(tf.cast(comp_label, tf.float32))
    # get the training and testing indices
    idxs, n_splits = utils.create_train_test_indices(X)
    train_acc = np.zeros((n_splits, nepochs))
    test_acc, test_pred = np.zeros(n_splits), np.zeros(n_splits)
   
    with tf.Session() as ses:
        for cv_iter in range(n_splits):
            tf.global_variables_initializer().run()
            X_train = X[idxs["train"][cv_iter]]
            Y_train = Y[idxs["train"][cv_iter]]
            training_acc = np.zeros(nepochs)
            
            for ep_iter in range(nepochs):
                ses.run(
                    opt, 
                    feed_dict={tf_X:X_train, tf_Y:Y_train}
                )
                loss, _, accur = ses.run(
                    [cost, opt, acc], 
                    feed_dict={tf_X:X_train, tf_Y:Y_train}
                )
                training_acc[ep_iter] = accur

            train_acc[cv_iter, :] = training_acc
            X_test = X[idxs["test"][cv_iter]]
            Y_test = Y[idxs["test"][cv_iter]]
            accur_test, pred_test = ses.run(
                [acc, tf.round(pred_label)],
                feed_dict={tf_X:X_test, tf_Y:Y_test}
            )
            test_acc[cv_iter] = accur_test
            test_pred[cv_iter] = pred_test

    return train_acc, test_acc, test_pred, Y.ravel()


def rf_feature_selector(data, y, thresh, cv, clf, max_feat=10):
    
    X, N, P = utils.extract_values(data)
    results = {"pred":np.zeros(N), "imp":[], "imp_idx":[]}
    results_null = {"pred":np.zeros(N), "imp":[], "imp_idx":[]}

    for idx, (train, test) in enumerate(cv.split(X, y)):
        sfm = SelectFromModel(clf, threshold=thresh)
        sfm.fit(X[train], y[train])
        X_transform = sfm.transform(X[train])
        n_features = X_transform.shape[1]
        
        while n_features > max_feat:
            sfm.threshold += 0.01
            X_transform = sfm.transform(X[train])
            n_features = X_transform.shape[1]

        clf.fit(X_transform, y[train])
        results["pred"][idx] = clf.predict(sfm.transform(X[test]))
        results["imp_idx"].append(sfm.get_support())
        results["imp"].append(clf.feature_importances_)

        try:
            y_shuff = np.copy(y[train])
            np.random.shuffle(y_shuff)
            clf.fit(X_transform, y_shuff)
            results_null["pred"][idx] = clf.predict(sfm.transform(X[test]))
            results_null["imp_idx"].append(sfm.get_support())
            results_null["imp"].append(clf.feature_importances_)
        except:
            print("couldn't compute null model iteration: {}".format(idx))


    return (results, r2_score(y, results["pred"]),
            results_null, r2_score(y, results_null["pred"]))
