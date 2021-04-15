import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def model_build():
    data = pd.read_csv('./result_datasets/preprocessed_data.csv')
    # spliting data to train and test data
    X = data.drop('Score', axis=1)
    Y = data.Score.values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, stratify=Y, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # # Normalising all the numerical features
    std_scaler = Normalizer()
    min_max = MinMaxScaler()

    # payment_sequential feature
    payment_sequential_train = std_scaler.fit_transform(X_train.payment_sequential.values.reshape(-1, 1))
    payment_sequential_test = std_scaler.transform(X_test.payment_sequential.values.reshape(-1, 1))

    # payment_installments feature
    payment_installments_train = std_scaler.fit_transform(X_train.payment_installments.values.reshape(-1, 1))
    payment_installments_test = std_scaler.transform(X_test.payment_installments.values.reshape(-1, 1))

    # Payment value feature
    payment_value_train = std_scaler.fit_transform(X_train.payment_value.values.reshape(-1, 1))
    payment_value_test = std_scaler.transform(X_test.payment_value.values.reshape(-1, 1))

    # price
    price_train = std_scaler.fit_transform(X_train.price.values.reshape(-1, 1))
    price_test = std_scaler.transform(X_test.price.values.reshape(-1, 1))

    # freight_value
    freight_value_train = std_scaler.fit_transform(X_train.freight_value.values.reshape(-1, 1))
    freight_value_test = std_scaler.transform(X_test.freight_value.values.reshape(-1, 1))

    # product_name_length
    product_name_length_train = std_scaler.fit_transform(X_train.product_name_length.values.reshape(-1, 1))
    product_name_length_test = std_scaler.transform(X_test.product_name_length.values.reshape(-1, 1))

    # product_description_length
    product_description_length_train = std_scaler.fit_transform(X_train.product_description_length.values.reshape(-1, 1))
    product_description_length_test = std_scaler.transform(X_test.product_description_length.values.reshape(-1, 1))

    # product_photos_qty
    product_photos_qty_train = std_scaler.fit_transform(X_train.product_photos_qty.values.reshape(-1, 1))
    product_photos_qty_test = std_scaler.transform(X_test.product_photos_qty.values.reshape(-1, 1))

    # delivery_days
    delivery_days_train = std_scaler.fit_transform(X_train.delivery_days.values.reshape(-1, 1))
    delivery_days_test = std_scaler.transform(X_test.delivery_days.values.reshape(-1, 1))

    # estimated_days
    estimated_days_train = std_scaler.fit_transform(X_train.estimated_days.values.reshape(-1, 1))
    estimated_days_test = std_scaler.transform(X_test.estimated_days.values.reshape(-1, 1))

    # ships_in
    ships_in_train = std_scaler.fit_transform(X_train.ships_in.values.reshape(-1, 1))
    ships_in_test = std_scaler.transform(X_test.ships_in.values.reshape(-1, 1))

    # seller_popularity
    seller_popularity_train = min_max.fit_transform(X_train.seller_popularity.values.reshape(-1, 1))
    seller_popularity_test = min_max.transform(X_test.seller_popularity.values.reshape(-1, 1))

    # # Normalising Categorical features

    # In[169]:


    # initialising oneHotEncoder

    onehot = CountVectorizer()
    cat = OneHotEncoder()
    # payment_type
    payment_type_train = onehot.fit_transform(X_train.payment_type.values)
    payment_type_test = onehot.transform(X_test.payment_type.values)

    # customer_state
    customer_state_train = onehot.fit_transform(X_train.customer_state.values)
    customer_state_test = onehot.transform(X_test.customer_state.values)

    # seller_state
    seller_state_train = onehot.fit_transform(X_train.seller_state.values)
    seller_state_test = onehot.transform(X_test.seller_state.values)

    # product_category_name
    product_category_name_train = onehot.fit_transform(X_train.product_category_name.values)
    product_category_name_test = onehot.transform(X_test.product_category_name.values)

    # arrival_time
    arrival_time_train = onehot.fit_transform(X_train.arrival_time.values)
    arrival_time_test = onehot.transform(X_test.arrival_time.values)

    # delivery_impression
    delivery_impression_train = onehot.fit_transform(X_train.delivery_impression.values)
    delivery_impression_test = onehot.transform(X_test.delivery_impression.values)

    # estimated_del_impression
    estimated_del_impression_train = onehot.fit_transform(X_train.estimated_del_impression.values)
    estimated_del_impression_test = onehot.transform(X_test.estimated_del_impression.values)

    # ship_impression
    ship_impression_train = onehot.fit_transform(X_train.ship_impression.values)
    ship_impression_test = onehot.transform(X_test.ship_impression.values)

    # existing_cust
    existing_cust_train = cat.fit_transform(X_train.existing_cust.values.reshape(-1, 1))
    existing_cust_test = cat.transform(X_test.existing_cust.values.reshape(-1, 1))

    # **Stacking the data**

    # stacking up all the encoded features
    X_train_vec = hstack((payment_sequential_train, payment_installments_train, payment_value_train, price_train,
                          freight_value_train, product_name_length_train, product_description_length_train,
                          product_photos_qty_train, delivery_days_train, estimated_days_train, ships_in_train,
                          payment_type_train, customer_state_train, seller_state_train, product_category_name_train,
                          arrival_time_train, delivery_impression_train, estimated_del_impression_train,
                          ship_impression_train, seller_popularity_train))

    X_test_vec = hstack((payment_sequential_test, payment_installments_test, payment_value_test, price_test,
                         freight_value_test, product_name_length_test, product_description_length_test,
                         product_photos_qty_test, delivery_days_test, estimated_days_test, ships_in_test,
                         payment_type_test, customer_state_test, seller_state_test, product_category_name_test,
                         arrival_time_test, delivery_impression_test, estimated_del_impression_test,
                         ship_impression_test, seller_popularity_test))

    print(X_train_vec.shape, X_test_vec.shape)

    # # Naive Bayes

    # # Hyper parameter Tuning


    naive = MultinomialNB(class_prior=[0.5, 0.5])

    param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    # for the bow based model
    NB = GridSearchCV(naive, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
    NB.fit(X_train_vec, y_train)


    NB.best_params_

    # # Fitting the Model


    clf = MultinomialNB(alpha=0.0001, class_prior=[0.5, 0.5])
    clf.fit(X_train_vec, y_train)

    # predicted value of y probabilities
    y_pred_train = clf.predict_proba(X_train_vec)
    y_pred_test = clf.predict_proba(X_test_vec)

    # predicted values of Y labels
    pred_label_train = clf.predict(X_train_vec)
    pred_label_test = clf.predict(X_test_vec)

    # Confusion Matrix
    cf_matrix_train = confusion_matrix(y_train, pred_label_train)
    cf_matrix_test = confusion_matrix(y_test, pred_label_test)

    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

    train_auc = round(auc(fpr_train, tpr_train), 3)
    test_auc = round(auc(fpr_test, tpr_test), 3)

    plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('Best AUC for the model is {} '.format(test_auc))


    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()


    # f1 score
    print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
    print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))


    print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
    print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

    # # Observations
    #
    # 1. Naive bayes performed pretty decent in terms of minimal overfitting in train and test performances.
    # 2. Both train and test f1 score was 0.86 and accuracy 77%.
    # 3. But the confusion matrix says it has misclassified many points as False Positives.
    # 4. AUC score for test data was 0.694.

    # # Logistic Regression

    # # Hyper parameter Tuning

    # we have used max_iter 1000 as it was causing exception while fitting
    Logi = LogisticRegression(max_iter=1000, solver='lbfgs')

    param = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30]}

    # for the bow based model
    LR = GridSearchCV(Logi, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
    LR.fit(X_train_vec, y_train)


    LR.best_params_

    # **NOTE**
    #
    # * For performance measurement we will not use accuracy as a metric as the data set is highly imbalanced.
    # * We will use AUC score and f1 score as performance metric.

    # model
    clf = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
    clf.fit(X_train_vec, y_train)

    # In[180]:


    # predicted value of y probabilities
    y_pred_train = clf.predict_proba(X_train_vec)
    y_pred_test = clf.predict_proba(X_test_vec)

    # predicted values of Y labels
    pred_label_train = clf.predict(X_train_vec)
    pred_label_test = clf.predict(X_test_vec)

    # Confusion Matrix
    cf_matrix_train = confusion_matrix(y_train, pred_label_train)
    cf_matrix_test = confusion_matrix(y_test, pred_label_test)

    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

    train_auc = round(auc(fpr_train, tpr_train), 3)
    test_auc = round(auc(fpr_test, tpr_test), 3)

    plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('Best AUC for the model is {} '.format(test_auc))

    # In[181]:


    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()

    # In[182]:


    # f1 score
    print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
    print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

    # In[183]:


    print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
    print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

    # # Observations
    #
    # 1. Logistic regression performs considerably better than Naive bayes in terms of f1 score, however AUC score being almost the same.
    # 2. Misclassification of False positives reduced which resulted in the increase of f1 score of 92%.
    # 3. Accuracy was 86% for both train and test which shows the model doesn't overfit at all.

    # # Decision Tree

    # # HyperParmater tuning

    # In[184]:


    # model initialize
    DT = DecisionTreeClassifier(class_weight='balanced')

    # hyper parameters
    param = {'max_depth': [1, 5, 10, 15, 20], 'min_samples_split': [5, 10, 100, 300, 500, 1000]}

    # Grid search CV
    DT = GridSearchCV(DT, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
    DT.fit(X_train_vec, y_train)

    # In[185]:


    # best params
    DT.best_params_

    # In[186]:


    # model
    clf = DecisionTreeClassifier(class_weight='balanced', max_depth=20, min_samples_split=300)
    clf.fit(X_train_vec, y_train)

    # predicted value of y probabilities
    y_pred_train = clf.predict_proba(X_train_vec)
    y_pred_test = clf.predict_proba(X_test_vec)

    # predicted values of Y labels
    pred_label_train = clf.predict(X_train_vec)
    pred_label_test = clf.predict(X_test_vec)

    # Confusion Matrix
    cf_matrix_train = confusion_matrix(y_train, pred_label_train)
    cf_matrix_test = confusion_matrix(y_test, pred_label_test)

    # taking the probabilit scores instead of the predicted label
    # predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

    train_auc = round(auc(fpr_train, tpr_train), 3)
    test_auc = round(auc(fpr_test, tpr_test), 3)

    plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('Best AUC for the model is {} '.format(test_auc))

    # In[187]:


    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()

    # In[188]:


    # f1 score
    print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
    print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

    # In[189]:


    print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
    print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

    # # Observations
    #
    # 1. Decision Tree does nothing better interms of both f1 score , auc score and accuracy comes out to be 0.708 and 70%.
    # 2. It misclassfied False Positives to a lot.
    # 3. Model doesn't overfit but doesn't perform better either.

    # # Random Forest

    # # Hyperparameter Tuning

    # In[190]:


    # param grid
    # we have limit max_depth to 10 so that the model doesn't overfit
    param = {'min_samples_split': [5, 10, 30, 50, 100], 'max_depth': [5, 7, 10]}

    # Random forest classifier
    RFclf = RandomForestClassifier(class_weight='balanced')

    # using grid search cv to tune parameters
    RF = GridSearchCV(RFclf, param, cv=5, refit=False, n_jobs=-1, verbose=1, return_train_score=True, scoring='roc_auc')
    RF.fit(X_train_vec, y_train)

    # In[191]:


    RF.best_params_

    # In[192]:


    # model
    clf = RandomForestClassifier(class_weight='balanced', max_depth=10, min_samples_split=5)
    clf.fit(X_train_vec, y_train)

    # predicted value of y probabilities
    y_pred_train = clf.predict_proba(X_train_vec)
    y_pred_test = clf.predict_proba(X_test_vec)

    # predicted values of Y labels
    pred_label_train = clf.predict(X_train_vec)
    pred_label_test = clf.predict(X_test_vec)

    # Confusion Matrix
    cf_matrix_train = confusion_matrix(y_train, pred_label_train)
    cf_matrix_test = confusion_matrix(y_test, pred_label_test)

    # taking the probabilit scores instead of the predicted label
    # predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

    train_auc = round(auc(fpr_train, tpr_train), 3)
    test_auc = round(auc(fpr_test, tpr_test), 3)

    plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('Best AUC for the model is {} '.format(test_auc))

    # In[193]:


    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()

    # In[194]:


    # f1 score
    print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
    print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

    # In[195]:


    print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
    print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

    # # Observations
    #
    # 1. Random forest performs better than logistic regression in terms of f1 score and accuracy.
    # 2. It gives an f1 score of 90.13% and doesn't seem to overfit.
    # 3. Misclassification rate is still not that great.
    # 4. AUC is score is 0.718
    # 5. Accuracy score is 83%.

    # # GBDT

    # # Hyper parameter tuning

    # In[196]:


    # param grid
    # we have limit max_depth to 8 so that the model doesn't overfit
    param = {'min_samples_split': [5, 10, 30, 50], 'max_depth': [3, 5, 7, 8]}

    GBDTclf = GradientBoostingClassifier()

    clf = GridSearchCV(RFclf, param, cv=5, refit=False, return_train_score=True, scoring='roc_auc')
    clf.fit(X_train_vec, y_train)

    # In[197]:


    # best parameters
    clf.best_params_

    # In[198]:


    import pickle

    # In[199]:


    # Model
    clf = GradientBoostingClassifier(max_depth=8, min_samples_split=5)
    clf.fit(X_train_vec, y_train)

    # save the model to disk
    Pkl_Filename = "final_model.pkl"
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(clf, file)

    # predicted value of y probabilities
    y_pred_train = clf.predict_proba(X_train_vec)
    y_pred_test = clf.predict_proba(X_test_vec)

    # predicted values of Y labels
    pred_label_train = clf.predict(X_train_vec)
    pred_label_test = clf.predict(X_test_vec)

    # Confusion Matrix
    cf_matrix_train = confusion_matrix(y_train, pred_label_train)
    cf_matrix_test = confusion_matrix(y_test, pred_label_test)

    # taking the probabilit scores instead of the predicted label
    # predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

    train_auc = round(auc(fpr_train, tpr_train), 3)
    test_auc = round(auc(fpr_test, tpr_test), 3)

    plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('Best AUC for the model is {} '.format(test_auc))

    # In[200]:


    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()

    # In[201]:


    # f1 score
    print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
    print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

    # In[202]:


    print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
    print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

    # # Observations
    #
    # 1. Gradient Boosted classifier results the best f1 score of 0.9243 and auc score of 0.745.
    # 2. Misclassification of False Positives and True negetives is also reduced to 11% also true positive rate is 83%.
    # 3. Accuracy score is 86% for test and 87% for train data.
    # 4. Model does overfit a slight comapred to rest of the models.

    # # Observations
    #
    # 1. We created a standard deep Neural network model and trained it for 20 epochs this resulted f1 score very similar to our best ML model yet which is GBDT.
    # 2. Kindly note that this neural network was very little hyper-parameter tuning done,and still results in a very decent performance.
    # 3. However the auc score of GBDT is still better than the NN model.
    # 4. Important thing to note that NN based models can be much better than conventional ML models for such problems.

    # # Results

    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Model", "F1_score", " AUC_score ", " Accuracy "]

    table.add_row(["Naive Bayes", '0.8575', '0.694', '0.7689'])
    table.add_row(["Logistic Regression", '0.9217', '0.699', '0.8605'])
    table.add_row(["Decision Tree", '0.8031', '0.713', '0.7021'])
    table.add_row(["Random Forest", '0.9013', '0.718', '0.8315', ])
    table.add_row(["GBDT**(BEST)", '0.9243', '0.745', '0.8651'])
    # table.add_row(["Deep NN",'0.9233','0.710','0.8629'])

    print(table)
    return
