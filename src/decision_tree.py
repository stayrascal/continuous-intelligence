from enum import Enum
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import sys, os, json
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
sys.path.append(os.path.join('..', 'src'))
sys.path.append(os.path.join('src'))

from sklearn.linear_model import ElasticNet
from sklearn import tree, ensemble, metrics
import evaluation
import tracking

class Model(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1
    ADABOOST = 2
    GRADIENT_BOOST = 3


def load_data():
    filename = "data/splitter/train.csv"
    print("Loading data from {}".format(filename))
    train = pd.read_csv(filename)

    filename = 'data/splitter/validation.csv'
    print("Loading data from {}".format(filename))
    validate = pd.read_csv(filename)

    return train, validate


def join_tables(train, validate):
    print("Joining tables for consistent encoding")
    return train.append(validate).drop('date', axis=1)


def encode_categorical_columns(df):
    obj_df = df.select_dtypes(include=['object', 'bool']).copy().fillna('-1')
    lb = LabelEncoder()
    for col in obj_df.columns:
        df[col] = lb.fit_transform(obj_df[col])
    return df


def encode(train, validate):
    print("Encoding categorical variables")
    train_ids = train.id
    validate_ids = validate.id

    joined = join_tables(train, validate)

    encoded = encode_categorical_columns(joined.fillna(-1))

    print("Not predicting returns...")
    encoded.loc[encoded.unit_sales < 0, 'unit_sales'] = 0

    validate = encoded[encoded['id'].isin(validate_ids)]
    train = encoded[encoded['id'].isin(train_ids)]
    return train, validate


def train_model(train, model=Model.DECISION_TREE, seed=None):
    print("Training model using regressor: {}".format(model.name))
    train_dropped = train.drop('unit_sales', axis=1)
    target = train['unit_sales']

    if model == Model.RANDOM_FOREST:
        params = {'n_estimators': 10}
        clf = ensemble.RandomForestRegressor(random_state=seed, **params)
    elif model == Model.ADABOOST:
        params = {'n_estimators': 50, 'learning_rate': 1.0, 'loss':'linear'}
        clf = ensemble.AdaBoostRegressor(random_state=seed, **params)
    elif model == Model.GRADIENT_BOOST:
        params = {'n_estimators': 200, 'max_depth': 4}
        clf = ensemble.GradientBoostingRegressor(random_state=seed, **params)
    else:
        params = {'criterion': 'mse'}
        clf = tree.DecisionTreeRegressor(random_state=seed)

    trained_model = clf.fit(train_dropped, target)
    return (trained_model,params)

def make_elastic_net(train, alpha, l1_ratio, random_state=None):
    print("Creating elastic net model")
    train_dropped = train.drop('unit_sales', axis=1)
    target = train['unit_sales']

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr = lr.fit(train_dropped, target)
    return lr


def make_decision_tree(train, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                    max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, presort=False):
    print("Creating decision tree model")
    train_dropped = train.drop('unit_sales', axis=1)
    target = train['unit_sales']

    # mlflow.log_param("criterion", criterion)
    # mlflow.log_param("splitter", splitter)
    # mlflow.log_param("min_samples_split", min_samples_split)
    # mlflow.log_param("min_samples_leaf", min_samples_leaf)
    # mlflow.log_param("min_weight_fraction_leaf", min_weight_fraction_leaf)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("max_features", max_features)
    # mlflow.log_param("random_state", random_state)
    # mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
    # mlflow.log_param("min_impurity_decrease", min_impurity_decrease)
    # mlflow.log_param("min_impurity_split", min_impurity_split)
    # mlflow.log_param("presort", presort)
    
    clf = tree.DecisionTreeRegressor(
        criterion=criterion, # mse, friedman_mse, mae, 
        splitter=splitter, # best, random
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features, # int, float, auto, sqrt, log2
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        presort=presort)
    clf = clf.fit(train_dropped, target)
    return clf


def overwrite_unseen_prediction_with_zero(preds, train, validate):
    cols_item_store = ['item_nbr', 'store_nbr']
    cols_to_use = validate.columns.drop('unit_sales') if 'unit_sales' in validate.columns else validate.columns
    validate_train_joined = pd.merge(validate[cols_to_use], train, on=cols_item_store, how='left')
    unseen = validate_train_joined[validate_train_joined['unit_sales'].isnull()]
    validate['preds'] = preds
    validate.loc[validate.id.isin(unseen['id_x']), 'preds'] = 0
    preds = validate['preds'].tolist()
    return preds


def make_predictions(model, validate):
    print("Making prediction on validation data")
    validate_dropped = validate.drop('unit_sales', axis=1).fillna(-1)
    validate_preds = model.predict(validate_dropped)
    return validate_preds


def write_predictions_and_score(evaluation_metrics, model, columns_used):
    key = "decision_tree"
    if not os.path.exists('data/{}'.format(key)):
        os.makedirs('data/{}'.format(key))
    filename = 'data/{}/model.pkl'.format(key)
    print("Writing to {}".format(filename))
    joblib.dump(model, filename)

    filename = 'results/metrics.json'
    print("Writing to {}".format(filename))
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(filename, 'w+') as score_file:
        json.dump(evaluation_metrics, score_file)

def write_score(validation_score, model):
    filename = 'results/score.txt'
    print("Writing to {}".format(filename))
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(filename, 'w+') as score_file:
        score_file.write(str(validation_score))
    print("Done deciding with trees")


def main(model=Model.RANDOM_FOREST, seed=None):
    original_train, original_validate = load_data()
    train, validate = encode(original_train, original_validate)
    with tracking.track() as track:
        track.set_model(model)
        model, params = train_model(train, model, seed)
        track.log_params(params)
        validation_predictions = make_predictions(model, validate)

        print("Calculating metrics")
        evaluation_metrics = {
            'nwrmsle': evaluation.nwrmsle(validation_predictions, validate['unit_sales'].values, validate['perishable'].values),
            'r2_score': metrics.r2_score(y_true=validate['unit_sales'].values, y_pred=validation_predictions)
        }
        track.log_metrics(evaluation_metrics)

        write_predictions_and_score(evaluation_metrics, model, original_train.columns)

        print("Evaluation done with metrics {}.".format(json.dumps(evaluation_metrics)))

def run(max_features=None, max_depth=None, seed=None, index=''):
    original_train, original_validate = load_data()
    train, validate = encode(original_train, original_validate)

    mlflow.set_tracking_uri(uri='http://127.0.0.1:5000')
    mlflow.set_experiment('Dev')

    with mlflow.start_run():
        model = make_decision_tree(train, max_depth=max_depth, max_features=max_features, random_state=seed)

        validation_predictions = make_predictions(model, validate)
        validation_score = evaluation.nwrmsle(validation_predictions, validate['unit_sales'].values, validate['perishable'].values)
        (rmse, mae, r2) = evaluation.eval_metrics(validate['unit_sales'].values, validation_predictions)
        
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("random_state", seed)
        mlflow.log_metric('validation_score', validation_score)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)
        mlflow.sklearn.log_model(model, 'decision_tree')
        
        write_score(validation_score, model)

if __name__ == "__main__":
    run(max_depth=4, seed=8675309, index='')
    run(max_features='sqrt', seed=8675309, index=1)
    run(max_features='sqrt', max_depth=4, seed=8675309, index=2)
    run(seed=8675309, index=3)
    # main(model=Model.DECISION_TREE, seed=8675309)
