import math
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import LinearSVC
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score


def load_data():
    train_df = pd.read_csv(r'orange_small_churn_train_data.csv', index_col='ID')
    test_df  = pd.read_csv(r'orange_small_churn_test_data.csv', index_col='ID')
    X = train_df.drop(['labels'], axis=1)
    y = train_df['labels'].apply(lambda x: 1 if x==1 else 0)

    X_train, X_hold, \
    y_train, y_hold = train_test_split(X, y,
                                       test_size=0.2,
                                       random_state=9,
                                       shuffle=True,
                                       stratify=y)
    X_test = test_df

    return X_train, X_hold, X_test, y_train, y_hold


def get_pipeline(model,
                 missing='mean',
                 encoder='dummy', enc_params=None,
                 selector=None,   sel_params=None):

    # choose encoder

    enc_params = {} if enc_params is None else enc_params
    if encoder == 'dummy':
        encoder = DummyEncoder(**enc_params)
    elif encoder == 'mean_target':
        encoder = MeanTargetEncoder(**enc_params)
    elif encoder == 'frequency':
        encoder = FrequencyEncoder(**enc_params)
    else:
        encoder = NopeTransformer()

    # choose selector

    sel_params = {} if sel_params is None else sel_params
    if selector == 'lasso_svc':
        selector = LassoSelector(**sel_params)
    elif selector == 'correlation':
        selector = CorrelationSelector(**sel_params)
    else:
        selector = NopeTransformer()

    # construct pipeline

    pipeline = Pipeline(steps=[
            # preprocessing
            ('preprocessing', FeatureUnion([

                # numeric features
                ('numeric', Pipeline(steps=[
                    ('selecting',      FunctionTransformer(lambda data: data.iloc[:, :190], validate=False)),
                    ('float_nan_mean', Imputer(strategy=missing)),
                    ('scaling',        StandardScaler())
                ])),

                # categorical features
                ('categorical',   Pipeline(steps=[
                    ('selecting', FunctionTransformer(lambda data: data.iloc[:, 190:], validate=False)),
                    ('encoding',  encoder)
                ]))
            ])),

            # feature selection
            ('feature_selection', selector),

            # model
            ('model', model)
        ])

    return pipeline


class DummyEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features as one-hot variables with max_categories restriction
    '''
    def __init__(self, columns=None, max_categories=200):
        self.columns = columns
        self.dummy_columns = None
        self.max_categories = max_categories


    def fit(self, X, y=None, **kwargs):
        self.dummy_columns = None
        return self


    def transform(self, X, y=None, **kwargs):
        if self.max_categories is not None:
            X = X[self.columns] if self.columns is not None else X.copy()
            for col in X.columns:
                top_cats = X[col].value_counts()[:self.max_categories].index.values
                X[col] = X[col].apply(lambda x: x if (x in top_cats or x is None) else 'aggr')

        dummy_df = pd.get_dummies(X, columns=self.columns, sparse=True, dummy_na=True)
        new_cols = dummy_df.columns.values
        if self.dummy_columns is None:
            self.dummy_columns = new_cols
            return dummy_df
        else:
            res_df = pd.DataFrame()
            for col in self.dummy_columns:
                res_df[col] = dummy_df[col] if col in new_cols else np.zeros((len(X),), dtype=int)
        return res_df


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features by its mean on target variable
    '''
    def __init__(self, columns=None):
        self.columns = columns
        self.dict = None
        return


    def fit(self, X, y=None, **kwargs):
        columns = X.columns if self.columns is None else self.columns
        dict = {}

        X = X.astype(str)

        for col in columns:
            vals = X[col].unique()
            dict[col] = { val: y[X[col] == val].mean() for val in vals }

        self.dict = dict

        return self


    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self, ['dict'])

        X = X.astype(str)
        columns = X.columns if self.columns is None else self.columns

        for col in columns:
            col_dict = self.dict[col]
            X[col] = X[col].apply(lambda x: col_dict.get(x, 0))

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    '''
    Encodes categorical features by its frequency
    '''
    def __init__(self, columns=None):
        self.columns = columns
        self.dict = None
        return


    def fit(self, X, y=None, **kwargs):
        columns = X.columns if self.columns is None else self.columns
        dict = {}

        X = X.astype(str)
        n = len(X)

        for col in columns:
            vals = X[col].unique()
            dict[col] = { val: (X[col] == val).sum()/n for val in vals }

        self.dict = dict

        return self


    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self, ['dict'])
        X = X.astype(str)
        columns = X.columns if self.columns is None else self.columns

        for col in columns:
            col_dict = self.dict[col]
            X[col] = X[col].apply(lambda x: col_dict.get(x, 0))

        return X


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_top=100):
        self.n_top = n_top


    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. filter out const columns
        vc = X.apply(lambda col: len(col.value_counts()))
        all_cols = vc[vc > 1].index.values

        # 2. correlation feature selection
        num_corrs = X[all_cols].apply(lambda col: correlation(col.values, y), axis=0)
        top_corrs = num_corrs.abs().sort_values(ascending=False)[:self.n_top]

        self.new_cols = sorted(top_corrs.index)

        return self


    def transform(self, X):
        check_is_fitted(self, ['new_cols'])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.loc[:, self.new_cols]


class LassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=None, C=0.1):
        self.threshold = threshold
        self.C = C
        return


    def fit(self, X, y=None):
        model = LinearSVC(C=self.C, penalty='l1', dual=False)
        model.fit(X, y)
        self.selector = SelectFromModel(model, prefit=True, threshold=self.threshold)
        return self


    def transform(self, X):
        check_is_fitted(self, ['selector'])
        return self.selector.transform(X)


class NopeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def filter_outliers(X, y, cols, alpha=0.01):
    print('filtering outliers...')
    for col in cols:
        var = X[col]
        var_churn = var[y==1]
        var_loyal = var[y==0]

        outliers = len(X)
        condition = None
        col_a = alpha

        while outliers > 200:
            churn_min, churn_max = var_churn.quantile([col_a, 1 - col_a])
            loyal_min, loyal_max = var_loyal.quantile([col_a, 1 - col_a])

            condition = var.isnull() | \
                        ((y==1) & (churn_min <= var) & (var <= churn_max)) | \
                        ((y==0) & (loyal_min <= var) & (var <= loyal_max))

            outliers = len(X) - len(X[condition])
            col_a /= 2

        if condition is not None:
            X = X[condition]
            y = y[condition]
    print('finished: ', len(X))

    return X, y


def undersample(X, y, coeff):

    np.random.seed(9)

    churn = X[y==1].index
    loyal = X[y==0].index.values
    np.random.shuffle(loyal)
    u_loyal = loyal[: int(coeff*len(loyal))]

    u_ids = list(churn) + list(u_loyal)

    return X.ix[u_ids, :], y.ix[u_ids]


def correlation(x, y):
    if set(np.unique(x)) == { 0.0, 1.0 }:
        return matthews_corrcoef(x, y)
    else:
        return point_biserial_corr(x, y)


def point_biserial_corr(x, y):
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]
    
    if len(x) == 0 or len(y) == 0:
        return 0
    
    p = y.mean()
    q = 1 - p
    ex = x.mean()
    sx = x.std(ddof=0)

    px = x[y==1]
    nx = x[y==0]

    mpx = px.mean() if len(px)>0 else 0
    mnx = nx.mean() if len(nx)>0 else 0

    if mpx == mnx:
        return 0
    
    return (mpx - mnx)/sx*math.sqrt(p*q)


def cramers_corrected_stat(x, y):
    cm = confusion_matrix(y, x)
    cm = cm[~np.all(cm == 0, axis=1)]
    if np.all(cm == 0, axis=0).sum() > 0:
        return 0
    
    res = stats.chi2_contingency(cm)
        
    chi2 = res[0]
    n = cm.sum()
    phi2 = chi2/n
    r, k = cm.shape

    return np.sqrt(phi2 / (min(k, r) - 1))


def plot_hist(X, y, cols, n_cols, figsize=(16, 10)):
    current_palette = sns.color_palette()

    n_rows = math.ceil(len(cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    legend_lines = [Line2D([0], [0], color=current_palette[0], lw=4),
                    Line2D([0], [0], color=current_palette[1], lw=4)]
    plt.figlegend(legend_lines, ['Loyal', 'Churn'], loc = 'upper center')

    for idx, col in enumerate(cols):
        if len(cols) <= n_cols:
            ax = axes[idx]
        else:
            i = idx//n_cols
            j = idx%n_cols
            ax = axes[i, j]
        ax.text(.5, .9, col, horizontalalignment='center', transform=ax.transAxes)
        
        x = X[col].dropna()
        x[y==0].hist(ax = ax, bins=50, color=current_palette[0], alpha=1, density=True)
        x[y==1].hist(ax = ax, bins=50, color=current_palette[1], alpha=0.8, density=True)
    
    plt.show()
    
  
def plot_cat_classes(X, y, cols, n_cols, figsize=(16, 3)):
    current_palette = sns.color_palette()
    plot_data = []
    
    for col in cols:
        x = X[col].dropna()
        x_loyal = x[y==0].value_counts()
        x_churn = x[y==1].value_counts()

        plot_data.append((col, x_loyal, x_churn))
  
    n_rows = math.ceil(len(plot_data)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    legend_lines = [Line2D([0], [0], color=current_palette[0], lw=4),
                    Line2D([0], [0], color=current_palette[1], lw=4)]
    plt.figlegend(legend_lines, ['Loyal', 'Churn'], loc = 'upper center')
        
    for idx, p in enumerate(plot_data): 
        col, x_loyal, x_churn = p
        
        if len(cols) <= n_cols:
            ax = axes[idx]
        else:
            i = idx//n_cols
            j = idx%n_cols
            ax = axes[i, j]

        ax.text(.5, .9, col, horizontalalignment='center', transform=ax.transAxes)
        
        if len(x_loyal) > 0:
            x_loyal.plot(ax = ax, kind='bar', color=current_palette[0], alpha=1)
        if len(x_churn) > 0:
            x_churn.plot(ax = ax, kind='bar', color=current_palette[1], alpha=0.8)
      
    plt.show()
    
    
def L(y_test, y_pred, **kwargs):
    '''
    Полный экономический эффект кампании с учетом всех параметров и ограничений
    '''
    M_, H_, p_, g_, T_ =  kwargs['M'], kwargs['H'], kwargs['p'], kwargs['g'], kwargs['T']
    
    threshold = np.percentile(y_pred, 100 - g_)
    a = (y_pred > threshold).astype(int)

    TP = ((y_test == 1) & (a == 1)).sum()
    FP = ((y_test == 0) & (a == 1)).sum()
    TN = ((y_test == 0) & (a == 0)).sum()
    FN = ((y_test == 1) & (a == 0)).sum()
    
    if (TP + FP)*H_ > T_:
        return None
    
    L = TP*(p_*M_ - H_) - FP*H_
    
    return L
    

def involved_vs_effect(y_test, y_pred, **kwargs):
    M_, H_, p_, T_ =  kwargs['M'], kwargs['H'], kwargs['p'], kwargs['T']
    
    gs = np.linspace(0, 2, 100)
    ls = np.array([L(y_test, y_pred, M=M_, H=H_, p=p_, g=g_, T=T_)  for g_ in gs])
    gs = gs[np.where(ls != None)]
    ls = ls[np.where(ls != None)]
    
    plt.plot(gs, ls)
    plt.grid()
    plt.xlabel('% вовлеченных в кампанию')
    plt.ylabel('экономический эффект')
    plt.show()
    
    return gs, ls


def involved_vs_other(y_test, y_pred, **kwargs):
    M_, p_, T_ =  kwargs['M'], kwargs['p'], kwargs['T']
    
    Hs = np.linspace(10, 80, 100)
    gm = []
    lm = []
    for H_ in Hs:
        gs = np.linspace(0, 2, 100)
        ls = np.array([L(y_test, y_pred, M=M_, H=H_, p=p_, g=g_, T=T_)  for g_ in gs])
        gs = gs[np.where(ls != None)]
        ls = ls[np.where(ls != None)]
        idx = np.argmax(np.array(ls))
        gm.append(gs[idx])
        lm.append(ls[idx])
    
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    plt.plot(Hs, gm)
    plt.grid()
    plt.xlabel('стоимость удержания')
    plt.ylabel('% вовлеченных в кампанию')
    
    plt.subplot(122)
    plt.plot(Hs, lm)
    plt.grid()
    plt.xlabel('стоимость удержания')
    plt.ylabel('эффект кампании')
    
    plt.show()
    
    return

    
def prob_vs_other(y_test, y_pred, **kwargs):
    M_, H_, T_ =  kwargs['M'], kwargs['H'], kwargs['T']
    
    ps = np.linspace(0.1, 0.9, 50)
    gm = []
    lm = []
    for p_ in ps:
        gs = np.linspace(0, 2, 100)
        ls = np.array([L(y_test, y_pred, M=M_, H=H_, p=p_, g=g_, T=T_)  for g_ in gs])
        gs = gs[np.where(ls != None)]
        ls = ls[np.where(ls != None)]
        idx = np.argmax(np.array(ls))
        gm.append(gs[idx])
        lm.append(ls[idx])
    
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    plt.plot(ps, gm)
    plt.grid()
    plt.xlabel('вероятность принять предложение')
    plt.ylabel('% вовлеченных в кампанию')
    
    plt.subplot(122)
    plt.plot(ps, lm)
    plt.grid()
    plt.xlabel('вероятность принять предложение')
    plt.ylabel('эффект кампании')
    
    plt.show()
    
    return


def score_vs_campaigh_eff(y_test, y_pred, **kwargs):
    M_, H_, p_, g_, T_ =  kwargs['M'], kwargs['H'], kwargs['p'], kwargs['g'], kwargs['T']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for H_ in [20, 30, 40]:
        scores = []
        ls = []
    
        for t in np.linspace(0, 0.99, 100):
            y_mod = y_test*t + y_pred*(1 - t)
            score = roc_auc_score(y_test, y_mod)
            scores.append(score)
            l = L(y_test, y_mod, M=M_, H=H_, p=p_, g=g_, T=T_)
            ls.append(l)
            
        axes[0].plot(scores, ls, label='H = {0}'.format(H_))
        axes[0].legend(loc='upper center')
        axes[0].grid()  
        axes[0].set_xlabel('roc-auc')
        axes[0].set_ylabel('эффект кампании')
    
    for p_ in [0.3, 0.5, 0.7]:
        scores = []
        ls = []
    
        for t in np.linspace(0, 0.99, 100):
            y_mod = y_test*t + y_pred*(1 - t)
            score = roc_auc_score(y_test, y_mod)
            scores.append(score)
            l = L(y_test, y_mod, M=M_, H=H_, p=p_, g=g_, T=T_)
            ls.append(l)
            
        axes[1].plot(scores, ls, label='p = {0}'.format(p_))
        axes[1].legend(loc='upper center')
        axes[1].grid()  
        axes[1].set_xlabel('roc-auc')
        axes[1].set_ylabel('эффект кампании')
    
    
    for g_ in [0.5, 1, 2]:
        scores = []
        ls = []
    
        for t in np.linspace(0, 0.99, 100):
            y_mod = y_test*t + y_pred*(1 - t)
            score = roc_auc_score(y_test, y_mod)
            scores.append(score)
            l = L(y_test, y_mod, M=M_, H=H_, p=p_, g=g_, T=T_)
            ls.append(l)
            
        axes[2].plot(scores, ls, label='g = {0}'.format(g_))
        axes[2].legend(loc='upper center')
        axes[2].grid()  
        axes[2].set_xlabel('roc-auc')
        axes[2].set_ylabel('эффект кампании')
        
    plt.show()
    
    return
