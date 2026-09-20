"""Training-fitted, estimator-appropriate feature representations."""
from .common import *
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack


class Features:
    def __init__(self, meta, cfg, seed):
        self.meta, self.cfg, self.seed = meta, cfg, seed

    def fit(self, x, y):
        if self.meta['kind'] == 'text':
            self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                        max_features=self.cfg['word_features'], dtype=np.float32)
            self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True,
                        max_features=self.cfg['char_features'], dtype=np.float32)
            w = self.word.fit_transform(x.text)
            self.char.fit(x.text)
            self.selector = SelectKBest(chi2, k=min(w.shape[1], self.cfg['selected_features'])).fit(w, y)
            self.svd = TruncatedSVD(n_components=max(1, min(self.cfg['svd_components'], min(w.shape) - 1)),
                                   random_state=self.seed)
            self.scaler = StandardScaler().fit(self.svd.fit_transform(w))
        else:
            self.cats = list(x.select_dtypes(include=['object', 'string', 'category', 'bool']).columns)
            self.nums = [c for c in x if c not in self.cats]
            self.medians = x[self.nums].median().fillna(0)
            self.tab = ColumnTransformer([
                ('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), self.nums),
                ('category', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), self.cats)], sparse_threshold=0).fit(x)
        return self

    def transform(self, x):
        if self.meta['kind'] == 'text':
            word = self.word.transform(x.text)
            selected = self.selector.transform(word).tocsr()
            dense = selected.toarray()
            return dict(linear=hstack([word, self.char.transform(x.text)], format='csr'),
                        word=word, tree=selected, dense=dense,
                        knn=self.scaler.transform(self.svd.transform(word)).astype(np.float32),
                        cat=dense)
        dense = np.asarray(self.tab.transform(x), dtype=np.float32)
        native = x.copy()
        native[self.nums] = native[self.nums].fillna(self.medians).astype(float)
        for col in self.cats:
            native[col] = native[col].fillna('__MISSING__').astype(str)
        return dict(linear=dense, word=dense, tree=dense, dense=dense, knn=dense, cat=native)

    def description(self, key):
        if self.meta['kind'] != 'text':
            return 'native categorical + numeric median imputation' if key == 'cat' else 'numeric imputation/scaling + categorical one-hot'
        return {'linear': 'word+character TF-IDF', 'word': 'word/bigram TF-IDF',
                'tree': 'word TF-IDF + training-label chi-square selection (sparse)',
                'dense': 'word TF-IDF + training-label chi-square selection (dense)',
                'cat': 'word TF-IDF + training-label chi-square selection (dense)',
                'knn': 'word TF-IDF + SVD + scaling'}[key]
