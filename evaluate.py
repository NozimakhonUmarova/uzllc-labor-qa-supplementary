import pandas as pd, numpy as np, re, json, math, random, os, csv, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sk_normalize

BASE=Path(__file__).resolve().parents[1]
corpus=pd.read_csv(BASE/'data/corpus_units.csv')
questions=pd.read_csv(BASE/'data/questions.csv')

# ensure str fields
for c in ['article_title','raw_text','normalized_text','legal_topics','source_url','answer_source']:
    if c in corpus.columns:
        corpus[c]=corpus[c].fillna('').astype(str)

# text for retrieval: title + normalized text + topics (without JSON quotes loosely)
def norm_text(s):
    if pd.isna(s): return ''
    s=str(s).lower()
    # unify Uzbek apostrophe variants (for retrieval only)
    s=s.replace('ʻ', "'").replace('ʼ', "'").replace('‘', "'").replace('’', "'").replace('`', "'").replace('´', "'")
    # keep letters/numbers/apostrophe, split others
    s=re.sub(r"[^0-9a-zа-яёғқҳўʼ'\-]+", ' ', s, flags=re.IGNORECASE)
    s=re.sub(r"\s+", ' ', s).strip()
    return s

def tokenize(s):
    s=norm_text(s)
    # tokens incl hyphen? split hyphen separately? keep words with apostrophe
    toks=re.findall(r"[0-9a-zа-яёғқҳў]+(?:'[0-9a-zа-яёғқҳў]+)?", s, flags=re.IGNORECASE)
    return toks

corpus['retrieval_text']=(corpus['article_title'].fillna('')+' '+corpus['normalized_text'].fillna('')+' '+corpus['legal_topics'].fillna('')).map(norm_text)
questions['q_norm']=questions['question_text'].map(norm_text)

# gold map
def parse_gold(x):
    if pd.isna(x) or str(x).strip()=='' or str(x).lower()=='nan': return []
    return [t.strip() for t in str(x).split('|') if t.strip()]
questions['gold_list']=questions['validated_gold_corpus_ids'].apply(parse_gold)

# Split
is_dev=(questions['split']=='dev') & (questions['answerability']=='answerable')
is_test=(questions['split']=='test') & (questions['answerability']=='answerable')
is_test_all=(questions['split']=='test')

# BM25 implementation
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.docs_tokens=docs_tokens
        self.N=len(docs_tokens)
        self.k1=k1; self.b=b
        self.doc_len=np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl=float(self.doc_len.mean()) if self.N else 0
        self.term_freqs=[]
        df=Counter()
        for toks in docs_tokens:
            tf=Counter(toks)
            self.term_freqs.append(tf)
            for term in tf: df[term]+=1
        self.idf={term: math.log(1+(self.N-n+0.5)/(n+0.5)) for term,n in df.items()}
    def score(self, query_tokens):
        scores=np.zeros(self.N, dtype=float)
        qterms=Counter(query_tokens)
        for term, qf in qterms.items():
            idf=self.idf.get(term)
            if idf is None: continue
            for i,tf in enumerate(self.term_freqs):
                f=tf.get(term,0)
                if f==0: continue
                denom=f + self.k1*(1-self.b + self.b*self.doc_len[i]/self.avgdl)
                scores[i]+=idf * f*(self.k1+1)/denom
        return scores

corpus_tokens=[tokenize(t) for t in corpus['retrieval_text']]
q_tokens={idx: tokenize(t) for idx,t in questions['q_norm'].items()}

def rank_from_scores(scores, topk=10):
    # stable sort by score desc then corpus index asc
    return np.lexsort((np.arange(len(scores)), -scores))[:topk]

def eval_scores_for_method(score_func, qdf):
    rows=[]; metrics=[]
    for _,q in qdf.iterrows():
        scores=score_func(q)
        order=rank_from_scores(scores, topk=10)
        gold=set(q['gold_list'])
        rr=0.0
        for rank,idx in enumerate(order, start=1):
            cid=corpus.iloc[idx]['corpus_id']
            if cid in gold and rr==0:
                rr=1.0/rank
                break
        hit1=any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:1])
        hit3=any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:3])
        hit5=any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:5])
        hit10=any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:10])
        metrics.append((hit1,hit3,hit5,hit10,rr))
    if not metrics:
        return {}
    arr=np.array(metrics, dtype=float)
    return {
        'Recall@1': arr[:,0].mean(),
        'Recall@3': arr[:,1].mean(),
        'Recall@5': arr[:,2].mean(),
        'Recall@10': arr[:,3].mean(),
        'MRR@10': arr[:,4].mean(),
        'n': len(metrics),
        'hits@1': int(arr[:,0].sum()),
        'hits@3': int(arr[:,1].sum()),
        'hits@5': int(arr[:,2].sum()),
        'hits@10': int(arr[:,3].sum()),
        'rr_sum': float(arr[:,4].sum())
    }

# dev grid BM25
bm25_candidates=[]
for k1 in [1.2,1.5,1.8,2.0]:
    for b in [0.3,0.5,0.75,0.9]:
        model=BM25(corpus_tokens,k1=k1,b=b)
        met=eval_scores_for_method(lambda q, m=model: m.score(q_tokens[q.name]), questions[is_dev])
        bm25_candidates.append({'k1':k1,'b':b,**met})
bm25_grid=pd.DataFrame(bm25_candidates)
# select by MRR@10 then R@3 R@1
bm25_best=bm25_grid.sort_values(['MRR@10','Recall@3','Recall@1'], ascending=False).iloc[0].to_dict()
bm25=BM25(corpus_tokens,k1=bm25_best['k1'],b=bm25_best['b'])

# TF-IDF + LSA grid
vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1,2), min_df=1, max_df=0.95, sublinear_tf=True, norm='l2')
X_tfidf=vectorizer.fit_transform(corpus['retrieval_text'])
Q_tfidf_all=vectorizer.transform(questions['q_norm'])
max_comp=min(200, X_tfidf.shape[0]-1, X_tfidf.shape[1]-1)
component_grid=[25,50,75,100,150,200]
component_grid=[c for c in component_grid if c<=max_comp]
lsa_models={}
lsa_candidates=[]
for ncomp in component_grid:
    svd=TruncatedSVD(n_components=ncomp, random_state=42)
    X_lsa=svd.fit_transform(X_tfidf)
    X_lsa=sk_normalize(X_lsa)
    Q_lsa=svd.transform(Q_tfidf_all)
    Q_lsa=sk_normalize(Q_lsa)
    lsa_models[ncomp]=(svd,X_lsa,Q_lsa,svd.explained_variance_ratio_.sum())
    met=eval_scores_for_method(lambda q, X=X_lsa, Q=Q_lsa: X @ Q[q.name], questions[is_dev])
    lsa_candidates.append({'n_components':ncomp,'explained_variance_ratio_sum':svd.explained_variance_ratio_.sum(),**met})
lsa_grid=pd.DataFrame(lsa_candidates)
lsa_best=lsa_grid.sort_values(['MRR@10','Recall@3','Recall@1'], ascending=False).iloc[0].to_dict()
svd, X_lsa, Q_lsa, evr=lsa_models[int(lsa_best['n_components'])]

# Score normalization for hybrid
def minmax(scores):
    mn=float(np.min(scores)); mx=float(np.max(scores))
    if mx-mn < 1e-12:
        return np.zeros_like(scores,dtype=float)
    return (scores-mn)/(mx-mn)

def bm25_score(q): return bm25.score(q_tokens[q.name])
def lsa_score(q): return X_lsa @ Q_lsa[q.name]

hyb_candidates=[]
for alpha in [round(i/10,1) for i in range(11)]:
    def hs(q, a=alpha):
        return a*minmax(bm25_score(q)) + (1-a)*minmax(lsa_score(q))
    met=eval_scores_for_method(hs, questions[is_dev])
    hyb_candidates.append({'alpha':alpha, **met})
hybrid_grid=pd.DataFrame(hyb_candidates)
hybrid_best=hybrid_grid.sort_values(['MRR@10','Recall@3','Recall@1'], ascending=False).iloc[0].to_dict()
alpha=float(hybrid_best['alpha'])

def hybrid_score(q):
    return alpha*minmax(bm25_score(q)) + (1-alpha)*minmax(lsa_score(q))

# Evaluation on test answerable
methods={
    'BM25': bm25_score,
    'TF-IDF+LSA': lsa_score,
    'BM25+LSA hybrid': hybrid_score,
}
summary=[]
perq_metrics={}
for name,func in methods.items():
    qdf=questions[is_test].copy()
    metrics=[]
    top_rows=[]
    for _,q in qdf.iterrows():
        scores=func(q)
        order=rank_from_scores(scores, topk=10)
        gold=set(q['gold_list'])
        rr=0.0
        hitrank=None
        for rank,idx in enumerate(order, start=1):
            cid=corpus.iloc[idx]['corpus_id']
            hit = cid in gold
            if hit and rr==0:
                rr=1.0/rank; hitrank=rank
            top_rows.append({
                'method':name,
                'question_id':q['question_id'],
                'question_text':q['question_text'],
                'question_type':q['question_type'],
                'legal_topic':q['legal_topic'],
                'answerability':q['answerability'],
                'gold_corpus_ids':'|'.join(q['gold_list']),
                'rank':rank,
                'retrieved_corpus_id':cid,
                'score':float(scores[idx]),
                'hit': bool(hit),
                'retrieved_article_number':corpus.iloc[idx]['article_number'],
                'retrieved_unit_number':corpus.iloc[idx]['unit_number'],
                'retrieved_answer_source':corpus.iloc[idx]['answer_source'],
                'retrieved_text':corpus.iloc[idx]['raw_text'][:500]
            })
        m={
            'question_id':q['question_id'],
            'method':name,
            'hit@1': any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:1]),
            'hit@3': any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:3]),
            'hit@5': any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:5]),
            'hit@10': any(corpus.iloc[idx]['corpus_id'] in gold for idx in order[:10]),
            'rr':rr,
            'first_relevant_rank':hitrank
        }
        metrics.append(m)
    mdf=pd.DataFrame(metrics)
    perq_metrics[name]=mdf
    n=len(mdf)
    summary.append({
        'method':name,
        'test_questions_total': int((questions['split']=='test').sum()),
        'answerable_test_questions':n,
        'unanswerable_test_questions': int(((questions['split']=='test') & (questions['answerability']=='unanswerable')).sum()),
        'Recall@1':mdf['hit@1'].mean(),
        'Recall@1_count': f"{int(mdf['hit@1'].sum())}/{n}",
        'Recall@3':mdf['hit@3'].mean(),
        'Recall@3_count': f"{int(mdf['hit@3'].sum())}/{n}",
        'Recall@5':mdf['hit@5'].mean(),
        'Recall@5_count': f"{int(mdf['hit@5'].sum())}/{n}",
        'MRR@10':mdf['rr'].mean(),
        'RR_sum':mdf['rr'].sum()
    })
    pd.DataFrame(top_rows).to_csv(BASE/f"{name.lower().replace('+','_').replace(' ','_').replace('-','_')}_top10_results_v1.csv",index=False)

summary_df=pd.DataFrame(summary)

# confidence intervals
# Wilson interval for recall proportions
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return center-half, center+half

ci_rows=[]
for _,row in summary_df.iterrows():
    n=int(row['answerable_test_questions'])
    for kname in ['Recall@1','Recall@3','Recall@5']:
        k=int(str(row[f'{kname}_count']).split('/')[0])
        lo,hi=wilson(k,n)
        ci_rows.append({'method':row['method'],'metric':kname,'estimate':row[kname],'count':f'{k}/{n}','ci_95_low':lo,'ci_95_high':hi,'ci_method':'Wilson score interval'})
    # bootstrap CI for MRR
    vals=perq_metrics[row['method']]['rr'].to_numpy()
    rng=np.random.default_rng(42)
    boots=np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(10000)])
    lo,hi=np.percentile(boots,[2.5,97.5])
    ci_rows.append({'method':row['method'],'metric':'MRR@10','estimate':row['MRR@10'],'count':'','ci_95_low':lo,'ci_95_high':hi,'ci_method':'Nonparametric bootstrap over questions, 10,000 resamples'})
ci_df=pd.DataFrame(ci_rows)

# paired bootstrap difference hybrid - bm25 for recall3 and MRR
bm=perq_metrics['BM25'].set_index('question_id')
hy=perq_metrics['BM25+LSA hybrid'].set_index('question_id')
common=list(bm.index.intersection(hy.index))
rng=np.random.default_rng(123)
diff_rows=[]
for metric,col in [('Recall@1','hit@1'),('Recall@3','hit@3'),('Recall@5','hit@5'),('MRR@10','rr')]:
    diff_obs=(hy.loc[common,col].astype(float)-bm.loc[common,col].astype(float)).mean()
    n=len(common)
    boots=[]
    diffs=(hy.loc[common,col].astype(float)-bm.loc[common,col].astype(float)).to_numpy()
    for _ in range(10000):
        idx=rng.integers(0,n,n)
        boots.append(diffs[idx].mean())
    lo,hi=np.percentile(boots,[2.5,97.5])
    # two-sided p approximate: twice min mass <=0 / >=0
    boots=np.array(boots)
    p=2*min(np.mean(boots<=0), np.mean(boots>=0))
    p=min(p,1.0)
    diff_rows.append({'comparison':'BM25+LSA hybrid - BM25','metric':metric,'difference':diff_obs,'ci_95_low':lo,'ci_95_high':hi,'bootstrap_p_approx':p})
diff_df=pd.DataFrame(diff_rows)

# error examples: hybrid misses or bm25/hybrid disagreements
err_rows=[]
for name,mdf in perq_metrics.items():
    misses=mdf[~mdf['hit@3']].head(10)
    topfile=BASE/f"{name.lower().replace('+','_').replace(' ','_').replace('-','_')}_top10_results_v1.csv"
    tdf=pd.read_csv(topfile)
    for _,m in misses.iterrows():
        qrow=questions[questions.question_id==m.question_id].iloc[0]
        top1=tdf[(tdf.question_id==m.question_id)&(tdf['rank']==1)].iloc[0]
        err_rows.append({
            'method':name,
            'question_id':m.question_id,
            'question_text':qrow.question_text,
            'gold_corpus_ids':'|'.join(qrow.gold_list),
            'top1_retrieved_corpus_id':top1.retrieved_corpus_id,
            'top1_answer_source':top1.retrieved_answer_source,
            'error_type':'top3_miss',
            'explanation':'The correct gold unit was not ranked in the top 3; terminology may overlap with a different provision.'
        })
error_df=pd.DataFrame(err_rows)

# all-test strict (including unanswerable as misses)
strict_rows=[]
for name,mdf in perq_metrics.items():
    n_ans=len(mdf); n_total=int((questions['split']=='test').sum())
    strict_rows.append({'method':name,'test_questions_total':n_total,'note':'Unanswerable questions treated as misses because no abstention threshold was applied.',
        'Strict_R@1_all_test':mdf['hit@1'].sum()/n_total,'Strict_R@1_count':f"{int(mdf['hit@1'].sum())}/{n_total}",
        'Strict_R@3_all_test':mdf['hit@3'].sum()/n_total,'Strict_R@3_count':f"{int(mdf['hit@3'].sum())}/{n_total}",
        'Strict_R@5_all_test':mdf['hit@5'].sum()/n_total,'Strict_R@5_count':f"{int(mdf['hit@5'].sum())}/{n_total}",
        'Strict_MRR_all_test':mdf['rr'].sum()/n_total})
strict_df=pd.DataFrame(strict_rows)

# write grids and configs
summary_df.to_csv(BASE/'retrieval_results_summary_v1.csv',index=False)
ci_df.to_csv(BASE/'retrieval_confidence_intervals_v1.csv',index=False)
diff_df.to_csv(BASE/'paired_bootstrap_differences_v1.csv',index=False)
bm25_grid.to_csv(BASE/'bm25_dev_grid_v1.csv',index=False)
lsa_grid.to_csv(BASE/'lsa_dev_grid_v1.csv',index=False)
hybrid_grid.to_csv(BASE/'hybrid_alpha_dev_grid_v1.csv',index=False)
error_df.to_csv(BASE/'retrieval_error_examples_v1.csv',index=False)
strict_df.to_csv(BASE/'strict_all_test_metrics_v1.csv',index=False)

params={
    'corpus_file':'data/corpus_units.csv',
    'questions_file':'data/questions.csv',
    'gold_mapping_file':'data/gold_mapping.csv',
    'corpus_units':int(len(corpus)),
    'test_questions_total':int((questions['split']=='test').sum()),
    'answerable_test_questions':int(is_test.sum()),
    'unanswerable_test_questions':int(((questions['split']=='test') & (questions['answerability']=='unanswerable')).sum()),
    'evaluation_scope':'Primary Recall@k and MRR@10 are computed on answerable test questions only. Unanswerable questions are retained for future abstention/answerability experiments.',
    'normalization':'lowercase; Unicode apostrophe variants unified; repeated whitespace removed; non-letter/number punctuation treated as token boundaries',
    'tokenization':'Unicode-aware word-level tokenization; numbers retained; Uzbek Cyrillic/Latin characters retained',
    'stopword_handling':'no stopword removal; legal function words and negation markers retained',
    'retrieval_text':'article_title + normalized_text + legal_topics',
    'bm25':{
        'k1':float(bm25_best['k1']),
        'b':float(bm25_best['b']),
        'selection':'grid search on development answerable questions; selected by MRR@10, then Recall@3, then Recall@1',
        'grid':{'k1':[1.2,1.5,1.8,2.0],'b':[0.3,0.5,0.75,0.9]}
    },
    'tfidf':{
        'analyzer':'word',
        'ngram_range':[1,2],
        'min_df':1,
        'max_df':0.95,
        'sublinear_tf':True,
        'norm':'l2',
        'vocabulary_size':int(len(vectorizer.vocabulary_))
    },
    'lsa':{
        'n_components':int(lsa_best['n_components']),
        'explained_variance_ratio_sum':float(lsa_best['explained_variance_ratio_sum']),
        'selection':'grid search on development answerable questions; selected by MRR@10, then Recall@3, then Recall@1',
        'component_grid':component_grid,
        'cosine_similarity':'cosine over L2-normalized LSA vectors'
    },
    'hybrid':{
        'formula':'H(q,d)=alpha*minmax(BM25(q,d))+(1-alpha)*minmax(LSA(q,d))',
        'score_normalization':'per-query min-max normalization for BM25 and LSA scores',
        'alpha':alpha,
        'alpha_grid':[round(i/10,1) for i in range(11)],
        'selection':'alpha selected on development answerable questions using MRR@10'
    },
    'confidence_intervals':{
        'recall':'Wilson score interval, 95%',
        'mrr':'nonparametric bootstrap over questions, 10000 resamples, 95%',
        'paired_difference':'paired bootstrap over questions, 10000 resamples'
    },
    'random_seed':42
}
with open(BASE/'retrieval_parameters_v1.json','w',encoding='utf-8') as f:
    json.dump(params,f,ensure_ascii=False,indent=2)

# Markdown report
md=[]
md.append('# UZLLC Retrieval Evaluation Report v1\n')
md.append('## Dataset\n')
md.append(f'- Corpus units: {len(corpus)}\n')
md.append(f'- Development questions: {(questions["split"]=="dev").sum()} total, {is_dev.sum()} answerable\n')
md.append(f'- Test questions: {(questions["split"]=="test").sum()} total, {is_test.sum()} answerable and {((questions["split"]=="test") & (questions["answerability"]=="unanswerable")).sum()} unanswerable\n')
md.append('- Primary retrieval metrics are calculated on answerable test questions only, because BM25/LSA/hybrid ranking does not include an abstention threshold for unanswerable queries.\n')
md.append('\n## Selected parameters\n')
md.append(f'- BM25: k1={bm25_best["k1"]}, b={bm25_best["b"]}\n')
md.append(f'- TF-IDF: word-level, ngram_range=(1,2), min_df=1, max_df=0.95, sublinear_tf=True, no stopword removal\n')
md.append(f'- LSA components: {int(lsa_best["n_components"])}; explained variance sum={lsa_best["explained_variance_ratio_sum"]:.4f}\n')
md.append(f'- Hybrid alpha: {alpha}; selected on dev set by MRR@10\n')
md.append('\n## Test results on answerable questions\n')
md.append(summary_df.to_markdown(index=False))
md.append('\n\n## 95% confidence intervals\n')
md.append(ci_df.to_markdown(index=False))
md.append('\n\n## Paired bootstrap differences\n')
md.append(diff_df.to_markdown(index=False))
md.append('\n\n## Strict all-test metrics including unanswerable as misses\n')
md.append(strict_df.to_markdown(index=False))
md.append('\n\n## Notes\n')
md.append('- Exact counts are included to avoid overinterpreting small percentage differences.\n')
md.append('- Unanswerable questions should be evaluated with a separate abstention or answerability threshold in a future experiment.\n')
(BASE/'retrieval_evaluation_report_v1.md').write_text('\n'.join(md),encoding='utf-8')

# package
zip_path=BASE/'uzllc_retrieval_evaluation_v1_package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    files=[
        'retrieval_results_summary_v1.csv','retrieval_confidence_intervals_v1.csv','paired_bootstrap_differences_v1.csv',
        'bm25_dev_grid_v1.csv','lsa_dev_grid_v1.csv','hybrid_alpha_dev_grid_v1.csv','retrieval_parameters_v1.json',
        'retrieval_evaluation_report_v1.md','retrieval_error_examples_v1.csv','strict_all_test_metrics_v1.csv',
        'bm25_top10_results_v1.csv','tf_idf_lsa_top10_results_v1.csv','bm25_lsa_hybrid_top10_results_v1.csv'
    ]
    for fn in files:
        p=BASE/fn
        if p.exists(): z.write(p,arcname=fn)

print('BM25 best',bm25_best)
print('LSA best',lsa_best)
print('Hybrid best',hybrid_best)
print(summary_df.to_string(index=False))
print(ci_df.to_string(index=False))
print(diff_df.to_string(index=False))
print('zip',zip_path)
