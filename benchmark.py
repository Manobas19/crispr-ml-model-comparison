#!/usr/bin/env python3
"""Group-aware regression benchmark for measured CRISPR guide activity."""
import argparse, hashlib, json, time, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def features(sequences):
    bases='ACGT'; L=len(sequences[0]); rows=[]
    names=[f'pos_{i+1}_{b}' for i in range(L) for b in bases]+['GC_fraction','max_homopolymer','base_entropy']+[a+b+'_fraction' for a in bases for b in bases]
    for s in sequences:
        f=[float(c==b) for c in s for b in bases]
        prob=np.array([s.count(b)/L for b in bases]); nz=prob[prob>0]
        longest=cur=1
        for i in range(1,L):
            cur=cur+1 if s[i]==s[i-1] else 1; longest=max(longest,cur)
        f += [(s.count('G')+s.count('C'))/L,longest,float(-(nz*np.log2(nz)).sum())]
        f += [sum(s[i:i+2]==a+b for i in range(L-1))/(L-1) for a in bases for b in bases]
        rows.append(f)
    return np.asarray(rows),names


def validate(df,length):
    needed=['guide_id','guide_sequence','activity','target_group']
    if not set(needed)<=set(df): raise ValueError('Required CSV columns: '+','.join(needed))
    x=df[needed].copy()
    if x.isna().any().any(): raise ValueError('Missing values are not allowed.')
    for c in ['guide_id','target_group','guide_sequence']:
        x[c]=x[c].astype(str).str.strip()
        if x[c].eq('').any(): raise ValueError('Empty value in '+c)
    x.guide_sequence=x.guide_sequence.str.upper()
    if not x.guide_sequence.str.fullmatch('[ACGT]{'+str(length)+'}').all():
        raise ValueError(f'Use exactly {length} A/C/G/T guide bases, excluding PAM. No N bases or mixed lengths.')
    if x.guide_id.duplicated().any(): raise ValueError('guide_id must be unique.')
    if x.guide_sequence.duplicated().any():
        raise ValueError('Repeated guide sequence detected. Resolve duplicate records/replicates within a consistent assay before benchmarking. Do not average incompatible conditions.')
    x.activity=pd.to_numeric(x.activity,errors='raise')
    if not np.isfinite(x.activity).all() or x.activity.nunique()<3:
        raise ValueError('activity must be finite numeric measurements with meaningful variation, not binary classes.')
    if len(x)<40 or x.target_group.nunique()<12:
        raise ValueError('This benchmark requires at least 40 unique guides and 12 independent target groups; more data is preferable.')
    return x.reset_index(drop=True)


def demo(path):
    rng=np.random.default_rng(42); rows=[]
    for g in range(30):
        group_effect=rng.normal(0,.07)
        for j in range(12):
            seq=''.join(rng.choice(list('ACGT'),20)); gc=(seq.count('G')+seq.count('C'))/20
            y=.45+.25*(seq[3]=='G')-.18*(seq[15]=='T')-.6*(gc-.5)**2+group_effect+rng.normal(0,.05)
            rows.append([f'demo_{g:02d}_{j:02d}',seq,float(np.clip(y,0,1)),f'DEMO_TARGET_{g:02d}'])
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    pd.DataFrame(rows,columns=['guide_id','guide_sequence','activity','target_group']).to_csv(p,index=False)
    print('SIMULATED teaching data:',p)


def models(seed,quick,names):
    trees=60 if quick else 300
    specs={
      'MeanBaseline':(DummyRegressor(strategy='mean'),{}),
      'Ridge':(Pipeline([('scale',StandardScaler()),('model',Ridge())]),{'model__alpha':[.1,1,10,100,1000]}),
      'SVR':(Pipeline([('scale',StandardScaler()),('model',SVR())]),{'model__C':[.1,1,10,100],'model__epsilon':[.01,.05,.1],'model__gamma':['scale',.001,.01]}),
      'RandomForest':(RandomForestRegressor(n_estimators=trees,random_state=seed,n_jobs=1),{'max_depth':[None,5,10],'min_samples_leaf':[1,3,6],'max_features':[.5,1.0]})
    }
    if 'XGBoost' in names:
        try: from xgboost import XGBRegressor
        except ImportError as e: raise ImportError('Install requirements.txt to include XGBoost; or explicitly choose --models MeanBaseline,Ridge,SVR,RandomForest.') from e
        specs['XGBoost']=(XGBRegressor(objective='reg:squarederror',tree_method='hist',n_estimators=trees,random_state=seed,n_jobs=1),
                          {'max_depth':[2,3,5],'learning_rate':[.03,.1,.2],'subsample':[.8,1.0],'colsample_bytree':[.8,1.0],'reg_lambda':[1,5]})
    bad=set(names)-set(specs)
    if bad: raise ValueError('Unknown models: '+str(bad))
    return {k:specs[k] for k in names}


def metrics(y,p):
    rho=float(spearmanr(y,p).statistic) if np.std(y)>0 and np.std(p)>0 else np.nan
    r2=float(r2_score(y,p)) if np.std(y)>0 else np.nan
    return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'R2':r2,'Spearman':rho}


def run(a):
    if not .1<=a.test_fraction<=.4 or a.bootstrap<100 or a.sequence_length<2: raise ValueError('Invalid settings.')
    df=validate(pd.read_csv(a.input,dtype={'guide_id':str,'guide_sequence':str,'target_group':str}),a.sequence_length)
    names=list(dict.fromkeys(a.models.split(',')))
    specs=models(a.seed,a.quick,names)
    X,feature_names=features(df.guide_sequence.tolist()); y=df.activity.to_numpy(); groups=df.target_group.to_numpy()
    split=GroupShuffleSplit(n_splits=1,test_size=a.test_fraction,random_state=a.seed)
    train,test=next(split.split(X,y,groups))
    if len(set(groups[test]))<3 or len(set(groups[train]))<5:
        raise ValueError('Need >=3 held-out groups and >=5 training groups. Increase groups/test_fraction.')
    assert not set(groups[train]) & set(groups[test])
    assert not set(df.guide_sequence.iloc[train]) & set(df.guide_sequence.iloc[test])
    folds=min(3 if a.quick else 5,len(set(groups[train])))
    # Materialize identical group-aware folds for every candidate model.
    cv=list(GroupKFold(n_splits=folds).split(X[train],y[train],groups[train]))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    split_table=df[['guide_id','target_group']].copy(); split_table['split']='train'; split_table.loc[test,'split']='test'
    split_table['cv_validation_fold']=pd.Series([None]*len(df),dtype='Int64')
    for f,(_,valid) in enumerate(cv): split_table.loc[train[valid],'cv_validation_fold']=f
    split_table.to_csv(out/'split_manifest.csv',index=False)
    fitted={}; cv_rows=[]; best_parameters={}
    # All tuning and model-family selection take place before test predictions.
    for name,(estimator,space) in specs.items():
        start=time.perf_counter()
        if space:
            search=RandomizedSearchCV(estimator,space,n_iter=2 if a.quick else 10,scoring='neg_root_mean_squared_error',cv=cv,random_state=a.seed,n_jobs=a.jobs,refit=True,error_score='raise')
            search.fit(X[train],y[train])
            fitted[name]=search.best_estimator_; cv_rmse=-search.best_score_; params=search.best_params_
            trials=pd.DataFrame(search.cv_results_); trials.to_csv(out/f'{name}_tuning.csv',index=False)
        else:
            cv_rmse=-cross_val_score(estimator,X[train],y[train],cv=cv,scoring='neg_root_mean_squared_error',n_jobs=a.jobs,error_score='raise').mean()
            estimator.fit(X[train],y[train]); fitted[name]=estimator; params={}
        cv_rows.append({'model':name,'training_CV_RMSE':cv_rmse,'fit_and_search_seconds':time.perf_counter()-start})
        best_parameters[name]=params
        print('Fitted:',name,flush=True)
    selected=min(cv_rows,key=lambda row:row['training_CV_RMSE'])['model']
    pd.DataFrame(cv_rows).sort_values('training_CV_RMSE').to_csv(out/'training_cv_comparison.csv',index=False)
    preds={name:est.predict(X[test]) for name,est in fitted.items()}
    pred_table=df.iloc[test].copy()
    for name,pred in preds.items(): pred_table[name+'_prediction']=pred
    pred_table.to_csv(out/'heldout_predictions.csv',index=False)
    # Cluster bootstrap: resample whole test groups, same samples for all models.
    rng=np.random.default_rng(a.seed+1); ug=np.unique(groups[test]); indices={g:np.flatnonzero(groups[test]==g) for g in ug}
    boot={name:[] for name in names}
    for _ in range(a.bootstrap):
        pick=np.concatenate([indices[g] for g in rng.choice(ug,size=len(ug),replace=True)])
        for name,pred in preds.items(): boot[name].append(metrics(y[test][pick],pred[pick]))
    rows=[]
    for name,pred in preds.items():
        row={'model':name,'selected_by_training_CV':name==selected,**metrics(y[test],pred)}
        for m in ['RMSE','MAE','R2','Spearman']:
            values=np.array([b[m] for b in boot[name]]); values=values[np.isfinite(values)]
            lo,hi=np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
            row[m+'_CI_low']=lo; row[m+'_CI_high']=hi
        rows.append(row)
    result=pd.DataFrame(rows)
    # Keep CV order: the test set is for reporting, not further model selection.
    result=result.set_index('model').loc[pd.DataFrame(cv_rows).sort_values('training_CV_RMSE').model].reset_index()
    result.to_csv(out/'heldout_metrics.csv',index=False)
    if 'MeanBaseline' in names:
        improvements=[]
        for name in names:
            if name=='MeanBaseline': continue
            diffs=np.array([b['RMSE']-m['RMSE'] for b,m in zip(boot['MeanBaseline'],boot[name])])
            improvements.append({'model':name,'mean_baseline_RMSE_minus_model_RMSE':metrics(y[test],preds['MeanBaseline'])['RMSE']-metrics(y[test],preds[name])['RMSE'],
                                 'paired_group_bootstrap_low':np.quantile(diffs,.025),'paired_group_bootstrap_high':np.quantile(diffs,.975)})
        pd.DataFrame(improvements).to_csv(out/'improvement_over_baseline.csv',index=False)
    sns.set_theme(style='whitegrid')
    fig,ax=plt.subplots(figsize=(8,4)); xp=np.arange(len(result))
    ax.bar(xp,result.RMSE,color='#458bb7'); ax.vlines(xp,result.RMSE_CI_low,result.RMSE_CI_high,color='black',lw=2)
    ax.set_xticks(xp,result.model,rotation=20); ax.set(ylabel='Held-out RMSE (lower is better)',title=('SIMULATED | ' if a.demo_label else '')+'Held-out target groups')
    fig.tight_layout(); fig.savefig(out/'model_comparison.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,len(names),figsize=(4*len(names),4),squeeze=False)
    for ax,(name,pred) in zip(axs[0],preds.items()):
        ax.scatter(y[test],pred,s=15,alpha=.65); lo=min(y[test].min(),pred.min()); hi=max(y[test].max(),pred.max()); ax.plot([lo,hi],[lo,hi],'k--')
        ax.set(title=name,xlabel='Observed activity',ylabel='Predicted activity')
    fig.tight_layout(); fig.savefig(out/'predicted_vs_observed.png',dpi=160); plt.close(fig)
    joblib.dump({'model':fitted[selected],'feature_names':feature_names,'sequence_length':a.sequence_length,'selection':'training grouped CV only','name':selected},out/'selected_model.joblib')
    report=vars(a).copy(); report.update({'selected_model':selected,'best_parameters':best_parameters,'training_groups':len(set(groups[train])),'test_groups':len(ug),'training_rows':len(train),'test_rows':len(test),'CV_folds':folds,'python':platform.python_version(),'mode':'SIMULATED' if a.demo_label else 'USER_DATA','input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'features':feature_names,'primary_metric':'Row-weighted RMSE','interval_scope':'95% cluster bootstrap intervals conditional on fitted models and observed held-out groups; not retraining uncertainty'})
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2))
    print('Selected by training CV:',selected,'| Results:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-demo'); p.add_argument('--input'); p.add_argument('--out',default='benchmark_results')
    p.add_argument('--models',default='MeanBaseline,Ridge,SVR,RandomForest,XGBoost')
    p.add_argument('--sequence-length',type=int,default=20); p.add_argument('--test-fraction',type=float,default=.2)
    p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--jobs',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--quick',action='store_true'); p.add_argument('--demo-label',action='store_true')
    a=p.parse_args()
    if a.make_demo: demo(a.make_demo)
    elif a.input: run(a)
    else: p.error('Provide --input or --make-demo')
if __name__=='__main__': main()
