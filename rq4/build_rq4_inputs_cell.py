"""Extraction of rq4_inputs.npz (per-learner skill, O1 trace features, corrected snapshot features and gain counts).
Recorded for transparency: this cell was executed interactively after `run_all.py analyse` and relies on objects from that
session (recs, eps_by, rows from cptrace.study.episode_strategy_table, oracle_tv, grid_for, T_END). The resulting
rq4_inputs.npz is supplied and is the input of rq4_run.py; it is not regenerated automatically."""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
DAY=86400.
keys=["n_reads","n_reads_path","n_writes","n_writes_off","n_deletes","recreate","n_errors","n_fam","dur","world_open"]
XO1=np.array([[r["f1"][k] for k in keys]+[r["cli"]*1.0] for r in rows],float); XO1=np.column_stack([XO1,np.log1p(np.abs(XO1))])
y3=np.array([r['label3'] for r in rows]); grp=[r['sid'] for r in rows]
oof=np.empty(len(rows),dtype=object); oofp=np.zeros(len(rows))
for tr,te in GroupKFold(n_splits=5).split(XO1,y3,grp):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(XO1[tr],y3[tr])
    oof[te]=m.predict(XO1[te]); oofp[te]=m.predict_proba(XO1[te])[:,list(m.classes_).index('H')]
for i,x in enumerate(rows): x['clf_H']=oof[i]=='H'; x['clf_pH']=oofp[i]; x['rule_H']=A.rule_strategy(x['f1'],'O1')=='H'
def snap_feats(r, horizon_days=1.0, dmin=60):
    t,v=oracle_tv[r['sid']]
    dsec=dmin*60.; g=grid_for(r,t,dsec); sv=A.snapshot_vectors(t,v,g)
    sigs=[s for _,_,s in r['oracle']]; idx=np.searchsorted(np.asarray(t),g,side='right')-1
    ev=[e for e in r['events'] if e['eventCategory']=='Management']
    h=A.ts(ev[0]['eventTime'])+horizon_days*DAY
    hat=[e for e in A.episodes(list(g),sv,T_END,codes=INV.FUNCTIONAL)]
    E=[e for e in hat if e['t0']<=h]
    pers=sum(min(e['t1'],h)-e['t0'] for e in E)/3600
    churn=0
    for e in E:
        if e['t1']>h: continue
        ks=[k for k in range(len(g)) if e['t0']<=g[k]<=e['t1'] and idx[k]>=0]
        if len(ks)>=2:
            s0,s1=sigs[idx[ks[0]]],sigs[idx[ks[-1]]]
            churn+=int(any(set(s0[kk])!=set(s1[kk]) and s1[kk] for kk in s0))
    return [len(E), math.log1p(pers), churn]
t0=time.time()
FS=np.array([snap_feats(r) for r in recs]); print('FS',time.time()-t0, FS.mean(0), FS_old.mean(0))
def h2_feats(r,horizon_days=7.0):
    ev=[e for e in r['events'] if e['eventCategory']=='Management']
    h=A.ts(ev[0]['eventTime'])+horizon_days*DAY
    R_=[x for x in rows_by[r['sid']] if x['t1']<=h]
    n=len(R_)
    return [n, np.mean([x['rule_H'] for x in R_]) if n else np.nan, np.mean([x['clf_H'] for x in R_]) if n else np.nan,
            np.mean([x['clf_pH'] for x in R_]) if n else np.nan, np.mean([x['label']=='H' for x in R_]) if n else np.nan,
            np.mean([x['label3']=='H' for x in R_]) if n else np.nan]
H2=np.array([h2_feats(r) for r in recs])
print('H2 eligible', np.mean(H2[:,0]>0), np.nanmean(H2[:,1:],0))
theta=np.array([r['latent']['theta'] for r in recs]); cli=np.array([r['latent']['cli'] for r in recs],float)
gainHhelp=np.array([sum(1 for e in eps_by[r['sid']] if e['code'].startswith('F') and e['label'] in ('H','help')) for r in recs],float)
gainH=np.array([sum(1 for e in eps_by[r['sid']] if e['code'].startswith('F') and e['label']=='H') for r in recs],float)
np.savez('revision/rq4_inputs.npz', sid=np.array([r['sid'] for r in recs]), theta=theta, cli=cli, gain_H_help=gainHhelp, gain_H=gainH,
         F1=F1, FS_corrected=FS, FS_old=FS_old, H2=H2,
         F1_names=np.array(["n_functional_episodes","log_persistence_h","share_rule_H","log_api_errors","read_write_ratio","n_recreate"]),
         FS_names=np.array(["n_detected_functional_episodes","log_detected_persistence_h","n_observed_recreate"]),
         H2_names=np.array(["n_resolved_week1","share_rule_H","share_clfH_hard","share_clfH_prob","share_true_H","share_true_H_or_help"]))
print('saved')
