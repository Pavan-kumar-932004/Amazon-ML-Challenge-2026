# Amazon ML Challenge 2026 - detailed parallel data analysis
# No argparse. Edit CONFIG below if needed.
import json, time, re, unicodedata, math
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# The terminal shown in VS Code is currently:
#   D:\Amazon-ML-Challenge-2026\dataset
# Therefore this script detects whether it is being run from
# the dataset directory or the project root.
HERE = Path.cwd()
DATASET = HERE if (HERE / 'train').exists() else HERE / 'dataset'
PROJECT_ROOT = DATASET.parent

TRAIN = DATASET / 'train'
TEST = DATASET / 'test'
OUT = PROJECT_ROOT / 'analysis' / 'detailed_parallel'
OUT.mkdir(parents=True, exist_ok=True)
MAX_WORKERS=3; CHUNK_SIZE=250_000; TOP_K=100
FILES={
 'train_source1':TRAIN/'train_source1.tsv','train_source2':TRAIN/'train_source2.tsv','train_source3':TRAIN/'train_source3.tsv',
 'test_source1':TEST/'test_source1.tsv','test_source2':TEST/'test_source2.tsv','test_source3':TEST/'test_source3.tsv'}

def s(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return ''
    return str(x)
def norm(x):
    x=unicodedata.normalize('NFKC',s(x)).casefold().replace('&',' and ')
    x=re.sub(r'[\u200b-\u200d\ufeff]','',x); x=re.sub(r'[^\w\s]',' ',x,flags=re.UNICODE); return re.sub(r'\s+',' ',x).strip()
def tokens(x): return norm(x).split()
def grams(x,n=3):
    x=re.sub(r'\W+','',norm(x),flags=re.UNICODE)
    return {x} if len(x)<n and x else {x[i:i+n] for i in range(len(x)-n+1)}
def script(x):
    c=Counter()
    for ch in s(x):
        if ch.isspace() or ch.isdigit() or unicodedata.category(ch).startswith('P'): continue
        u=unicodedata.name(ch,'')
        c['Latin' if 'LATIN' in u else 'Devanagari' if 'DEVANAGARI' in u else 'Arabic' if 'ARABIC' in u else 'Cyrillic' if 'CYRILLIC' in u else 'Greek' if 'GREEK' in u else 'Other']+=1
    return c.most_common(1)[0][0] if c else 'NoLetters'
def col(cols,names):
    d={str(x).lower():x for x in cols}
    for n in names:
        if n.lower() in d:return d[n.lower()]
    for x in cols:
        for n in names:
            if n.lower() in str(x).lower():return x
    return None
def cols(c):
    return {'id':col(c,['id','entity_id','business_id','source_id']),'name':col(c,['business_name','name','business']),'address':col(c,['business_address','address','location']),'country':col(c,['country','country_code','country_name'])}
def top(c,k=TOP_K): return [{'value':v,'count':int(n)} for v,n in c.most_common(k)]
def q(v):
    if not v:return {}
    x=pd.Series(v,dtype='float64'); return {k:(int(x.min()) if k in ('min','max') else float(getattr(x,k)() if k in ('mean','median') else x.quantile(float(k[1:])/100))) for k in ['min','p01','p05','p25','median','p75','p95','p99','max','mean']}

def analyze_source(job):
    name,path=job; path=Path(path); ck=OUT/f'{name}.json'
    if ck.exists():
        try:
            r=json.loads(ck.read_text(encoding='utf8')); r['_status']='checkpoint'; return r
        except Exception: pass
    t=time.time()
    if not path.exists(): return {'source':name,'error':'FILE_NOT_FOUND'}
    rows=0; ids=set(); names=set(); addrs=set(); nf=Counter(); af=Counter(); cf=Counter(); nn=Counter(); na=Counter(); nraw=defaultdict(set); araw=defaultdict(set); miss=Counter(); nl=[]; al=[]; ns=Counter(); ass=Counter(); columns=None; inferred=None
    for ch in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,na_filter=False,chunksize=CHUNK_SIZE,on_bad_lines='warn'):
        rows+=len(ch)
        if inferred is None: columns=list(ch.columns); inferred=cols(columns)
        if inferred['id']: ids.update(v for v in ch[inferred['id']] if v)
        if inferred['name']:
            for v in ch[inferred['name']]:
                v=s(v)
                if not v: miss['business_name']+=1; continue
                names.add(v); nf[v]+=1; z=norm(v); nn[z]+=1; nraw[z].add(v); nl.append(len(v)); ns[script(v)]+=1
        if inferred['address']:
            for v in ch[inferred['address']]:
                v=s(v)
                if not v: miss['business_address']+=1; continue
                addrs.add(v); af[v]+=1; z=norm(v); na[z]+=1; araw[z].add(v); al.append(len(v)); ass[script(v)]+=1
        if inferred['country']:
            for v in ch[inferred['country']]:
                v=s(v)
                if not v: miss['country']+=1
                else: cf[v]+=1
    r={'source':name,'path':str(path),'rows':rows,'columns':columns,'inferred_columns':inferred,'unique_ids':len(ids),'unique_raw_names':len(names),'unique_raw_addresses':len(addrs),'unique_normalized_names':len(nn),'unique_normalized_addresses':len(na),'missing':dict(miss),'country_distribution':top(cf,max(TOP_K,len(cf))),'name_length_stats':q(nl),'address_length_stats':q(al),'name_scripts':dict(ns),'address_scripts':dict(ass),'raw_name_repeated_values':sum(v>1 for v in nf.values()),'raw_address_repeated_values':sum(v>1 for v in af.values()),'normalized_name_repeated_values':sum(v>1 for v in nn.values()),'normalized_address_repeated_values':sum(v>1 for v in na.values()),'name_normalization_collision_values':sum(len(v)>1 for v in nraw.values()),'address_normalization_collision_values':sum(len(v)>1 for v in araw.values()),'generic_name_frequency':{f'>={k}':sum(v>=k for v in nf.values()) for k in [2,5,10,50,100]},'top_raw_names':top(nf),'top_raw_addresses':top(af),'top_normalized_names':top(nn),'top_normalized_addresses':top(na),'top_name_normalization_collisions':[{'normalized':z,'raw_variants':sorted(list(v))[:20],'variant_count':len(v),'frequency':nn[z]} for z,v in sorted(nraw.items(),key=lambda x:nn[x[0]],reverse=True)[:TOP_K]],'top_address_normalization_collisions':[{'normalized':z,'raw_variants':sorted(list(v))[:20],'variant_count':len(v),'frequency':na[z]} for z,v in sorted(araw.items(),key=lambda x:na[x[0]],reverse=True)[:TOP_K]],'elapsed_seconds':round(time.time()-t,2)}
    ck.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8'); return r

def load_gt():
    p=TRAIN/'train_ground_truth.tsv'
    if not p.exists(): return {'error':'GROUND_TRUTH_NOT_FOUND'}
    df=pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False)
    cs=list(df.columns); s1=col(cs,['source1_id','source_1_id','entity_id','id']) or cs[0]
    mc=[c for c in cs if any(x in str(c).lower() for x in ['source2','source_2','source3','source_3','match'])] or cs[1:]
    mp={}; mult=Counter(); links=0; s2=s3=0
    for _,r in df.iterrows():
        a=s(r[s1]).strip(); vals=[]
        for c in mc:
            v=s(r[c]).strip()
            if v:
                parts=[x.strip() for x in re.split(r'[|;,]',v) if x.strip()] if any(x in v for x in '|;,') else [v]
                vals += [(c,x) for x in parts]
        mp[a]=vals; mult[len(vals)]+=1; links+=len(vals)
        for c,_ in vals:
            lc=str(c).lower(); s2+=('source2' in lc or 'source_2' in lc); s3+=('source3' in lc or 'source_3' in lc)
    r={'path':str(p),'rows':len(df),'columns':cs,'source1_column':s1,'match_columns':mc,'unique_source1_entities':len(mp),'total_match_links':links,'source2_links':s2,'source3_links':s3,'multiplicity_distribution':dict(sorted(mult.items())),'zero_match_entities':mult[0],'one_to_one_entities':mult[1],'multi_match_entities':sum(v for k,v in mult.items() if k>=2),'_mapping':mp}
    (OUT/'ground_truth.json').write_text(json.dumps({k:v for k,v in r.items() if k!='_mapping'},ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'ground_truth_mapping.json').write_text(json.dumps({k:v for k,v in mp.items()},ensure_ascii=False),encoding='utf8'); return r

def lookup(source):
    p=FILES[source]; out={}
    for ch in pd.read_csv(p,sep='\t',dtype=str,keep_default_na=False,na_filter=False,chunksize=CHUNK_SIZE):
        c=cols(ch.columns)
        if not c['id']:continue
        for _,r in ch.iterrows(): out[s(r[c['id']])]={k:(s(r[c[k]]) if c[k] else '') for k in ['name','address','country']}
    return out
def jac(a,b):
    a=set(tokens(a));b=set(tokens(b)); return 1.0 if not a and not b else 0.0 if not a or not b else len(a&b)/len(a|b)
def pairdiag(a,b):
    an,bn=norm(a['name']),norm(b['name']); aa,ba=norm(a['address']),norm(b['address'])
    return {'name_exact_raw':bool(a['name'] and a['name']==b['name']),'name_exact_normalized':bool(an and an==bn),'address_exact_raw':bool(a['address'] and a['address']==b['address']),'address_exact_normalized':bool(aa and aa==ba),'country_exact':bool(a['country'] and b['country'] and norm(a['country'])==norm(b['country'])),'name_token_jaccard':jac(an,bn),'address_token_jaccard':jac(aa,ba),'name_char3_jaccard':jac(' '.join(grams(an)), ' '.join(grams(bn))),'address_char3_jaccard':jac(' '.join(grams(aa)), ' '.join(grams(ba)))}
def positive(gt):
    s1=lookup('train_source1');s2=lookup('train_source2');s3=lookup('train_source3'); mp=gt['_mapping']; n=Counter(); sums=Counter(); ex=[]; missing=Counter()
    for i,ms in mp.items():
        if i not in s1: missing['s1']+=1; continue
        for c,j in ms:
            lc=str(c).lower(); target=(s2.get(j), 'source2') if ('source2' in lc or 'source_2' in lc) else (s3.get(j),'source3') if ('source3' in lc or 'source_3' in lc) else ((s2.get(j),'source2') if j in s2 else (s3.get(j),'source3'))
            b,src=target
            if not b: missing['target']+=1; continue
            d=pairdiag(s1[i],b); n['total']+=1
            for k in ['name_exact_raw','name_exact_normalized','address_exact_raw','address_exact_normalized','country_exact']: n[k]+=d[k]
            for k in ['name_token_jaccard','address_token_jaccard','name_char3_jaccard','address_char3_jaccard']: sums[k]+=d[k]
            if len(ex)<100: ex.append({'s1_id':i,'target_id':j,'source':src,'s1':s1[i],'target':b,'diagnostics':d})
    r={'total_links':n['total'],'rates':{k:n[k]/n['total'] for k in ['name_exact_raw','name_exact_normalized','address_exact_raw','address_exact_normalized','country_exact']} if n['total'] else {},'mean_similarities':{k:sums[k]/n['total'] for k in sums} if n['total'] else {},'missing_ids':dict(missing),'examples':ex}
    (OUT/'positive_pair_analysis.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8'); return r

def blocking(gt):
    s1=lookup('train_source1');s2=lookup('train_source2');s3=lookup('train_source3');mp=gt['_mapping']; rules=['same_country','exact_raw_name','exact_normalized_name','exact_raw_address','exact_normalized_address','name_or_address_normalized','name_and_country','address_and_country','name_and_address_normalized']; hit=Counter(); src=defaultdict(Counter); total=Counter()
    for i,ms in mp.items():
        a=s1.get(i)
        if not a:continue
        for c,j in ms:
            lc=str(c).lower(); source='source2' if ('source2' in lc or 'source_2' in lc) else 'source3' if ('source3' in lc or 'source_3' in lc) else ('source2' if j in s2 else 'source3'); b=(s2 if source=='source2' else s3).get(j)
            if not b:continue
            total['all']+=1;total[source]+=1; an,bn=norm(a['name']),norm(b['name']); aa,ba=norm(a['address']),norm(b['address']); same=bool(a['country'] and b['country'] and norm(a['country'])==norm(b['country'])); rn=bool(a['name'] and a['name']==b['name']); nn=bool(an and an==bn); ra=bool(a['address'] and a['address']==b['address']); na=bool(aa and aa==ba)
            vals={'same_country':same,'exact_raw_name':rn,'exact_normalized_name':nn,'exact_raw_address':ra,'exact_normalized_address':na,'name_or_address_normalized':nn or na,'name_and_country':nn and same,'address_and_country':na and same,'name_and_address_normalized':nn and na}
            for r,v in vals.items():
                if v:hit[r]+=1;src[source][r]+=1
    r={'total_evaluable_true_links':total['all'],'overall_recall':{x:hit[x]/total['all'] if total['all'] else 0 for x in rules},'source2_recall':{x:src['source2'][x]/total['source2'] if total['source2'] else 0 for x in rules},'source3_recall':{x:src['source3'][x]/total['source3'] if total['source3'] else 0 for x in rules},'hit_counts':dict(hit),'source_totals':dict(total)}
    (OUT/'blocking_recall.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8'); return r

def report(sr,gt,p,b):
    L=['='*90,'AMAZON ML CHALLENGE 2026 — DETAILED DATA ANALYSIS','='*90,'','SOURCE OVERVIEW','-'*90]
    for n in sorted(sr):
        r=sr[n];L.append(f"{n}: {r.get('rows',0):,} rows | {r.get('unique_ids',0):,} IDs | {r.get('unique_raw_names',0):,} names | {r.get('unique_raw_addresses',0):,} addresses")
    L += ['','MISSINGNESS','-'*90]
    for n,r in sorted(sr.items()):
        rows=max(1,r.get('rows',1));L.append(n+': '+', '.join(f'{k}={v:,} ({v/rows:.2%})' for k,v in r.get('missing',{}).items()))
    L += ['','GROUND TRUTH','-'*90]
    for k in ['rows','unique_source1_entities','total_match_links','zero_match_entities','one_to_one_entities','multi_match_entities','multiplicity_distribution']:L.append(f'{k}: {gt.get(k)}')
    L += ['','TRUE-PAIR DIAGNOSTICS','-'*90]
    L.append(f"links: {p.get('total_links',0):,}");L += [f'{k}: {v:.4%}' for k,v in p.get('rates',{}).items()];L += [f'{k}: {v:.4f}' for k,v in p.get('mean_similarities',{}).items()]
    L += ['','BLOCKING RECALL','-'*90];L += [f'{k:<35} {v:.4%}' for k,v in b.get('overall_recall',{}).items()];L += ['','SOURCE 2'];L += [f'{k:<35} {v:.4%}' for k,v in b.get('source2_recall',{}).items()];L += ['','SOURCE 3'];L += [f'{k:<35} {v:.4%}' for k,v in b.get('source3_recall',{}).items()]
    L += ['', '='*90];pth=OUT/'DETAILED_ANALYSIS_REPORT.txt';pth.write_text('\n'.join(L),encoding='utf8');return pth

def main():
    t=time.time();print(f'Working directory: {HERE}\nDataset: {DATASET}\nOutput: {OUT}\nWorkers: {MAX_WORKERS}\nChunk: {CHUNK_SIZE:,}\n')
    sr={}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs={ex.submit(analyze_source,(n,str(p))):n for n,p in FILES.items()}
        for f in as_completed(fs):
            n=fs[f]
            try:r=f.result();sr[n]=r;print(f"DONE {n:<18} {r.get('rows',0):>12,} rows ({r.get('elapsed_seconds',0):.1f}s)")
            except Exception as e:sr[n]={'error':repr(e)};print('ERROR',n,e)
    (OUT/'source_summary.json').write_text(json.dumps(sr,ensure_ascii=False,indent=2),encoding='utf8')
    print('\nGround truth...');gt=load_gt();print('links:',gt.get('total_match_links',0))
    print('Positive pairs...');p=positive(gt)
    print('Blocking recall...');b=blocking(gt)
    for k,v in b['overall_recall'].items():print(f'  {k:<35}{v:.4%}')
    rp=report(sr,gt,p,b)
    final={'runtime_seconds':round(time.time()-t,2),'output_directory':str(OUT),'sources':sr,'ground_truth':{k:v for k,v in gt.items() if k!='_mapping'},'positive_pairs':p,'blocking':b,'report':str(rp)}
    (OUT/'FINAL_ANALYSIS.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf8');print(f'\nCOMPLETE in {final["runtime_seconds"]:.1f}s\nReport: {rp}')
if __name__=='__main__':main()
