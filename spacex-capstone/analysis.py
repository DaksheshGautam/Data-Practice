"""Reproducible IBM Applied Data Science Capstone analysis.
Run: python analysis.py. Historical course snapshots are included in data/.
"""
from pathlib import Path
import json, sqlite3, io, contextlib, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import folium
from folium.plugins import MarkerCluster, MousePosition
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})

def collect_api():
    if not (DATA/'API_call_spacex_api.json').exists():
        import urllib.request
        url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
        # GET the course archive of the SpaceX v4 launches/past response.
        (DATA/'API_call_spacex_api.json').write_bytes(urllib.request.urlopen(url,timeout=60).read())
    raw=json.loads((DATA/'API_call_spacex_api.json').read_text())
    rows=[]
    for r in raw:
        core=(r.get('cores') or [{}])[0]
        rows.append({'APIFlightNumber':r['flight_number'],'Date':r['date_utc'][:10],
          'Name':r['name'],'RocketId':r['rocket'],'LaunchpadId':r['launchpad'],
          'PayloadIds':','.join(r['payloads']), 'LandingSuccess':core.get('landing_success'),
          'LandingType':core.get('landing_type'),'CoreId':core.get('core'),
          'Reused':core.get('reused'),'Flight':core.get('flight')})
    api=pd.DataFrame(rows); api.to_csv(DATA/'api_launches.csv',index=False)
    print('SpaceX API historical snapshot:',len(api),'launches')
    print(api.head().to_string(index=False))
    return api

def scrape():
    if not (DATA/'wikipedia_launches.html').exists():
        import urllib.request
        url='https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches_%282010%E2%80%932019%29'
        request=urllib.request.Request(url,headers={'User-Agent':'EducationalCapstone/1.0'})
        (DATA/'wikipedia_launches.html').write_bytes(urllib.request.urlopen(request,timeout=60).read())
    soup=BeautifulSoup((DATA/'wikipedia_launches.html').read_text(encoding='utf-8'),'html.parser')
    for ref in soup.select('sup.reference'): ref.decompose()
    rows=[]
    for ti,table in enumerate(soup.select('table.wikitable')):
        headers=[x.get_text(' ',strip=True) for x in table.select('tr')[:1] for x in x.select('th')]
        if not any('Flight' in h or 'Date' in h for h in headers): continue
        for tr in table.select('tr')[1:]:
            cells=tr.find_all(['td','th'],recursive=False)
            if len(cells)>=10:
                vals=[c.get_text(' ',strip=True) for c in cells]
                rows.append({'Table':ti,'Flight':vals[0],'Date':vals[1],
                    'Booster':vals[2],'LaunchSite':vals[3],'Payload':vals[4],
                    'PayloadMass':vals[5],'Orbit':vals[6],
                    'Customer':vals[7] if len(vals)>7 else '',
                    'LaunchOutcome':vals[8] if len(vals)>8 else '',
                    'LandingOutcome':vals[9] if len(vals)>9 else ''})
    out=pd.DataFrame(rows)
    assert len(out)>70, 'Launch-table extraction failed'
    out.to_csv(DATA/'web_scraped.csv',index=False)
    print('Wikipedia snapshot: extracted',len(out),'launch rows using BeautifulSoup')
    print(out.head().to_string(index=False)); return out

def wrangle():
    df=pd.read_csv(DATA/'dataset_part_1.csv')
    bad={'False ASDS','False Ocean','False RTLS','None ASDS','None None'}
    df['Class']=(~df.Outcome.isin(bad)).astype(int)
    missing=df.isna().sum(); df['PayloadMass']=df.PayloadMass.fillna(df.PayloadMass.mean())
    assert len(df)==90 and df.FlightNumber.is_unique
    assert np.array_equal(df.Class,pd.read_csv(DATA/'dataset_part_2.csv').Class)
    df.to_csv(DATA/'analysis_dataset.csv',index=False)
    print('Published course API-derived checkpoint: 90 Falcon 9 launches')
    print('Date range:',df.Date.min(),'to',df.Date.max())
    print('Missing values before processing:\n',missing[missing>0].to_string())
    print('Outcome counts:\n',df.Outcome.value_counts().to_string())
    print('Class counts:',df.Class.value_counts().to_dict())
    return df

def saveplot(name):
    plt.tight_layout(); plt.savefig(RESULTS/(name+'.png'),dpi=170,bbox_inches='tight'); plt.close()

def eda(df):
    specs=[('FlightNumber','LaunchSite','flight_site'),('PayloadMass','LaunchSite','payload_site'),
           ('FlightNumber','Orbit','flight_orbit'),('PayloadMass','Orbit','payload_orbit')]
    for x,y,name in specs:
        fig,ax=plt.subplots(figsize=(9,4.4))
        for label,color in [(0,'#d97732'),(1,'#087f8c')]:
            q=df[df.Class==label]; ax.scatter(q[x],q[y],s=45,alpha=.78,c=color,label='Failure / no recovery' if label==0 else 'Success')
        ax.set_xlabel('Payload mass (kg)' if x=='PayloadMass' else 'Flight number'); ax.set_ylabel('Launch site' if y=='LaunchSite' else 'Orbit')
        ax.legend(loc='best',fontsize=10); saveplot(name)
    orbit=df.groupby('Orbit').Class.agg(['mean','count','sum']).sort_values('mean')
    fig,ax=plt.subplots(figsize=(9,4.4)); ax.bar(orbit.index,orbit['mean']*100,color='#087f8c')
    for i,(_,r) in enumerate(orbit.iterrows()): ax.text(i,r['mean']*100+2,f"n={int(r['count'])}",ha='center',fontsize=10)
    ax.set_ylim(0,115); ax.set_ylabel('Landing success (%)'); saveplot('orbit_rate')
    yearly=df.assign(Year=pd.to_datetime(df.Date).dt.year).groupby('Year').Class.agg(['mean','count','sum'])
    fig,ax=plt.subplots(figsize=(9,4.4)); ax.plot(yearly.index,yearly['mean']*100,'o-',color='#087f8c',linewidth=2)
    ax.set_ylim(0,105); ax.set_ylabel('Landing success (%)'); ax.set_xlabel('Year'); ax.grid(alpha=.2); saveplot('year_rate')
    site=df.groupby('LaunchSite').Class.agg(['mean','count','sum'])
    for name,table in [('orbit',orbit),('year',yearly),('site',site)]: table.to_csv(RESULTS/(name+'_summary.csv'))
    print('Launch-site success:\n',site.to_string()); print('Orbit success:\n',orbit.to_string()); print('Yearly success:\n',yearly.to_string())
    return site,orbit,yearly

SQL_QUERIES={
 '01_unique_sites':'SELECT DISTINCT Launch_Site FROM SPACEX ORDER BY Launch_Site',
 '02_cca_records':"SELECT Date, Booster_Version, Launch_Site FROM SPACEX WHERE Launch_Site LIKE 'CCA%' ORDER BY Date LIMIT 5",
 '03_nasa_payload':"SELECT SUM(PAYLOAD_MASS__KG_) AS TotalPayloadKg FROM SPACEX WHERE Customer LIKE '%NASA%'",
 '04_v11_average':"SELECT AVG(PAYLOAD_MASS__KG_) AS AveragePayloadKg FROM SPACEX WHERE Booster_Version='F9 v1.1'",
 '05_first_ground':"SELECT MIN(Date) AS FirstGroundLanding FROM SPACEX WHERE Landing_Outcome='Success (ground pad)'",
 '06_drone_payload':"SELECT Date,Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEX WHERE Landing_Outcome='Success (drone ship)' AND PAYLOAD_MASS__KG_>4000 AND PAYLOAD_MASS__KG_<6000 ORDER BY Date",
 '07_mission_outcomes':"SELECT Mission_Outcome,COUNT(*) AS Launches FROM SPACEX GROUP BY Mission_Outcome",
 '08_max_payload':"SELECT Date,Booster_Version,PAYLOAD_MASS__KG_ FROM SPACEX WHERE PAYLOAD_MASS__KG_=(SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEX)",
 '09_2015_failures':"SELECT Date,Booster_Version,Launch_Site FROM SPACEX WHERE Date LIKE '2015%' AND Landing_Outcome='Failure (drone ship)'",
 '10_rank_outcomes':"SELECT Landing_Outcome,COUNT(*) AS Launches FROM SPACEX WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Landing_Outcome ORDER BY Launches DESC"
}
def sql_analysis():
    df=pd.read_csv(DATA/'Spacex.csv'); df['Date']=pd.to_datetime(df.Date).dt.strftime('%Y-%m-%d')
    for col in df.select_dtypes(include=['object','str']).columns: df[col]=df[col].str.strip()
    with sqlite3.connect(RESULTS/'spacex.sqlite') as conn:
        df.to_sql('SPACEX',conn,if_exists='replace',index=False)
        outputs={}
        for name,query in SQL_QUERIES.items():
            result=pd.read_sql_query(query,conn); result.to_csv(RESULTS/(name+'.csv'),index=False)
            outputs[name]=result.to_dict('records'); print(name, '\n',query,'\n',result.to_string(index=False),'\n')
    (ROOT/'queries.sql').write_text('\n\n'.join('-- '+k+'\n'+v+';' for k,v in SQL_QUERIES.items()))
    return outputs

def map_analysis():
    df=pd.read_csv(DATA/'spacex_launch_geo.csv'); m=folium.Map(location=[30,-90],zoom_start=4,tiles='OpenStreetMap')
    sites=df.groupby('Launch Site')[['Lat','Long']].first()
    for site,r in sites.iterrows():
        folium.Circle([r.Lat,r.Long],radius=1000,color='#087f8c',fill=True,popup=site).add_to(m)
        folium.Marker([r.Lat,r.Long],tooltip=site,popup=site).add_to(m)
    cluster=MarkerCluster(name='Launch outcomes').add_to(m)
    for _,r in df.iterrows():
        folium.Marker([r.Lat,r.Long],popup=f"Flight {r['Flight Number']} | {r['Date']} | class={r['class']}",icon=folium.Icon(color='green' if r['class']==1 else 'red')).add_to(cluster)
    MousePosition(position='topright',separator=' | ',prefix='Latitude / longitude').add_to(m)
    folium.LayerControl().add_to(m); m.save(ROOT/'launch_sites_map.html')
    # Nearest points on mapped OSM line segments, using a local planar projection.
    site=sites.loc['CCAFS SLC-40']; import math, xml.etree.ElementTree as ET
    tree=ET.parse(DATA/'osm_proximity.xml').getroot()
    nodes={n.attrib['id']:(float(n.attrib['lat']),float(n.attrib['lon'])) for n in tree.findall('node')}
    nearest={}; feature_meta={}; lat0=float(site.Lat); lon0=float(site.Long); cosine=math.cos(math.radians(lat0))
    for way in tree.findall('way'):
        tags={t.attrib['k']:t.attrib['v'] for t in way.findall('tag')}
        label='Coastline' if tags.get('natural')=='coastline' else ('Railway' if tags.get('railway')=='rail' else ('Road' if tags.get('highway') in ['primary','secondary','tertiary','unclassified','residential'] else None))
        if not label:continue
        pts=[nodes[n.attrib['ref']] for n in way.findall('nd') if n.attrib['ref'] in nodes]
        for (la,lo),(lb,lob) in zip(pts,pts[1:]):
            ax=(lo-lon0)*cosine;ay=la-lat0;bx=(lob-lon0)*cosine;by=lb-lat0
            dx=bx-ax;dy=by-ay; den=dx*dx+dy*dy
            t=max(0,min(1,-(ax*dx+ay*dy)/den)) if den else 0
            x=ax+t*dx;y=ay+t*dy;dist=x*x+y*y
            if label not in nearest or dist<nearest[label][0]:
                nearest[label]=(dist,(lat0+y,lon0+x/cosine)); feature_meta[label]={'way_id':way.attrib['id'],'tags':tags}
    assert len(nearest)==3, 'Missing proximity feature geometry'
    refs={label:point for label,(_,point) in nearest.items()}
    (RESULTS/'proximity_features.json').write_text(json.dumps(feature_meta,indent=2))
    detail=folium.Map([site.Lat,site.Long],zoom_start=14,tiles='OpenStreetMap')
    distances={}
    for label,(lat,lon) in refs.items():
        p1,p2=math.radians(site.Lat),math.radians(lat); dlat=p2-p1; dlon=math.radians(lon-site.Long)
        km=6371*2*math.asin(math.sqrt(math.sin(dlat/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2))
        distances[label]=round(km,3)
        folium.PolyLine([[site.Lat,site.Long],[lat,lon]],tooltip=f'{label}: {km:.3f} km',color='blue').add_to(detail)
        folium.Marker([lat,lon],popup=f'{label}: {km:.3f} km straight line').add_to(detail)
    folium.Marker([site.Lat,site.Long],popup='CCAFS SLC-40').add_to(detail); detail.save(ROOT/'site_proximity_map.html')
    print(sites.to_string()); print('Nearest OSM feature straight-line distances (km):',distances)
    return distances

def prediction(df):
    # Outcomes and landing pad are excluded: they would reveal post-launch information.
    numeric=['FlightNumber','PayloadMass','Flights','Block','ReusedCount','GridFins','Reused','Legs']
    categorical=['Orbit','LaunchSite']
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    X=df[numeric+categorical]; y=df.Class
    train,test=train_test_split(np.arange(len(df)),test_size=.2,random_state=2,stratify=y)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    configs={
      'Logistic regression':(LogisticRegression(max_iter=3000),{'model__C':[.01,.1,1,10]}),
      'SVM':(SVC(),{'model__C':[.1,1,10],'model__kernel':['linear','rbf'],'model__gamma':['scale','auto']}),
      'Decision tree':(DecisionTreeClassifier(random_state=42),{'model__max_depth':[2,3,4,6],'model__min_samples_leaf':[1,2,4]}),
      'KNN':(KNeighborsClassifier(),{'model__n_neighbors':[3,5,7,9],'model__weights':['uniform','distance']})}
    rows=[]; models={}
    for name,(model,grid) in configs.items():
        search=GridSearchCV(Pipeline([('pre',pre),('model',model)]),grid,cv=cv,scoring='accuracy',n_jobs=1)
        search.fit(X.iloc[train],y.iloc[train]); pred=search.predict(X.iloc[test]); models[name]=search
        rows.append({'Model':name,'CVAccuracy':search.best_score_,'TestAccuracy':accuracy_score(y.iloc[test],pred),'BalancedAccuracy':balanced_accuracy_score(y.iloc[test],pred),'Parameters':str(search.best_params_)})
    res=pd.DataFrame(rows); res.to_csv(RESULTS/'model_results.csv',index=False)
    best=res.sort_values('CVAccuracy',ascending=False).iloc[0].Model
    cm=confusion_matrix(y.iloc[test],models[best].predict(X.iloc[test]),labels=[0,1])
    fig,ax=plt.subplots(figsize=(8.8,4.4)); ix=np.arange(len(res)); ax.bar(ix-.18,res.CVAccuracy,.36,label='5-fold CV',color='#087f8c'); ax.bar(ix+.18,res.TestAccuracy,.36,label='Test',color='#e6a04b'); ax.set_xticks(ix,res.Model); ax.set_ylim(0,1.05); ax.set_ylabel('Accuracy'); ax.legend(); saveplot('model_accuracy')
    fig,ax=plt.subplots(figsize=(6,4.4)); ax.imshow(cm,cmap='Blues'); ax.set_xticks([0,1],['Failure / no recovery','Success']); ax.set_yticks([0,1],['Failure / no recovery','Success']); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=26,color='white' if cm[i,j]>cm.max()/2 else 'black')
    saveplot('confusion_matrix')
    # Time split is a robustness check for changing launch technology.
    order=np.argsort(pd.to_datetime(df.Date)); old=order[:72]; recent=order[72:]
    temporal=models[best].best_estimator_; temporal.fit(X.iloc[old],y.iloc[old]); timeacc=accuracy_score(y.iloc[recent],temporal.predict(X.iloc[recent]))
    info={'best_model':best,'confusion_matrix':cm.tolist(),'test_size':len(test),'train_size':len(train),'baseline_test':float(y.iloc[test].mean()),'temporal_accuracy':timeacc,'temporal_baseline':float(y.iloc[recent].mean()),'test_indices':test.tolist()}
    print(res.to_string(index=False)); print(info); return info

def dashboard_evidence():
    df=pd.read_csv(DATA/'spacex_launch_dash.csv'); stats=df.groupby('Launch Site')['class'].agg(['sum','count','mean'])
    fig,ax=plt.subplots(figsize=(8.8,4.4)); ax.pie(stats['sum'],labels=stats.index,autopct='%1.1f%%',colors=['#087f8c','#54a6a6','#e6a04b','#738295']); saveplot('dash_all')
    best=stats['mean'].idxmax(); q=df[df['Launch Site']==best]
    fig,ax=plt.subplots(figsize=(8.8,4.4)); ax.pie([q['class'].sum(),len(q)-q['class'].sum()],labels=['Success','Failure'],autopct='%1.1f%%',colors=['#087f8c','#e6a04b']); saveplot('dash_best')
    for name,lo,hi in [('dash_payload',0,10000),('dash_payload_filtered',2000,6000)]:
        q=df[df['Payload Mass (kg)'].between(lo,hi)]
        fig,ax=plt.subplots(figsize=(8.8,4.4))
        for version,r in q.groupby('Booster Version Category'): ax.scatter(r['Payload Mass (kg)'],r['class'],label=version,alpha=.8)
        ax.set_xlabel('Payload mass (kg)'); ax.set_ylabel('Landing class'); ax.set_yticks([0,1]); ax.legend(fontsize=10); saveplot(name)
    print(stats.to_string()); return {'best_site':best,'stats':stats.reset_index().to_dict('records')}

def main():
    collect_api(); scrape(); df=wrangle(); eda(df); sql=sql_analysis(); distances=map_analysis(); ml=prediction(df); dash=dashboard_evidence()
    result={'ml':ml,'dash':dash,'sql':sql,'distances':distances,'n':len(df),'successes':int(df.Class.sum()),'date_min':df.Date.min(),'date_max':df.Date.max()}
    (RESULTS/'summary.json').write_text(json.dumps(result,indent=2)); return result

if __name__=='__main__': main()
