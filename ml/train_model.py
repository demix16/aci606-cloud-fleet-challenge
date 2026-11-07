"""EDA + clasificación reproducible. Train/val/test separados por fleet_id."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['speed','battery','motor_temp','vibration','motor_current','obstacle_distance']
CLASSES=['NORMAL','WARNING','CRITICAL']

def main():
    path=ROOT/'data/training.csv'
    if not path.exists():
        raise SystemExit('Primero: python -m ml.build_fleet_dataset')
    df=pd.read_csv(path)
    required=set(FEATURES+['fleet_id','status'])
    if not required.issubset(df.columns):
        raise SystemExit(f'Faltan columnas: {required-set(df.columns)}')
    if df.fleet_id.nunique()<10: raise SystemExit('Se requieren al menos 10 flotas')
    if not set(CLASSES).issubset(set(df.status)):
        raise SystemExit('Faltan clases. Generen más flotas/datos.')
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    print('Dimensiones',df.shape)
    print('Tipos:',df.dtypes.to_dict())
    print('Nulos:',df.isna().sum().to_dict())
    print('Distribución:',df.status.value_counts().to_dict())
    print('Flotas:',df.fleet_id.nunique())
    df.status.value_counts().reindex(CLASSES,fill_value=0).plot(kind='bar',title='Distribución de clases')
    plt.ylabel('Muestras');plt.tight_layout();plt.savefig(out/'clases.png',dpi=140);plt.close()
    df[FEATURES].hist(figsize=(11,7),bins=30)
    plt.tight_layout();plt.savefig(out/'eda_variables.png',dpi=140);plt.close()
    corr=df[FEATURES].corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(FEATURES)),FEATURES,rotation=45,ha='right')
    ax.set_yticks(range(len(FEATURES)),FEATURES)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'correlacion.png',dpi=140);plt.close(fig)

    train_ix,test_ix=next(GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=606).split(df,df.status,groups=df.fleet_id))
    trainpool=df.iloc[train_ix].copy();test=df.iloc[test_ix].copy()
    subtrain_ix,val_ix=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=607).split(trainpool,trainpool.status,groups=trainpool.fleet_id))
    subtrain=trainpool.iloc[subtrain_ix]; val=trainpool.iloc[val_ix]
    assert set(subtrain.fleet_id).isdisjoint(val.fleet_id)
    assert set(trainpool.fleet_id).isdisjoint(test.fleet_id)
    print('TRAIN/VAL/TEST:',len(subtrain),len(val),len(test))
    print('Flotas train:',sorted(subtrain.fleet_id.unique()))
    print('Flotas validación:',sorted(val.fleet_id.unique()))
    print('Flotas test:',sorted(test.fleet_id.unique()))

    candidates={
        'LogisticRegression':lambda:Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler()),
            ('classifier',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=606))]),
        'RandomForest':lambda:Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('classifier',RandomForestClassifier(n_estimators=180,min_samples_leaf=3,
                                                class_weight='balanced_subsample',
                                                random_state=606,n_jobs=-1))]),
    }
    scores={}
    for name,make in candidates.items():
        model=make();model.fit(subtrain[FEATURES],subtrain.status)
        ypred=model.predict(val[FEATURES]);score=f1_score(val.status,ypred,labels=CLASSES,average='macro',zero_division=0)
        scores[name]=float(score);print(f'F1 macro VALIDACIÓN {name}: {score:.4f}')
    best_name=max(scores,key=scores.get)
    model=candidates[best_name](); model.fit(trainpool[FEATURES],trainpool.status)
    test_pred=model.predict(test[FEATURES])
    test_f1=float(f1_score(test.status,test_pred,labels=CLASSES,average='macro',zero_division=0))
    baseline=DummyClassifier(strategy='most_frequent')
    baseline.fit(trainpool[FEATURES],trainpool.status)
    baseline_f1=float(f1_score(test.status,baseline.predict(test[FEATURES]),labels=CLASSES,average='macro',zero_division=0))
    print('Modelo elegido:',best_name)
    print('F1 macro TEST:',round(test_f1,4),'Base mayoritaria:',round(baseline_f1,4))
    print(classification_report(test.status,test_pred,labels=CLASSES,zero_division=0))
    print('Matriz confusión:\n',confusion_matrix(test.status,test_pred,labels=CLASSES))
    disp=ConfusionMatrixDisplay.from_predictions(test.status,test_pred,labels=CLASSES,cmap='Blues',values_format='d')
    disp.figure_.tight_layout();disp.figure_.savefig(out/'matriz_confusion.png',dpi=150);plt.close(disp.figure_)
    meta={
        'model_version':'v1', 'chosen_model':best_name, 'features':FEATURES,
        'classes':CLASSES, 'validation_macro_f1':scores, 'test_macro_f1':test_f1,
        'baseline_macro_f1':baseline_f1, 'rows':len(df),
        'train_fleets':sorted(trainpool.fleet_id.unique().tolist()),
        'test_fleets':sorted(test.fleet_id.unique().tolist()),
        'seed':606,
    }
    (out/'metricas.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    models=ROOT/'models';models.mkdir(exist_ok=True)
    joblib.dump({'pipeline':model,'features':FEATURES,'model_version':'v1'},models/'model.joblib')
    print('Artefacto generado:',models/'model.joblib')
if __name__=='__main__':main()
