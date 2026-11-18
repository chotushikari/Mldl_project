"""Export only completed, measured experiments for the future website."""
import csv
import json
from pathlib import Path

root=Path('outputs')
def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path): return list(csv.DictReader(path.open(encoding='utf-8')))
experiments=[]
acr=root/'acrima_resnet18_frozen'
experiments.append({'id':'acrima_resnet18_frozen_internal','task':'classification','dataset':'ACRIMA','split':'test','metrics':read_json(acr/'metrics.json'),'predictions':read_csv(acr/'test_predictions.csv'),'gradcam_panels':[str(p).replace('\\','/') for p in sorted((acr/'gradcam').glob('*.png'))],'source':'outputs/acrima_resnet18_frozen'})
experiments.append({'id':'acrima_resnet18_frozen_rim_external','task':'classification','dataset':'RIM-ONE r3','split':'external','metrics':read_json(acr/'rim_external_metrics.json'),'predictions':read_csv(acr/'rim_external_predictions.csv'),'source':'outputs/acrima_resnet18_frozen'})
experiments.append({'id':'acrima_resnet18_frozen_origa_external','task':'classification','dataset':'ORIGA','split':'external_all_650','metrics':read_json(acr/'origa_external_metrics.json'),'predictions':read_csv(acr/'origa_external_predictions.csv'),'source':'outputs/acrima_resnet18_frozen'})
experiments.append({'id':'acrima_resnet18_frozen_refuge_validation_external','task':'classification','dataset':'REFUGE','split':'official_validation','metrics':read_json(acr/'refuge_val_external_metrics.json'),'predictions':read_csv(acr/'refuge_val_external_predictions.csv'),'source':'outputs/acrima_resnet18_frozen'})
rim=root/'rim_unet_baseline'
experiments.append({'id':'rim_unet_baseline','task':'segmentation','dataset':'RIM-ONE r3','split':'test','metrics':read_json(rim/'metrics.json'),'predictions':read_csv(rim/'test_predictions.csv'),'example_panels':[str(p).replace('\\','/') for p in sorted((rim/'examples').glob('*.png'))],'source':'outputs/rim_unet_baseline'})
pooled=root/'pooled_four_dataset_frozen'
pooled_metrics=read_json(pooled/'metrics.json')
for dataset,stem,split in (('ACRIMA','acrima','test'),('ORIGA','origa','provided_B_test'),('REFUGE','refuge','official_validation'),('RIM-ONE r3','rim_one_r3','fixed_test_non_suspect')):
    experiments.append({'id':'pooled_four_frozen_'+stem,'task':'classification','dataset':dataset,'split':split,'model_protocol':pooled_metrics['protocol'],'metrics':pooled_metrics['test_metrics'][dataset],'predictions':read_csv(pooled/(stem+'_test_predictions.csv')),'gradcam_panels':[str(p).replace('\\','/') for p in sorted((pooled/'gradcam'/stem).glob('*.png'))],'source':'outputs/pooled_four_dataset_frozen'})
experiments.append({'id':'pooled_four_frozen_refuge_official_test','task':'classification','dataset':'REFUGE','split':'official_test','metrics':read_json(pooled/'refuge_official_test_metrics.json'),'predictions':read_csv(pooled/'refuge_official_test_predictions.csv'),'source':'outputs/pooled_four_dataset_frozen'})
crop=root/'rim_whole_vs_crop'
if (crop/'metrics.json').exists():
    crop_metrics=read_json(crop/'metrics.json')
    for variant in ('whole','crop'):
        experiments.append({'id':'rim_matched_'+variant+'_frozen','task':'classification','dataset':'RIM-ONE r3','split':'fixed_test_non_suspect','metrics':{k:v for k,v in crop_metrics[variant].items() if k not in ('val_trials',)},'predictions':read_csv(crop/(variant+'_test_predictions.csv')),'protocol':crop_metrics['protocol'][variant],'source':'outputs/rim_whole_vs_crop'})
pooled_seg=root/'pooled_segmentation'
if (pooled_seg/'metrics.json').exists():
    pooled_seg_metrics=read_json(pooled_seg/'metrics.json')
    pooled_seg_predictions=read_csv(pooled_seg/'test_predictions.csv') if (pooled_seg/'test_predictions.csv').exists() else []
    for record in pooled_seg_metrics['test_metrics']:
        dataset=record['dataset']
        dataset_predictions=[row for row in pooled_seg_predictions if row['dataset']==dataset]
        experiments.append({'id':'pooled_segmentation_'+dataset.lower().replace(' ','_').replace('-','_'),'task':'segmentation','dataset':dataset,'split':'fixed_test','metrics':record,'predictions':dataset_predictions,'example_panels':[str(p).replace('\\','/') for p in sorted((pooled_seg/'examples').glob(f"*{dataset.lower().replace(' ','_').replace('-','_')}*.png"))],'source':'outputs/pooled_segmentation'})
drishti_seg=root/'drishti_gs1_mirror'
drishti_clf=root/'drishti_gs1_mirror_classifier'
if (drishti_seg/'metrics.json').exists():
    drishti_metrics=read_json(drishti_seg/'metrics.json')
    drishti_predictions=read_csv(drishti_seg/'test_predictions.csv') if (drishti_seg/'test_predictions.csv').exists() else []
    experiments.append({'id':'drishti_gs1_mirror_segmentation','task':'segmentation','dataset':'DRISHTI-GS1','split':'mirror_official_style_test','source':'lokeshsaipureddi Kaggle mirror','metrics':drishti_metrics,'predictions':drishti_predictions,'example_panels':[str(p).replace('\\','/') for p in sorted((drishti_seg/'examples').glob('*.png'))],'source':'outputs/drishti_gs1_mirror'})
if (drishti_clf/'metrics.json').exists():
    drishti_clf_metrics=read_json(drishti_clf/'metrics.json')
    experiments.append({'id':'drishti_gs1_mirror_frozen_classifier','task':'classification','dataset':'DRISHTI-GS1','split':'mirror_train_val_test_40_10_51','source':'lokeshsaipureddi Kaggle mirror','metrics':drishti_clf_metrics,'predictions':read_csv(drishti_clf/'test_predictions.csv'),'source':'outputs/drishti_gs1_mirror_classifier'})
finetuned=root/'pooled_finetuned_classifier'
if (finetuned/'metrics.json').exists():
    finetuned_metrics=read_json(finetuned/'metrics.json')
    finetuned_predictions=read_csv(finetuned/'test_predictions.csv') if (finetuned/'test_predictions.csv').exists() else []
    for dataset, metrics in finetuned_metrics['test_metrics'].items():
        if dataset == 'REFUGE':
            continue
        experiments.append({'id':'pooled_finetuned_'+dataset.lower().replace(' ','_').replace('-','_'),'task':'classification','dataset':dataset,'split':'fixed_test','source':'outputs/pooled_finetuned_classifier','metrics':metrics,'predictions':[row for row in finetuned_predictions if row['dataset']==dataset],'protocol':finetuned_metrics['protocol']})
    for source_split, split_name in (('val', 'official_validation'), ('test', 'official_test')):
        experiments.append({'id':'pooled_finetuned_refuge_'+source_split,'task':'classification','dataset':'REFUGE','split':split_name,'source':'outputs/pooled_finetuned_classifier','metrics':finetuned_metrics['refuge_test_metrics_by_source_split'][source_split],'predictions':[row for row in finetuned_predictions if row['dataset']=='REFUGE' and row['source_split']==source_split],'protocol':finetuned_metrics['protocol']})
payload={'schema_version':1,'status':'partial_experiments','note':'Only completed measured experiments are listed. DRISHTI-GS1 results use a third-party Kaggle mirror and are not official-source verified.','assets':{'baseline_architecture':'outputs/architecture_baselines.png','classification_comparison':'outputs/acrima_resnet18_frozen/comparison.png','same_split_classifier_comparison':'outputs/classifier_common_split_comparison.png','rim_whole_vs_crop_comparison':'outputs/rim_whole_vs_crop/comparison.png'},'same_split_comparison':read_json(root/'classifier_common_split_comparison.json'),'rim_whole_vs_crop_comparison':read_json(crop/'metrics.json'),'pooled_segmentation':read_json(pooled_seg/'metrics.json') if (pooled_seg/'metrics.json').exists() else None,'drishti_gs1_mirror':{'segmentation':read_json(drishti_seg/'metrics.json') if (drishti_seg/'metrics.json').exists() else None,'classification':read_json(drishti_clf/'metrics.json') if (drishti_clf/'metrics.json').exists() else None},'experiments':experiments}
dest=root/'website_results_partial.json'; dest.write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(dest)
