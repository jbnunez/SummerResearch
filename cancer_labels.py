#cancer_labels.py
import pandas as pd
import numpy as np
import pickle

IDs = pickle.load(open("cancer_IDs", "rb"))
labels_path = "../Desktop/cancer/label_data/"
label_options = {}


def labels_to_one_hot(series):
    inds = series.index
    seen = {}
    count = 0
    for i in inds:
        current = series[i]
        #get rid of duplicate labels
        if type(series[i]) is not str:#type(series):
            first = np.array(series[i])[0]
            series = series.drop([i])
            series[i] = first
            #print(type(series[i]))
            #print("after\n",series[i])

        if series[i] not in seen.keys():
            seen[series[i]] = count
            count += 1

    eye = np.eye(count)
    inds = series.index
    for i in inds:
        series[i] = eye[seen[series[i]]]
    return series




#nationwidechildrens.org_clinical_patient_gbm contains:
#new tumor event, days to death, vital status, tumor status
patient_file = "nationwidechildrens.org_clinical_patient_gbm.txt"
patient_df = pd.read_csv(labels_path+patient_file, sep='\t', index_col=1)
patient_df = patient_df.drop(patient_df.index[[0,1]])

patient_ts = patient_df['tumor_status']#.filter((lambda x: x!='[Not Available]'))
patient_ts = patient_ts[patient_ts!='[Not Available]']
label_options['patient_ts'] = labels_to_one_hot(patient_ts)

patient_vs = patient_df['vital_status']
patient_vs = patient_vs[patient_vs!='[Not Available]']
label_options['patient_vs'] = labels_to_one_hot(patient_vs)

patient_dd = patient_df['death_days_to']
patient_dd = patient_dd[patient_dd!='[Not Applicable]']
label_options['patient_dd'] = labels_to_one_hot(patient_dd)

patient_tss = patient_df['tissue_source_site']
patient_tss = patient_tss[patient_tss!='[Not Available]']
label_options['patient_tss'] = labels_to_one_hot(patient_tss)




#nationwidechildrens.org_clinical_follow_up_v1.0_nte_gbm contains:
#new neoplasm event type (upon follow up)
nte_file =  "nationwidechildrens.org_clinical_follow_up_v1.0_nte_gbm.txt"
nte_df = pd.read_csv(labels_path+nte_file, sep='\t', index_col=1)
nte_df = nte_df.drop(nte_df.index[[0,1]])

nte_nnet = nte_df['new_neoplasm_event_type']
nte_nnet = nte_nnet[nte_nnet!='[Unknown]']
#print(nte_nnet)
label_options['nte_nnet'] = labels_to_one_hot(nte_nnet)




#nationwidechildrens.org_clinical_follow_up_v1.0_gbm contains:
#vital status, days to death, tumor status
followup_file = "nationwidechildrens.org_clinical_follow_up_v1.0_gbm.txt"
followup_df = pd.read_csv(labels_path+followup_file, sep='\t', index_col=1)
followup_df = followup_df.drop(followup_df.index[[0,1]])

followup_ts = followup_df['tumor_status']#.filter((lambda x: x!='[Not Available]'))
followup_ts = followup_ts[followup_ts!='[Not Available]']
label_options['followup_ts'] = labels_to_one_hot(followup_ts)

followup_vs = followup_df['vital_status']
followup_vs = followup_vs[followup_vs!='[Not Available]']
label_options['followup_vs'] = labels_to_one_hot(followup_vs)

followup_dd = followup_df['death_days_to']
followup_dd = followup_dd[followup_dd!='[Not Applicable]']
label_options['followup_dd'] = labels_to_one_hot(followup_dd)


