from utils import dpo_dependencies
from utils import base_dependencies
import random
import pandas as pd
import numpy as np
import shutil
import pandas as pd  
from itertools import combinations
import math

annos_raw = dpo_dependencies.collect_annotations("anno")

def intra_ta_aa(annos_raw):
    tmp2 = annos_raw.groupby(by=['userID', 'day']).head(60)
    tmp2 = tmp2[tmp2['day'] >= pd.to_datetime('2025-03-20').date()]
    tmp2 = tmp2[~((tmp2['day'] > pd.to_datetime('2025-03-20').date()) & (tmp2['userID'].isin(['ta14', 'ta03'])))]
    tmp2 = tmp2[~tmp2['simp_pair_id'].str.contains('ovs')]
    tmp2 = tmp2[['userID', 'simp_pair_id']]

    tmp = annos_raw[~annos_raw['preference'].isna()]
    #tmp = tmp[tmp['info_equality'] == 'y']
    tmp = tmp[tmp['userID'].str.contains('ta')]
    tmp = tmp[tmp['userID'] != 'ta09']
    tmp = tmp[tmp['userID'] != 'ta06']
    tmp = tmp[tmp['userID'] != 'ta08']
    tmp = pd.merge(tmp, tmp2, how='right', on=['userID', 'simp_pair_id'])
    tmp = tmp[tmp['day'] >= pd.to_datetime('2025-02-12').date()]

    max_timestamps = tmp.groupby(['userID', 'day'])['timestamp'].transform('max')
    tmp = tmp[tmp['timestamp'] == max_timestamps]

    tmp = tmp.sort_values(by=['userID', 'day', 'simp_pair_id']).groupby(by=['userID', 'day', 'simp_pair_id']).tail(1)

    tmp = tmp[tmp.duplicated(subset=['userID', 'simp_pair_id'], keep=False)]

    return tmp.sort_values(by=['userID', 'simp_pair_id'])

def intra_ea_aa(annos_raw):
    tmp = annos_raw[~annos_raw['preference'].isna()]
    #tmp = tmp[tmp['info_equality'] == 'y']
    tmp = tmp[tmp['userID'].str.contains('ea')]
    tmp = tmp[tmp['day'] >= pd.to_datetime('2025-02-12').date()]

    tmp = tmp[tmp.duplicated(subset=['userID', 'simp_pair_id'], keep=False)]

    return tmp

def intra_aa_daydiff(intra_data, userID):
    tmp = intra_data[intra_data['userID'] == userID]
    first_occurence = tmp.sort_values(by=['userID', 'day', 'simp_pair_id']).groupby(by=['userID', 'simp_pair_id']).head(1)
    last_occurence = tmp.sort_values(by=['userID', 'day', 'simp_pair_id']).groupby(by=['userID', 'simp_pair_id']).tail(1)
    first_occurence['userID'] = userID + '0'
    last_occurence['userID'] = userID + '1'
    tmp = pd.concat([first_occurence, last_occurence])
    print(tmp['userID'].value_counts())
    if 'ta' in userID:
        tmp = tmp.drop_duplicates(['day', 'simp_pair_id'], keep=False)
    return dpo_dependencies.inter_annotator_agreement(tmp, detail=False, ret=True)

intra_ta_data = intra_ta_aa(annos_raw[annos_raw['day'] != pd.to_datetime('2025-03-13').date()])
iaas_ta = {}
for userID in intra_ta_data['userID'].unique():
    iaas_ta[userID] = intra_aa_daydiff(intra_ta_data, userID)

intra_ea_data = intra_ea_aa(annos_raw) #[annos_raw['day'] < pd.to_datetime('2025-03-13').date()])
print(intra_ea_data['userID'].value_counts())
iaas_ea = {}
for userID in intra_ea_data['userID'].unique():
    iaas_ea[userID] = intra_aa_daydiff(intra_ea_data, userID)

iaas = {**iaas_ta, **iaas_ea}
top_four_itra_ta = sorted([(k, v) for k, v in iaas.items() if 'ta' in k and not math.isnan(v[0])], key=lambda x: x[1][0], reverse=True)[0:4]
top_four_itra_ta = [tmp[0] for tmp in top_four_itra_ta]
top_two_itra_ea = sorted([(k, v) for k, v in iaas.items() if 'ea' in k and not math.isnan(v[0])], key=lambda x: x[1][0], reverse=True)[0:2]
top_two_itra_ea = [tmp[0] for tmp in top_two_itra_ea]
top_four_itra_ta, top_two_itra_ea

import numpy as np
import krippendorff

from statsmodels.stats import inter_rater as irr

import pandas as pd
import numpy as np
import itertools

created_pairs = pd.read_json('data/ATS pairs/all_created_pairs.jsonl', lines=True)
created_pairs = created_pairs[['model', 'original', 'prompt', 'simp_pair_id']]
clean_preference_data = pd.merge(annos_raw, created_pairs, how='left', on=['simp_pair_id','original'], indicator=True)

def inter_aa(all_winning_pairs, annos_raw, ta_or_ea, inter_size, info_equality = True, intra_target = False, filter_date = '2025-02-12', model = ''):

    created_pairs = pd.read_json('data/ATS pairs/all_created_pairs.jsonl', lines=True)
    created_pairs = created_pairs[['model', 'original', 'simp_pair_id']]
    annos_raw = pd.merge(annos_raw, created_pairs, how='left', on=['simp_pair_id','original'], indicator=True)

    if model != '':
        annos_raw = annos_raw[annos_raw['model'] == model]

    annos_raw = annos_raw[annos_raw['userID'] != 'ta09']
    annos_raw = annos_raw[annos_raw['userID'] != 'ta06']
    annos_raw = annos_raw[annos_raw['userID'] != 'ta08']
    annos_raw = annos_raw[annos_raw['day'] >= pd.to_datetime(filter_date).date()] # '2025-03-20'
    annos_raw = annos_raw[~annos_raw['day'].isna()]
    annos_raw = annos_raw[~annos_raw['preference'].isna()]

    # filter to cohort under consideration
    if 't' in ta_or_ea:
        annos_raw = annos_raw[annos_raw['userID'].str.contains('ta')]
    else:
        annos_raw = annos_raw[annos_raw['userID'].str.contains('ea')]

    # filter based on info equality if necessary
    if info_equality == True:
        all_winning_pairs = all_winning_pairs[all_winning_pairs['info_equality'] == 'y']
    all_winning_pairs = all_winning_pairs[all_winning_pairs['simp_pair_id'].str.contains('r2_')]
    legal_pairs = list(all_winning_pairs['simp_pair_id'].unique())

    # remove ones with negative intra-AA
    users = annos_raw['userID'].unique()

    if intra_target == True:
        for user in users:
            if (user in iaas.keys()) and (iaas[user][0] >= 0.05):
                pass
            else:
                annos_raw = annos_raw[annos_raw['userID'] != user]

    # acquire pairs with crossover
    tmp2 = annos_raw[annos_raw['simp_pair_id'].isin(legal_pairs)]
    tmp2 = tmp2.drop_duplicates(subset=['userID', 'simp_pair_id'], keep='last')
    pairs_w_crossover = list(tmp2[tmp2.duplicated(subset=['simp_pair_id'], keep=False)]['simp_pair_id'].unique())
    legal_pairs_w_crossover = [pwc for pwc in pairs_w_crossover if pwc in legal_pairs]
    #print(len(legal_pairs_w_crossover), pairs_w_crossover)

    # acquire the inter data
    tmp = annos_raw[annos_raw['simp_pair_id'].isin(legal_pairs_w_crossover)]
    tmp = tmp[~tmp['preference'].isna()]
    tmp = tmp.sort_values(by=['userID', 'simp_pair_id', 'timestamp'])
    tmp = tmp.drop_duplicates(subset=['userID', 'simp_pair_id'], keep='last')
    #tmp = tmp[['userID', 'simp_pair_id', 'preference', 'original', 'simplification1', 'simplification2']]
    tmp['pref'] = tmp['preference'].str[-1:].astype(int)
    tmp = tmp.sort_values(by=['simp_pair_id', 'userID'])[['userID', 'simp_pair_id', 'pref']]
    spid_vcs = tmp['simp_pair_id'].value_counts()
    spids = spid_vcs[spid_vcs > 3]
    tmp = tmp[tmp['simp_pair_id'].isin(spids.index)]
    #print(tmp['userID'].value_counts())
    user_ids = tmp['userID'].unique()
    #print(user_ids)
    if '2' not in ta_or_ea:
        if len(user_ids) != inter_size:
            inter_size = len(user_ids)

    # Iterate over combinations of userIDs of size 3 and larger
    best = -1
    best_user_set = []
    best_shared_pair_count = 0
    for size in [inter_size]: #range(2, len(user_ids) + 1):
        for user_combination in itertools.combinations(user_ids, size):
            #if ta_or_ea == 'ea':
                #print(user_combination)
            tmp_combination = tmp[tmp['userID'].isin(user_combination)]

            users = sorted(tmp_combination['userID'].unique())
            pairs = sorted(tmp_combination['simp_pair_id'].unique())

            user_index = {user: i for i, user in enumerate(users)}
            pair_index = {pair: i for i, pair in enumerate(pairs)}

            matrix = np.full((len(users), len(pairs)), np.nan)

            for _, row in tmp_combination.iterrows():
                u_idx = user_index[row['userID']]
                p_idx = pair_index[row['simp_pair_id']]
                matrix[u_idx, p_idx] = row['pref']

            data_for_krippendorf = matrix.tolist()

            krippendorff_alpha_nominal = krippendorff.alpha(reliability_data=data_for_krippendorf, level_of_measurement="nominal")

            if krippendorff_alpha_nominal > best:
                best = krippendorff_alpha_nominal
                best_user_set = user_combination
                best_shared_pair_count = len(list(tmp_combination['simp_pair_id'].unique()))
            #if (krippendorff_alpha_nominal > 0.5) or (len(user_combination) == inter_size):
                #print(str(krippendorff_alpha_nominal)[0:5], str(len(user_combination)) + ' users', str(len(list(tmp_combination['simp_pair_id'].unique()))) + ' shaired pairs')

    print(filter_date, info_equality, intra_target, ":", best, best_user_set, best_shared_pair_count)
    return best_user_set

all_winning_pairs = pd.read_json('data/ATS pairs/all_created_pairs.jsonl', lines=True)
top_four_iter_ta = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 11, info_equality = False, intra_target = False, filter_date = '2025-02-12')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 11, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'Disco Llama8B')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 11, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'Llama8B')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 11, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'LeoLM Mistral 7B')
#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 12, info_equality = False, intra_target = True, filter_date = '2025-02-12')
#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 12, info_equality = True, intra_target = True, filter_date = '2025-02-12')
#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 8, info_equality = False, intra_target = False, filter_date = '2025-03-20')
#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 8, info_equality = False, intra_target = True, filter_date = '2025-03-20')
#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ta', inter_size = 8, info_equality = True, intra_target = True, filter_date = '2025-03-20')
top_four_iter_ta = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='t2', inter_size = 4, info_equality=False, intra_target=False, filter_date = '2025-02-12')

top_two_iter_ea = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ea', inter_size = 4, info_equality = False, intra_target = False, filter_date = '2025-02-12')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ea', inter_size = 4, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'Disco Llama8B')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ea', inter_size = 4, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'Llama8B')
_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ea', inter_size = 4, info_equality = False, intra_target = False, filter_date = '2025-02-12', model = 'LeoLM Mistral 7B')

#_ = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='ea', inter_size = 4, info_equality = True, intra_target = False, filter_date = '2025-02-12')

top_two_iter_ea = inter_aa(all_winning_pairs, annos_raw, ta_or_ea='e2', inter_size = 2, info_equality = False, intra_target = False, filter_date = '2025-02-12')

#top_four_iter_ta, top_two_iter_ea
# krippendorfs


def resolve_preference(group):
    mode_values = group['preference'].mode()
    if len(mode_values) == 1:
        chosen_pref = mode_values.iloc[0]
    else:
        chosen_pref = group.loc[group['iaas_value'].idxmax(), 'preference']
    group = group.iloc[:1]
    group['preference'] = chosen_pref
    return group


def process_intersections(df, iaas, userGroup):
    inters = df[df.duplicated(subset=['simp_pair_id'], keep=False)]
    #sids = inters['simp_pair_id'].value_counts()
    #sids = sids[sids > 2]
    #sids = set(sids.index)
    #inters = inters[inters['simp_pair_id'].isin(sids)]

    inters = inters.copy()
    inters.loc[:, "iaas_value"] = inters["userID"].map(lambda user: iaas.get(user, (0, None))[0])

    inters = inters.sort_values(by=["simp_pair_id", "iaas_value"], ascending=[True, False])
    inters = inters.groupby("simp_pair_id", as_index=False).apply(resolve_preference).reset_index(drop=True)
    inters['userID'] = 'inter_'+userGroup
    return pd.concat([df[~df['simp_pair_id'].duplicated(keep=False)], inters])

def intersections_onward(annos_clean):
    ## settle on inter
    ### first, drop inter annotations from people with negative intra-AA if someone with non-negative intra-AA has annotated the same pair
    ta_rows = annos_clean[annos_clean['userID'].str.contains('ta', na=False)]
    lowest_iaa_users = sorted(iaas, key=lambda uid: iaas.get(uid, (0, 0))[0])[:3]
    valid_simp_pairs = set(ta_rows[~ta_rows['userID'].isin(lowest_iaa_users)]['simp_pair_id'])
    ta_rows = ta_rows[~((ta_rows['userID'].isin(lowest_iaa_users)) & (ta_rows['simp_pair_id'].isin(valid_simp_pairs)))]
    ea_rows = annos_clean[annos_clean['userID'].str.contains('ea', na=False)]
    ### second, take modal value for inter pairs
    ta_processed = process_intersections(ta_rows, iaas, 'ta')
    ea_processed = process_intersections(ea_rows, iaas, 'ea')
    ta_processed['userGroup'] = 'ta'
    ea_processed['userGroup'] = 'ea'
    annos_clean = pd.concat([ta_processed, ea_processed])

    print(len(annos_clean), len(ta_processed), len(ea_processed))

    # add in prompt info, format as trl preference dataset
    created_pairs = pd.read_json('data/ATS pairs/all_created_pairs.jsonl', lines=True)
    created_pairs = created_pairs[['model', 'original', 'prompt', 'simp_pair_id']]
    clean_preference_data = pd.merge(annos_clean, created_pairs, how='left', on=['simp_pair_id','original'], indicator=True)
    clean_preference_data = clean_preference_data[['prompt', 'original', 'simplification1', 'simplification2', 'preference', 'model', 'simp_pair_id', 'userID', 'userGroup', 'info_equality', 'issue']]
    pref1 = clean_preference_data[clean_preference_data['preference'] == 'Vereinfachung 1']
    pref2 = clean_preference_data[clean_preference_data['preference'] == 'Vereinfachung 2']
    pref1 = pref1.rename(columns={'simplification1':'chosen', 'simplification2':'rejected'})
    pref2 = pref2.rename(columns={'simplification1':'rejected', 'simplification2':'chosen'})
    clean_preference_data = pd.concat([pref1, pref2])[['simp_pair_id', 'model', 'prompt', 'original', 'chosen', 'rejected', 'userID', 'userGroup', 'info_equality', 'issue']]
    clean_preference_data = clean_preference_data.sample(frac=1, random_state = 5)

    # remove identified issue pairs
    clean_preference_data = clean_preference_data[clean_preference_data['issue'] != 'y']

    return clean_preference_data

# collect everything
annos_raw = dpo_dependencies.collect_annotations("anno")
## keep preferences only
annos_clean = annos_raw[~annos_raw['preference'].isna()]
## keep preferences from february sessions onward
annos_clean = annos_clean[annos_clean['day'] > pd.to_datetime('2025-01-29').date()]
## remove problem users
annos_clean = annos_clean[annos_clean['userID'] != 'ta09']
annos_clean = annos_clean[annos_clean['userID'] != 'ta06']
annos_clean = annos_clean[annos_clean['userID'] != 'ta08']
## remove ovs
annos_clean = annos_clean[~annos_clean['simp_pair_id'].str.contains('ovs')]
## remove intra (latest only)
annos_clean['index'] = annos_clean.index
max_timestamps = annos_clean.groupby(['userID', 'day'])['timestamp'].transform('max')
annos_pclean = annos_clean[annos_clean['timestamp'] == max_timestamps]
annos_clean = annos_clean.sort_values(by=['userID', 'simp_pair_id', 'timestamp', 'index']).groupby(by=['userID', 'simp_pair_id']).tail(1)
annos_clean = annos_clean.drop(columns=['index']).reset_index(drop=True)
## keep pairs created by team only
annos_clean['round'] = annos_clean['simp_pair_id'].str.split('_').str.get(0)
annos_clean = annos_clean[annos_clean['round'] == 'r2']
## de-dup intersections, format as trl preference dataset, remove pairs with issues
print(annos_clean['userID'].value_counts())
clean_preference_data = intersections_onward(annos_clean)
print(len(clean_preference_data), len(clean_preference_data[clean_preference_data['userGroup'] == 'ta']), len(clean_preference_data[clean_preference_data['userGroup'] == 'ea']))
print(clean_preference_data['userID'].value_counts())
#### Save most preference sets

## all data
clean_preference_data.to_json('data/preferences_all.jsonl', orient='records', lines=True)
print('all data: ', len(clean_preference_data)/2)

## info equality
a = clean_preference_data[clean_preference_data['info_equality'] == 'y']
a.to_json('data/preferences_ie.jsonl', orient='records', lines=True)
print('info_equality: ', len(a)/2)

## human + synthetic

## model-specific
b = clean_preference_data[clean_preference_data['model'] == 'Llama8B']
c = clean_preference_data[clean_preference_data['model'] == 'Disco Llama8B']
d = clean_preference_data[clean_preference_data['model'] == 'LeoLM Mistral 7B']
b.to_json('data/preferences_sft_llama8b_checkpoint2400.jsonl', orient='records', lines=True)
c.to_json('data/preferences_sft_disco_llama8b_checkpoint2800.jsonl', orient='records', lines=True)
d.to_json('data/preferences_sft_leolm_mistral7b_checkpoint1600.jsonl', orient='records', lines=True)
print('llama: ', len(b)/2, 'disco llama: ', len(c)/2, 'mistral: ', len(d)/2)

#### Save remaining preference sets (requires us to recreate clean_preference_data based on user set being considered)

## max intraAA
e = intersections_onward(annos_clean[annos_clean['userID'].isin(top_four_itra_ta) | annos_clean['userID'].isin(top_two_itra_ea)])
e.to_json('data/preferences_itra.jsonl', orient='records', lines=True)
print('intra ta: ', len(e[e['userGroup'] == 'ta']), 'intra ea: ', len(e[e['userGroup'] == 'ea']))

## max interAA
f = intersections_onward(annos_clean[annos_clean['userID'].isin(top_four_iter_ta) | annos_clean['userID'].isin(top_two_iter_ea)])
f.to_json('data/preferences_iter.jsonl', orient='records', lines=True)
print('inter ta: ', len(f[f['userGroup'] == 'ta']), 'inter ea: ', len(f[f['userGroup'] == 'ea']))


# preferences_raw.jsonl
## version of preferences with all annotation-stage data, including problematic pairs, evaluation pairs, and ovs pairs

# collect everything
annos_raw = dpo_dependencies.collect_annotations("anno")
## keep preferences only
annos_clean = annos_raw[~annos_raw['preference'].isna()]

# deduplicate intra annotations (latest only)
annos_clean['index'] = annos_clean.index
max_timestamps = annos_clean.groupby(['userID', 'day'])['timestamp'].transform('max')
annos_pclean = annos_clean[annos_clean['timestamp'] == max_timestamps]
annos_clean = annos_clean.sort_values(by=['userID', 'simp_pair_id', 'timestamp', 'index']).groupby(by=['userID', 'simp_pair_id']).tail(1)
annos_clean = annos_clean.drop(columns=['index']).reset_index(drop=True)

# list reasons for exclusion from final preference set
annos_clean['exclusion_reason'] = ''
annos_clean.loc[annos_clean['userID'].isin(['ta06', 'ta08', 'ta09']), 'exclusion_reason'] = 'annotator later excluded'
annos_clean.loc[annos_clean['simp_pair_id'].str.contains('ovs', na=False), 'exclusion_reason'] = 'orig-v-simp sense check'
annos_clean.loc[annos_clean['day'] < pd.to_datetime('2025-01-29'), 'exclusion_reason'] = 'pilot session data'
annos_clean['round'] = annos_clean['simp_pair_id'].str.split('_').str.get(0)
annos_clean.loc[annos_clean['round'] == 'r1', 'exclusion_reason'] = 'pair created before standardized creation process'
annos_clean.loc[annos_clean['round'] == 'r3', 'exclusion_reason'] = 'post-DPO eval pair'

# add in prompt info, format as trl preference dataset
created_pairs = pd.read_json('data/ATS pairs/all_created_pairs.jsonl', lines=True)
created_pairs = created_pairs[['model', 'original', 'prompt', 'simp_pair_id']]
eval_pairs_w_metadata = pd.read_json('data/ATS pairs/eval_pairs_w_metadata.jsonl', lines=True)
eval_pairs_w_metadata = eval_pairs_w_metadata[['prompt', 'original', 'group_1', 'group_2', 'simp_pair_id', 'backbone']]
eval_pairs_w_metadata = eval_pairs_w_metadata.rename(columns={'backbone': 'model'})
created_pairs = pd.concat([created_pairs, eval_pairs_w_metadata], ignore_index=True)
created_pairs['dpo_inference'] = 'neither'
created_pairs.loc[created_pairs['group_1'].str.contains('dpo', na=False), 'dpo_inference'] = 'simplification1'
created_pairs.loc[created_pairs['group_2'].str.contains('dpo', na=False), 'dpo_inference'] = 'simplification2'
clean_preference_data = pd.merge(annos_clean, created_pairs, how='left', on=['simp_pair_id','original'], indicator=True)
clean_preference_data['userGroup'] = 'ta'
clean_preference_data.loc[clean_preference_data['userID'].str.contains('ea'), 'userGroup'] = 'ea'
clean_preference_data = clean_preference_data[['prompt', 'original', 'simplification1', 'simplification2', 'preference', 'model', 'simp_pair_id', 'userID', 'userGroup', 'info_equality', 'issue', 'dpo_inference', 'exclusion_reason']]
pref1 = clean_preference_data[clean_preference_data['preference'] == 'Vereinfachung 1']
pref2 = clean_preference_data[clean_preference_data['preference'] == 'Vereinfachung 2']
pref1 = pref1.rename(columns={'simplification1':'chosen', 'simplification2':'rejected'})
pref1.loc[pref1['dpo_inference'] == 'simplification1', 'dpo_inference'] = 'chosen'
pref1.loc[pref1['dpo_inference'] == 'simplification2', 'dpo_inference'] = 'rejected'
pref2 = pref2.rename(columns={'simplification1':'rejected', 'simplification2':'chosen'})
pref2.loc[pref2['dpo_inference'] == 'simplification2', 'dpo_inference'] = 'chosen'
pref2.loc[pref2['dpo_inference'] == 'simplification1', 'dpo_inference'] = 'rejected'

clean_preference_data = pd.concat([pref1, pref2])[['simp_pair_id', 'model', 'prompt', 'original', 'chosen', 'rejected', 'userID', 'userGroup', 'info_equality', 'issue', 'exclusion_reason', 'dpo_inference']]
clean_preference_data = clean_preference_data.sample(frac=1, random_state = 5)

clean_preference_data.loc[clean_preference_data['issue'] == 'y', 'exclusion_reason'] = 'language issue'
clean_preference_data.rename(columns={'issue': 'language_issue'}, inplace=True)

clean_preference_data = clean_preference_data.replace('', np.nan)
clean_preference_data = clean_preference_data[~clean_preference_data['chosen'].isna()]
clean_preference_data.to_json('data/preferences_raw.jsonl', orient='records', lines=True)