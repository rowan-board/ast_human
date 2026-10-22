# useful functions for human data
import ast
import pandas as pd
import numpy as np
import cmdstanpy as cmd
import scripts.helper_functions as func

# calculate trajectories 

def calc_behavioural_metrics(data):
    traj_df = data.groupby(['Participant Public ID', 'condition', 'stage'])['n_errors_stage'].max()
    traj_df = traj_df.reset_index()

    # calculate shift costs
    costs_df = traj_df[traj_df['stage'].isin(['IDS4','EDS'])]
    costs_df = costs_df.pivot(index=['Participant Public ID', 'condition'],columns='stage',values='n_errors_stage').reset_index()
    costs_df['shift_cost'] = costs_df['EDS'] - costs_df['IDS4']
    costs_df = costs_df.dropna()
    costs_df = costs_df[['Participant Public ID', 'condition', 'shift_cost']]

    # calculate slopes 
    slope_df = traj_df[traj_df['stage'].isin(['IDS1','IDS4'])]
    slope_df = slope_df.pivot(index=['Participant Public ID', 'condition'],columns='stage',values='n_errors_stage').reset_index()
    slope_df['slope'] = slope_df['IDS1'] - slope_df['IDS4']
    slope_df = slope_df.dropna()
    slope_df = slope_df[['Participant Public ID', 'condition', 'slope']]

    tmp_df = costs_df.merge(slope_df,on=['Participant Public ID', 'condition'])

    return traj_df, tmp_df

def get_p_switch_after_correct(df, n_choices=3):
    results = []
    ids_stages = [s for s in df['stage'].unique() if 'IDS' in str(s)]
    
    for pid, pdata in df.groupby('Participant Public ID'):
        for stage, sdata in pdata[pdata['stage'].isin(ids_stages)].groupby('stage'):
            sdata = sdata.reset_index(drop=True)
            
            correct_idx = sdata[sdata['reward'] == 1].index
            if len(correct_idx) == 0:
                continue
            first_correct = correct_idx[0]
            
            # Trials immediately after first correct
            window = sdata.iloc[first_correct + 1 : first_correct + 1 + n_choices]
            if len(window) == 0:
                continue
            
            # No reward = switched away from correct dimension
            p_switch = (window['reward'] == 0).mean()
            results.append({
                'Participant Public ID': pid,
                'stage': stage,
                'p_switch': p_switch,
                'n_trials': len(window)
            })
    
    results_df = pd.DataFrame(results)
    participant_means = results_df.groupby('Participant Public ID')['p_switch'].mean().reset_index()
    return participant_means, results_df

def failures(raw_data):
    ids = set()

    # failed first stage
    tmp = raw_data.groupby('Participant Public ID').max('stage')
    ids |= set(tmp[tmp['stage'] == 0]['Participant Public ID'])

    # bots
    ids |= set(raw_data[raw_data['bot'] == 1]['Participant Public ID'])

    # sound check fails
    ids |= set(raw_data[raw_data['sound_check'] == 0]['Participant Public ID'])

    # remove them
    #raw_data = raw_data[~raw_data['Participant Private ID'].isin(ids)]

    return ids

def fix_stimuli(data):
    ids = data['Participant Public ID'].unique()
    ids = [x for x in ids if pd.notna(x)]  # safer way to filter out NaN

    clean_df = pd.DataFrame()

    for i in range(len(ids)):
        id_ = ids[i]
        tmp_data = data[data['Participant Public ID'] == id_].copy()  # ensure it's a deep copy
        tmp_data.reset_index(drop=True, inplace=True)  # drop old index

        for j in range(len(tmp_data)):
            c_val = tmp_data.loc[j, 'C']
            s_val = tmp_data.loc[j, 'S']

            if isinstance(c_val, str) and len(c_val) > 4:
                try:
                    c_parsed = ast.literal_eval(c_val)
                    a = c_parsed[0][0] + ',' + c_parsed[0][1]
                    b = c_parsed[1][0] + ',' + c_parsed[1][1]
                    tmp_data.at[j, 'C'] = [a, b]  # Use .at instead of .loc
                except Exception as e:
                    print(f"Error parsing C at row {j}: {e}")

            if isinstance(s_val, str):
                s_val_clean = s_val.replace('[', '').replace(']', '').replace('"', '')
                tmp_data.at[j, 'S'] = s_val_clean

        clean_df = pd.concat([clean_df, tmp_data], ignore_index=True)

    return clean_df

def old_fix_stim(data):
    
    # get id list with no nans
    ids = data['Participant Public ID'].unique()
    ids = [x for x in ids if x is not np.nan]

    # create empty df to fill
    clean_df = pd.DataFrame()

    # loop over subjs
    for i in range(len(ids)):
        id = ids[i]

        # isolate the columns we want
        tmp_data = data[data['Participant Public ID'] == id]
        tmp_data = tmp_data.reset_index(drop = True)
        tmp_choice = tmp_data['C'].copy()
        tmp_S = tmp_data['S'].copy()

        # now let's loop through the array and check each row
        for i in range(len(tmp_S)):
            # if we're in SD then stimulus strings will be 15 characters
            # we only want to convert stim in CD onwards as SD is correct already
            if len(tmp_S[i]) > 15:
                
                # convert string to list
                tmp_S[i]=ast.literal_eval(tmp_S[i])

                # get the list elements, separate by comma 
                a = tmp_S[i][0][0] + ',' + tmp_S[i][0][1]
                b = tmp_S[i][1][0] + ',' + tmp_S[i][1][1]

                # concat and turn back into string (as we will convert the whole dataset to lists later on)
                tmp_S[i] = str([a,b])

                # remove some characters in choice column
                tmp_choice[i] = tmp_choice[i].replace('[','')
                tmp_choice[i] = tmp_choice[i].replace(']','')
                tmp_choice[i] = tmp_choice[i].replace('"','')

        # overwrite
        tmp_data['C'] = tmp_choice
        tmp_data['S'] = tmp_S

        # append to clean_df
        if i == 0:
            clean_df = tmp_data
        else:
            clean_df = pd.concat([clean_df,tmp_data])

    # reset index and return
    clean_df = clean_df.reset_index(drop=True)
    return clean_df


# general preprocessing for all questionnaires, input is a list of qaires
def clean_qaires(qaires):
    for i in range(len(qaires)):
        
        # set current questionnaire to q
        q = qaires[i]

        # preprocessing a few bits to specific cols 
        if q.columns.isin(['Question Key']).any():
            q['Key'] = q['Question Key']

        # get cols we want
        q = q[['Participant Public ID', 'Response', 'Key']]

        q = q[q['Key'].str.contains('quantised', case=False, na=False)]

        # reassign questionnaire to array
        qaires[i] = q
    return qaires
    
### series of functions to clean specific quesionnaires
def clean_ctq(ctq):
    # remove any qs just called quantised (attention checks)
    ctq =  ctq[ctq['Key'] != 'quantised']

    # create q number column and transform response to scores
    ctq['q_number'] = ctq.groupby(['Participant Public ID']).cumcount()+1
    ctq['Response'] = pd.to_numeric(ctq['Response'], errors='coerce')
    ctq_reverse = [2, 5, 7, 13, 19, 26, 28]
    ctq.loc[ctq['q_number'].isin(ctq_reverse), 'Response'] = (6 - ctq.loc[ctq['q_number'].isin(ctq_reverse), 'Response'])
    ctq_denial = [10, 16, 22]
    ctq.loc[ctq['q_number'].isin(ctq_denial), 'Response'] = (ctq.loc[ctq['q_number'].isin(ctq_denial), 'Response'] == 5).astype(int)

    # get items then sum for scores and subscale scores
    ctq_items = ctq.copy()
    e_abuse = ctq_items[ctq_items['q_number'].isin([3,8,14,18,25])].groupby('Participant Public ID')['Response'].sum().reset_index().rename(columns={'Response': 'e_abuse'})
    p_abuse = ctq_items[ctq_items['q_number'].isin([9,11,12,15,17])].groupby('Participant Public ID')['Response'].sum().reset_index().rename(columns={'Response': 'p_abuse'})
    s_abuse = ctq_items[ctq_items['q_number'].isin([20,21,23,24,27])].groupby('Participant Public ID')['Response'].sum().reset_index().rename(columns={'Response': 's_abuse'})
    e_neglect = ctq_items[ctq_items['q_number'].isin([5,7,13,19,28])].groupby('Participant Public ID')['Response'].sum().reset_index().rename(columns={'Response': 'e_neglect'})
    p_neglect = ctq_items[ctq_items['q_number'].isin([1,2,4,6,26])].groupby('Participant Public ID')['Response'].sum().reset_index().rename(columns={'Response': 'p_neglect'})

    ctq_scores = ctq.groupby('Participant Public ID')['Response'].sum().reset_index()

    # bind them all
    ctq_scores = ctq_scores.merge(e_abuse,on='Participant Public ID') \
                  .merge(p_abuse, on='Participant Public ID') \
                  .merge(s_abuse, on='Participant Public ID') \
                  .merge(e_neglect, on='Participant Public ID') \
                  .merge(p_neglect, on='Participant Public ID') 

    return ctq_scores, ctq_items

def clean_ius(ius):

    # get q number and remove attention check
    ius['q_number'] = ius.groupby(['Participant Public ID']).cumcount()+1
    ius = ius[ius['q_number'] != 13]

    # make numeric
    ius['Response'] = pd.to_numeric(ius['Response'], errors='coerce')

    # get items and sum scores 
    ius_items = ius.copy()
    ius_scores = ius.groupby('Participant Public ID')['Response'].sum().reset_index()

    return ius_scores, ius_items

def clean_oci_r(oci_r):

    # make q number and subtract 1 from all responses (start from 0)
    oci_r['q_number'] = oci_r.groupby(['Participant Public ID']).cumcount()+1
    oci_r['Response'] = pd.to_numeric(oci_r['Response'], errors='coerce')
    oci_r['Response'] = oci_r['Response'] - 1

    # get items and sum for scores
    oci_r_items = oci_r.copy()
    oci_r_scores = oci_r.groupby('Participant Public ID')['Response'].sum().reset_index()

    return oci_r_scores, oci_r_items

def clean_schiz(schiz):

    # make q number
    schiz['q_number'] = schiz.groupby(['Participant Public ID']).cumcount()+1
    schiz['Response'] = pd.to_numeric(schiz['Response'], errors='coerce')
    schiz['Response'] = schiz['Response'] - 1

    # reverse score
    schiz_reverse = [26, 27, 28, 30 , 31, 34, 37 ,39]
    schiz.loc[~schiz['q_number'].isin(schiz_reverse), 'Response'] = (1 - schiz.loc[~schiz['q_number'].isin(schiz_reverse), 'Response'])

    # get items and sum scores
    schiz_items = schiz.copy()
    schiz_scores = schiz.groupby('Participant Public ID')['Response'].sum().reset_index()

    return schiz_scores, schiz_items

def clean_gad(gad):

    # make q number
    gad['q_number'] = gad.groupby(['Participant Public ID']).cumcount()+1
    gad['Response'] = pd.to_numeric(gad['Response'], errors='coerce')
    gad['Response'] = gad['Response'] - 1

    # get items and scores
    gad_items = gad.copy()
    gad_scores = gad.groupby('Participant Public ID')['Response'].sum().reset_index()

    return gad_scores, gad_items

def clean_phq(phq):

    # make q number
    phq['q_number'] = phq.groupby(['Participant Public ID']).cumcount()+1
    phq['Response'] = pd.to_numeric(phq['Response'], errors='coerce')
    phq['Response'] = phq['Response'] - 1

    # get items and scores
    phq_items = phq.copy()
    phq_scores = phq.groupby('Participant Public ID')['Response'].sum().reset_index()

    return phq_scores, phq_items

def clean_aes(aes):

    # make q number
    aes['q_number'] = aes.groupby(['Participant Public ID']).cumcount()+1
    aes['Response'] = pd.to_numeric(aes['Response'], errors='coerce')

    # reverse score
    aes_reverse = [6, 10, 11]
    aes.loc[~aes['q_number'].isin(aes_reverse), 'Response'] = (5 - aes.loc[~aes['q_number'].isin(aes_reverse), 'Response'])

    # get items and scores
    aes_items = aes.copy()
    aes_scores = aes.groupby('Participant Public ID')['Response'].sum().reset_index()

    return aes_scores, aes_items

# make datalist for modelling
def make_datalist(data, subj):
    tmp_data = data.loc[(data['Participant Public ID'] == subj)].copy()
    tmp_data = tmp_data.reset_index(drop=True)

   
    def parse_and_replace(s):
        # Make sure it's a string
        s = str(s).strip()
        try:
            # Parse string into a real list
            lst = ast.literal_eval(s)
            # Replace -1 → 0 inside the list
            return [-1 if x == -1 else x for x in lst]
        except Exception:
            # If parsing fails, return the original
            return s  

    # convert to non counterfactual feedback
    tmp_data["feedback"] = tmp_data["feedback"].apply(parse_and_replace)
    #tmp_data["feedback"] = tmp_data["feedback"].apply(
    #    lambda x: [-1, -1, -1] if sum(x) == 1 else x
    #)

    tmp_R = tmp_data['feedback']

    tmp_choice = tmp_data['C']
    tmp_S = tmp_data['S'].apply(ast.literal_eval)
    if tmp_data['initial_dimension'][0] == 0:
        tmp_d1 = 'A'
    else:
        tmp_d1 = 'B'

    model_data = {'R': tmp_R.to_list(), 'choice': tmp_choice.to_list(), 'S': tmp_S.to_list(), 'dimension1': tmp_d1, 'id': subj}
    return model_data

# useful functions 
def run_model(args):
    stan_file, data, seed = args
    model = cmd.CmdStanModel(stan_file=stan_file)
    return model.sample(
        data=data,
        chains=4,
        parallel_chains=4,
        seed=seed,
    )

def run_model_multi_thread(job, CHAINS, THREADS_PER_CHAIN):
    stan_file, data, seed = job
    import cmdstanpy as cmd



    model = cmd.CmdStanModel(stan_file=stan_file,
                             cpp_options={"STAN_THREADS": True})

    return model.sample(data=data,
                        chains=CHAINS,
                        parallel_chains=CHAINS,
                        threads_per_chain=THREADS_PER_CHAIN,
                        iter_warmup=1000,
                        iter_sampling=1000,
                        adapt_delta=0.95,
                        seed=seed)


def pad_feature_idx(x):
    # If it's already a list, keep it
    if isinstance(x, list):
        # If length 1, pad with 0
        return x + [0] if len(x) == 1 else x
    else:
        # If it's a single value (int or str), convert to list and pad
        return [x, 0]
    
def pad_to_nested(x, target_outer_len=3, target_inner_len=2):
    # If already nested (list of lists)
    if isinstance(x[0], list):
        # pad outer if needed
        padded = x.copy()
        while len(padded) < target_outer_len:
            padded.append([0]*target_inner_len)
        # pad inner lists
        padded = [sub + [0]*(target_inner_len - len(sub)) for sub in padded]
        return padded
    else:
        # flat list → convert to nested
        nested = []
        for i in range(target_outer_len):
            # take element if exists, else 0
            val = x[i] if i < len(x) else 0
            # make inner list of target_inner_len
            nested.append([val] + [0]*(target_inner_len - 1))
        return nested
    
def subset_stan_data(data, subj_idx):
    data_small = {}

    for key, val in data.items():
        if isinstance(val, list):
            arr = np.array(val)

            # subject-level vectors
            if arr.shape[0] == data["N"]:
                data_small[key] = arr[subj_idx].tolist()

            else:
                data_small[key] = val

        else:
            data_small[key] = val

    data_small["N"] = len(subj_idx)
    return data_small

def make_stan_datalist(data, save_trials=0):
    SUBJ = 'Participant Public ID'
    """Clean the task data and build the data dict for the Stan model."""
    data = data.copy()
    data['S'] = data['S'].apply(ast.literal_eval)
    data['stimuli'] = data['stimuli'].apply(ast.literal_eval)

    # =========================
    # EXCLUSIONS
    # =========================
    # participants with duplicated trials
    repeats = data.loc[data.duplicated([SUBJ, 'trial_number'], keep=False), SUBJ]
    # participants who never reach EDS (stage 6)
    max_stage = data.groupby(SUBJ)['stage'].max()
    failures = max_stage[max_stage < 6].index

    data = data[~data[SUBJ].isin(repeats) & ~data[SUBJ].isin(failures)]
    data = data[data['stages_attempted'] < 12].copy()  # drop stages after IDS8

    # =========================
    # PREPROCESSING
    # =========================
    data['idx'] = [s.index(c) for s, c in zip(data['S'], data['C'])]
    data['feature_idx'] = [
        func.pad_feature_idx(stim[i]) for stim, i in zip(data['stimuli'], data['idx'])
    ]
    data['stimuli'] = data['stimuli'].apply(func.pad_to_nested)
    data['trial'] = data.groupby(SUBJ).cumcount()

    subj_order = data[SUBJ].unique()

    def wide(col):
        """Subjects x trials matrix of `col`, rows in subj_order, NaN where padded."""
        return data.pivot(index=SUBJ, columns='trial', values=col).reindex(subj_order)

    # everything below is shifted +1 for Stan's 1-based indexing; padded trials -> 0
    choice = wide('feature_idx').map(
        lambda x: [int(i) + 1 for i in x] if isinstance(x, list) else [0, 0]
    )
    stimuli = wide('stimuli').map(
        lambda cell: [[int(x) + 1 for x in pair] for pair in cell]
        if isinstance(cell, list) else [[0, 0]] * 3
    )
    choice_stim = (wide('idx') + 1).fillna(0).astype(int)

    # padded trials -> -1
    block = wide('stages_attempted').fillna(-1).astype(int)
    reward = wide('reward').replace(0, -1).fillna(-1)

    T_subj = data.groupby(SUBJ)['trial_number'].max().reindex(subj_order).astype(int)
    initial_dimension = (
        data[[SUBJ, 'initial_dimension']]
        .drop_duplicates()
        .set_index(SUBJ)['initial_dimension']
        .reindex(subj_order)  # raises if a participant has >1 initial_dimension
        .astype(int)
    )

    # =========================
    # MODEL INPUT
    # =========================
    K = int(max(f[0] for f in data['feature_idx'])) + 1

    return {
        'N': len(subj_order),
        'T': int(data['trial_number'].max()),
        'T_subj': T_subj.tolist(),
        'block': block.to_numpy().tolist(),
        'reward': reward.to_numpy().tolist(),
        'choice': choice.to_numpy().tolist(),
        'choice_stim': choice_stim.to_numpy().tolist(),
        'stimuli': stimuli.to_numpy().tolist(),
        'initial_dimension': initial_dimension.tolist(),
        'K': K,
        'A_identity': np.identity(K),
        'B_identity': np.identity(K),
        'save_trials': save_trials,
    }

