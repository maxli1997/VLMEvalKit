# %%
import pandas as pd
import os
from vlmeval.dataset.utils import mathvista

model_name = 'R1-VL-2B'
choice = 4
dataset = 'MathVista_MINI'
judge = 'gpt-4o-mini'

# %%
file_path = os.path.join('outputs', model_name, f'{model_name}_{dataset}_{judge}.xlsx')
df_greedy = pd.read_excel(file_path)
# CoT results aggregation
df_list = [pd.DataFrame() for _ in range(2*choice)]
for i in range(choice):
    file_path = os.path.join('outputs', model_name, f'{model_name}_{dataset}_False_{i}_{judge}.xlsx')
    df_list[i] = pd.read_excel(file_path)
for i in range(choice):
    file_path = os.path.join('outputs', model_name, f'{model_name}_{dataset}_True_{i}_{judge}.xlsx')
    df_list[i+choice] = pd.read_excel(file_path)
summary = df_list[0].copy(deep=True)
summary['length'] = summary['length'].astype(float)
summary = summary.rename(columns={'prediction': 'ref'})
majority = df_list[choice].copy(deep=True)
majority['length'] = majority['length'].astype(float)
majority = majority.rename(columns={'prediction': 'ref'})
CoT_correct = 0
maj_correct = 0
greedy_correct = 0
for rows in zip(*[df.iterrows() for df in df_list]):
    index = rows[0][0]
    res_list = [rows[x][1].res for x in range(choice)]
    maj_res_list = [rows[x+choice][1].res for x in range(choice)]
    confidence_list = [rows[x][1].confidence for x in range(choice)]
    length_list = [rows[x][1].length for x in range(choice)]
    maj_length_list = [rows[x+choice][1].length for x in range(choice)]
    summary.at[index, 'length'] = sum(length_list)/choice
    majority.at[index, 'length'] = sum(maj_length_list)/choice
    aggr = {}
    for j, key in enumerate(res_list):
        aggr[key] = aggr.get(key, 0) + confidence_list[j]
    best_answer = max(aggr, key=aggr.get)
    summary.at[index, 'res'] = best_answer
    vote = {}
    for j, key in enumerate(maj_res_list):
        vote[key] = vote.get(key, 0) + 1
    most_answer = max(vote, key=vote.get)
    majority.at[index, 'res'] = most_answer
    summary.at[index, 'confidence'] = aggr[best_answer]
    if mathvista.post_check(summary.iloc[index], prefetch=False):
        summary.at[index, 'log'] = 'Correct'
        CoT_correct += 1
    else:
        summary.at[index, 'log'] = 'Wrong'
    if mathvista.post_check(majority.iloc[index], prefetch=False):
        majority.at[index, 'log'] = 'Correct'
        maj_correct += 1
    else:
        majority.at[index, 'log'] = 'Wrong'
    if mathvista.post_check(df_greedy.iloc[index], prefetch=False):
        df_greedy.at[index, 'log'] = 'Correct'
        greedy_correct += 1
    else:
        df_greedy.at[index, 'log'] = 'Wrong'
    select = []
    for k, key in enumerate(res_list):
        if best_answer == key:
            select.append(k+1)
    summary.at[index, 'ref'] = select
    maj_select = []
    for k, key in enumerate(maj_res_list):
        if most_answer == key:
            maj_select.append(k+1)
    majority.at[index, 'ref'] = maj_select
print('CoT Accuracy: ', CoT_correct/len(summary))
print('Majority Vote Accuracy: ', maj_correct/len(majority))
print('Greedy Accuracy: ', greedy_correct/len(df_greedy))

# %%
with pd.ExcelWriter(f'{model_name}_{dataset}_{choice}_{judge}.xlsx', engine='openpyxl') as writer:
    df_greedy.to_excel(writer, sheet_name=f'Greedy', index=False)
    summary.to_excel(writer, sheet_name=f'CoT-Aggregation', index=False)
    majority.to_excel(writer, sheet_name=f'Majority-Vote', index=False)
    for i in range(choice):
        df_list[i].to_excel(writer, sheet_name=f'CoT-{i+1}', index=False)
        df_list[i+choice].to_excel(writer, sheet_name=f'Majority-{i+1}', index=False)


