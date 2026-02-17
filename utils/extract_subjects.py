import pandas as pd

file_path = '/Users/tongshan/Documents/TemporalIntegration/data/Temporal_integrartion_subjectlist_2026_01_16_updated_filtering.xlsx'
df = pd.read_excel(file_path)

conditions = ['TI1_orig', 'TI1_word', 'TI1_sent']
lists = {}

for cond in conditions:
    # Filter where column is 1
    subjects = df[df[cond] == 1]['PID'].astype(str).tolist()
    lists[cond] = subjects

# Print in a format easy to copy-paste or read
print("SUBJECT_LISTS = {")
for cond, subs in lists.items():
    print(f"    '{cond}': [")
    # Format list with quotes
    formatted_subs = [f"'{s}'" for s in subs]
    # Break into lines of 5
    for i in range(0, len(formatted_subs), 5):
        chunk = formatted_subs[i:i+5]
        print(f"        {', '.join(chunk)},")
    print("    ],")
print("}")
