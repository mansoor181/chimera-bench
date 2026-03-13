import pandas as pd

def analyze_dataset(csv_path, label, is_energy=True):

    print(f"\n=== {label} ===")
    df = pd.read_csv(csv_path)

    mean_stats = df.groupby('cdr')[['rmsd', 'seqid']].mean().reset_index()

    if not is_energy:
        print(mean_stats)
        return mean_stats

    imp_df = df
    grouped = imp_df.groupby(['cdr', 'structure'])

    imp_results = []
    for (cdr, structure), group in grouped:
        total_samples = len(group)
        better_count = (group['dG_gen'] < group['dG_ref']).sum()
        imp_percentage = (better_count / total_samples) * 100
        imp_results.append({'cdr': cdr, 'IMP (%)': imp_percentage})

    imp_df = pd.DataFrame(imp_results).reset_index(drop=True)
    cdr_mean_imp = imp_df.groupby('cdr')['IMP (%)'].mean().reset_index()
    final_df = mean_stats.merge(cdr_mean_imp, on='cdr')
    
    print(final_df)
    
    return final_df



analyze_dataset("./summary.csv", "Method", is_energy=True)