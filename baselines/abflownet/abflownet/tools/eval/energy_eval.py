import pandas as pd
import argparse

parser = argparse.ArgumentParser()

# parse a csv file path
parser.add_argument("--csv_path", type=str)
args = parser.parse_args()

def rank_top1_and_compute_mean(csv_path):


    df = pd.read_csv(csv_path)
    df = df.loc[df['cdr'] == 'H_CDR3', :]


    df["rank_E"]  = df.groupby(["structure", "cdr"])["dG_ref"].rank(method="first", ascending=True) # E_total
    df["rank_dG"] = df.groupby(["structure", "cdr"])["dG_gen"].rank(method="first", ascending=True)
    df["composite_rank"] = df["rank_E"] + df["rank_dG"]


    top1 = df.loc[
        df.groupby(["structure", "cdr"])["composite_rank"].idxmin(),
        ["structure", "cdr", "filename", "dG_ref", "dG_gen"],
    ]


    mean_E_total = top1["dG_ref"].mean()
    mean_dG_gen = top1["dG_gen"].mean()

    print(f"Mean CDR E_total : {mean_E_total:.3f}")
    print(f"Mean CDR-Ag ΔG : {mean_dG_gen:.3f}\n")


rank_top1_and_compute_mean(args)
