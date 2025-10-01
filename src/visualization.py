# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_count_by_group(df, group_col, target_col='class', rotate_xticks=False, figsize=(8,4)):
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=group_col, hue=target_col)
    if rotate_xticks:
        plt.xticks(rotation=45)
    plt.title(f"{target_col} by {group_col}")
    plt.tight_layout()
    plt.show()

def plot_pareto(points, labels=None, xlabel='Fairness (lower is better)', ylabel='Accuracy (higher is better)'):
    """
    points: list of (fairness_metric, accuracy) tuples
    """
    df = pd.DataFrame(points, columns=['fairness','accuracy'])
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='fairness', y='accuracy', data=df)
    if labels is not None:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (df['fairness'].iat[i], df['accuracy'].iat[i]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Pareto: fairness vs accuracy")
    plt.show()
