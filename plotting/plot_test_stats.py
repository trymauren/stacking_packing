import matplotlib.pyplot as plt
import git
import seaborn as sns
import textwrap
plt.set_loglevel('WARNING')
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
plt.style.use(path_to_root + "/thesis/plot_config.mplstyle")

metrics_range_01 = [
    'stack_compactness', 'stack_gap_ratio', 'stack_stability_bool'
]


def plot_individual_stats(df, metrics, metric_names, algorithms, plot_dir):

    for metric, name in zip(metrics, metric_names):

        fig, ax = plt.subplots(figsize=(6*0.3, 6*0.3))
        data = [df.loc[df['algorithm'] == alg, metric].dropna().values for alg in algorithms]
        if metric in metrics_range_01:
            ax.set_ylim(0, 1)

        sns.despine()
        sns.violinplot(data=data, palette='flare', ax=ax)
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms)
        ax.set_ylabel(name)
        plt.savefig(f'{plot_dir}/{metric}.pdf', backend='pgf')
        plt.close(fig)


def plot_inference_time(data, algorithms, plot_dir):

    fig, ax = plt.subplots(figsize=(6*0.3, 6*0.3))
    sns.despine()
    sns.violinplot(data=data, palette='flare', ax=ax)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms)
    ax.set_ylabel('Time (s) per instance')
    plt.savefig(f'{plot_dir}/inference_time.pdf', backend='pgf')
    plt.close(fig)


def plot_combined_stats(
    df, metrics, metric_names, algorithms,
    plot_dir, wrap_width=12
):

    df_long = (
        df
        .loc[:, ['algorithm'] + metrics]
        .melt(id_vars='algorithm',
              value_vars=metrics,
              var_name='metric',
              value_name='value')
        .dropna(subset=['value'])
    )

    name_map = dict(zip(metrics, metric_names))
    df_long['metric_name'] = df_long['metric'].map(name_map)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.despine()
    g = sns.barplot(
        data=df_long,
        x='metric_name',
        y='value',
        hue='algorithm',
        errorbar=None,          # disable error bars (shows raw means)
        ax=ax,
        # linewidth=2,
        # edgecolor='w'
    )
    g.set_xlabel('')
    g.set_ylabel('')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    raw_labels = [t.get_text() for t in ax.get_xticklabels()]

    wrapped_labels = [
        "\n".join(textwrap.wrap(label, width=wrap_width))
        for label in raw_labels
    ]
    ax.set_xticklabels(wrapped_labels)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/combined_metrics', backend='pgf')
    plt.close(fig)
