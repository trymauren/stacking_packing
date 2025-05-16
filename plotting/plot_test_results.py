
stats = [rl_stats_cat, cp_sat_stats_cat, flb_stats_cat, random_stats_cat]
algorithms = ['RL', 'CP-SAT', 'Close-to-origin', 'Random']
for stat, alg in zip(stats, algorithms):
    stat['algorithm'] = alg
metrics = [
    'stack_compactness', 'stack_stability_bool', 'completed_act_ratio'
]
names = [
    'Compactness', 'Stable stacks ratio', 'Stable actions ratio'
]
all_data = pd.concat(
    [rl_stats_cat, cp_sat_stats_cat, flb_stats_cat, random_stats_cat],
    ignore_index=True
)
plot_individual_stats(all_data, metrics, names, algorithms, PLOT_PATH)
plot_combined_stats(all_data, metrics, names, algorithms, PLOT_PATH)

time_df = {
    'RL': rl_time, 'CP-SAT': cp_sat_time,
    'Close-to-origin': flb_time, 'Random': random_time,
}
plot_inference_time(time_df, algorithms, PLOT_PATH)

# Stack plots =====================================================

discretisation = vec_env_config['env_kwargs']['discretisation']
size = vec_env_config['env_kwargs']['size']

D, W = int(size[0] * discretisation), int(size[1] * discretisation)

if cfg.plot_ind_stacks:
    IND_PATH = STACK_PLOT_PATH + '/ind_stacks'
    if not os.path.exists(IND_PATH):
        os.makedirs(IND_PATH)

    for i in range(cfg.ind_num_stacks):
        pyvista_plot_stack(
            rl_stats_cat['stacked_items'].iloc[i],
            W,
            D,
            save_path=f'{IND_PATH}/rl_stack_{i}.pdf'
        )
        pyvista_plot_stack(
            random_stats_cat['stacked_items'].iloc[i],
            W,
            D,
            save_path=f'{IND_PATH}/random_stack_{i}.pdf'
        )
        pyvista_plot_stack(
            flb_stats_cat['stacked_items'].iloc[i],
            W,
            D,
            save_path=f'{IND_PATH}/flb_stack_{i}.pdf'
        )
        pyvista_plot_stack(
            cp_sat_stats_cat['stacked_items'].iloc[i],
            W,
            D,
            save_path=f'{IND_PATH}/cp_sat_stack_{i}.pdf'
        )

if cfg.plot_collection:
    COL_PATH = STACK_PLOT_PATH + '/col_stacks'
    if not os.path.exists(COL_PATH):
        os.makedirs(COL_PATH)

    n = cfg.coll_num_stacks
    c = rl_test_callback.get_stats()['stack_compactness'].round(3)[:n]
    pyvista_plot_many_stacks(
        rl_stats_cat['stacked_items'].iloc[:n].tolist(),
        W,
        D,
        compactness=c,
        heights=(rl_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
        lbs=(rl_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
        n_to_stack=rl_stats_cat['n_to_stack'].iloc[:n].tolist(),
        n_stacked=rl_stats_cat['n_stacked'].iloc[:n].tolist(),
        shape=(1, n),
        save_path=f'{COL_PATH}/rl_stacks.pdf'
    )

    c = random_test_callback.get_stats()['stack_compactness'].round(3)[:n]
    pyvista_plot_many_stacks(
        random_stats_cat['stacked_items'].iloc[:n].tolist(),
        W,
        D,
        compactness=c,
        heights=(random_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
        lbs=(random_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
        n_to_stack=random_stats_cat['n_to_stack'].iloc[:n].tolist(),
        n_stacked=random_stats_cat['n_stacked'].iloc[:n].tolist(),
        shape=(1, n),
        save_path=f'{COL_PATH}/random_stacks.pdf'
    )

    c = flb_test_callback.get_stats()['stack_compactness'].round(3)[:n]
    pyvista_plot_many_stacks(
        flb_stats_cat['stacked_items'].iloc[:n].tolist(),
        W,
        D,
        compactness=c,
        heights=(flb_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
        lbs=(flb_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
        n_to_stack=flb_stats_cat['n_to_stack'].iloc[:n].tolist(),
        n_stacked=flb_stats_cat['n_stacked'].iloc[:n].tolist(),
        shape=(1, n),
        save_path=f'{COL_PATH}/flb_stacks.pdf'
    )

    c = cp_sat_stats_cat['stack_compactness'].round(3)[:n]
    pyvista_plot_many_stacks(
        cp_sat_stats_cat['stacked_items'].iloc[:n].tolist(),
        W,
        D,
        compactness=c,
        heights=(cp_sat_stats_cat['stack_height'].iloc[:n]*discretisation).tolist(),
        lbs=(cp_sat_stats_cat['height_lower_bound'].iloc[:n]*discretisation).tolist(),
        n_to_stack=cp_sat_stats_cat['n_to_stack'].iloc[:n].tolist(),
        n_stacked=cp_sat_stats_cat['n_stacked'].iloc[:n].tolist(),
        shape=(1, n),
        save_path=f'{COL_PATH}/cp_sat_stacks.pdf'
    )

if cfg.plot_comparison:
    COMP_PATH = STACK_PLOT_PATH + '/comp_stacks'
    if not os.path.exists(COMP_PATH):
        os.makedirs(COMP_PATH)

    rows = 4
    cols = 3

    c1 = rl_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
    c2 = random_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
    c3 = flb_test_callback.get_stats()['stack_compactness'].round(3)[:cols]
    c4 = cp_sat_stats_cat['stack_compactness'].round(3)[:cols]

    rl_instances = rl_stats_cat['stacked_items'].iloc[:cols].tolist()
    random_instances = random_stats_cat['stacked_items'].iloc[:cols].tolist()
    flb_instances = flb_stats_cat['stacked_items'].iloc[:cols].tolist()
    cp_sat_instances = cp_sat_stats_cat['stacked_items'].iloc[:cols].tolist()

    instances = (
        list(rl_instances) +
        list(random_instances) +
        list(flb_instances) +
        list(cp_sat_instances)
    )

    c = list(c1) + list(c2) + list(c3) + list(c4)

    pyvista_plot_many_stacks(
        instances,
        W,
        D,
        compactness=c,
        shape=(rows, cols),
        save_path=f'{COMP_PATH}/comparison.pdf'
    )