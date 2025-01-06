import wandb

def log_dataset_to_wandb(dataset, project_name, run_name, split_name="dataset_split"):
    wandb.init(
        project=project_name,
        name=run_name,
        settings=wandb.Settings(_service_wait=300),
    )

    data_table = wandb.Table(columns=dataset.column_names)

    # Add rows from the dataset
    for row in dataset:
        data_table.add_data(*[row[col] for col in dataset.column_names])

    # Log the table to WandB
    wandb.log({split_name: data_table})

    wandb.finish()