import pandas as pd
import os 


if __name__ == '__main__':
    project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    data_dir = os.path.join(project_dir, 'local_database', 'MM-DatasetFallacies')

    # Step 1: load dataset
    # TODO: change df path

    df_path = os.path.join(data_dir, 'full', 'dataset.csv')
    df = pd.read_csv(df_path, sep = '\t')

    print(df.head())
    # Step 2: remove duplicated Snippets 
    df = df.drop_duplicates(subset=['Dialogue', 'Snippet'])

    # Step 3: save dataset
    df_path = os.path.join(data_dir, 'no_duplicates', 'dataset_dial.csv')
    df.to_csv(df_path, sep='\t', index=False)