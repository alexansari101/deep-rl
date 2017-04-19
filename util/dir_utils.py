import os

def get_last_experiment(parent_dir, env_name):
    """Searches through directory for highest numbered experiement"""
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    return experiment_id


def get_output_folder(parent_dir, env_name, load, trial=None):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    if trial is None:
        trial = get_last_experiment(parent_dir, env_name)
        if not load:
            trial += 1
            print('Starting new trial ' + str(trial))
        
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(trial)

    return parent_dir

def get_saved(parent_dir, env_name, exp_id=None, iter_num=None):
    """Loads the model and hyperparams. By default, uses the last available data
    """
    # print('getting saved')
    if exp_id is None:
        exp_id = get_last_experiment(parent_dir, env_name)
        
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(exp_id)

    # print(parent_dir)

    
    if iter_num is None:
        for file_name in os.listdir(parent_dir):
            # print('file...')
            # print(file_name)
            if not file_name.startswith('saved_model'):
                continue
            f_iter = int(file_name.split('saved_model_weights_')[-1])
            if iter_num is None or \
               f_iter > iter_num:
                iter_num = f_iter
        print('Load last iteration: ' + str(iter_num))
               
            # print(f_iter)

    return parent_dir, iter_num
