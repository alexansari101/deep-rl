import os
import shutil

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


def get_output_folder(parent_dir, env_name, load, trial=None, tmp=False):
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
    if tmp:
        d = os.path.join(parent_dir, 'tmp')
        for f in os.listdir(d):
            p = d+'/'+f
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)
        return d
        
        
    if trial is None:
        trial = get_last_experiment(parent_dir, env_name)
        if not load:
            trial += 1
            print()
            print('Starting new trial ' + str(trial))
            print()
        
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(trial)

    return parent_dir

def copy_files(outdir):
    """Copies files to the outdir to store complete script with each trial"""
    codedir = outdir+"/code"
    os.makedirs(codedir)

    code = []
    exclude = set(['trials'])
    for root, dirs, files in os.walk(".", topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]
    for r, f in code:
        os.makedirs(codedir+'/'+r, exist_ok=True)
        shutil.copy2(r+'/'+f, codedir+'/'+r+'/'+f)
            
        
    
def write_readme(outdir):
    inp = input('Write something about this trial:\n')
    target = open(outdir + '/readme', 'w')
    target.write(inp + '\n')
    target.close()
    print('Great! Starting work')
