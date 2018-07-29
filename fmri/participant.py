import os
import subprocess

def shell(cmd, split=True):
    """ utility function to call shell commands from python,
    it will return the stdout from the command that is run,
    as you would otherwise see it from the terminal
    """
    if not split:
        cmd = cmd.split()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = process.communicate()
    return out.decode("utf-8").split('\n')

def isdir_isempty(path):
    """ this is a utility function to check if the given path
    exists and if the deepest directory has any files in it
    path: a string - path to a directory
    """
    cur_path = os.getcwd()
    if os.path.isdir(path):
        last_dir = path.split('/')[-1]
        os.chdir(path)
        if len(os.listdir('.')):
            dir_empty = False
    else:
        last_dir, dir_empty = None, True
    os.chdir(cur_path)
    return dict(last_dir=last_dir, dir_empty=dir_empty)

class Participant(object):
    def __init__(self, base_path, par_name, use_num, use_type=None, use_mni=False):
        self.base_path = base_path
        self.par_name = par_name
        self.use_type = use_type
        self.use_num = use_num
        self.use_mni = use_mni
        self.run()

    def _is_path(self):
        if not os.path.isdir(self.base_path):
            raise Exception("base path to l1 analysis not found")
        else:
            self.path_found = True

    def _l1_outputs(self):
        data_path = os.path.join(self.base_path, self.par_name)
        if not os.path.isdir(data_path):
            msg = "participant {} outputs not found".format(self.par_name)
            raise Exception(msg)
        else:
            self.par_path = data_path
            self.dirs = [isdir_isempty(os.path.join(self.par_path, p)) 
                         for p in os.listdir(self.par_path)]
            
    def _use_type(self):
        if self.use_type is None:
            print("attempting to default to copes")
            for inner_dict in self.dirs:
                if "copes" in inner_dict.values() and not inner_dict["dir_empty"]:
                    self.use_type = "copes"
            if self.use_type is None:
                raise Exception("either could not find copes dir or it was empty")
         
    def _file(self):
        fnames = {"copes": "cope{}.nii.gz", "zstats": "zstat{}.nii.gz"}
        if self.use_mni:
            self.afile = os.path.join(
                self.base_path,
                self.par_name,
                self.use_type,
                "mni",
                fnames[self.use_type].format(self.use_num)
            )
        else:
            self.afile = os.path.join(
                self.base_path, 
                self.par_name,
                self.use_type,
                fnames[self.use_type].format(self.use_num)
            )    

    def run(self):
        self._is_path()
        self._l1_outputs()
        self._use_type()
        self._file()    
        

