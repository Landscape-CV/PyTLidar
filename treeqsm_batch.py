if __package__:
    from .treeqsm import calculate_optimal
    from .pipeline import load_cloud, centre, build_inputs, run_batch
    from .Utils import Utils
else:
    from treeqsm import calculate_optimal
    from pipeline import load_cloud, centre, build_inputs, run_batch
    import Utils.Utils as Utils
import os
import sys

import warnings
import traceback
import multiprocessing as mp


warnings.filterwarnings('ignore')

class BatchQSM():
    """
    Runs TreeQSM over every .las/.laz file in a folder, several trees at a time in worker processes.
    """
    def __init__(self, folder,files,args):
        self.folder = folder
        self.files = files
        self.args = args
        self.intensity_threshold = float(args["Intensity"])
        self.inputs = {"PatchDiam1":args["PatchDiam1"],"PatchDiam2Min":args["PatchDiam2Min"],"PatchDiam2Max":args["PatchDiam2Max"]}
        self.generate_values = not args["Custom"]
        self.num_cores = args["Cores"]
        self.normalize = args["Normalize"]
        self.runname = args["Name"]
        self.verbose = args["Verbose"]
        self.directory = args["Directory"]
        self.saved_files = []
    def file_cleanup(self):
        """
        Cleans up the files saved during the run, removing those that were not saved by the batch process.
        """
        if len(self.saved_files) == 0:
            print("No files were saved from this run.")
            return
        original_location = os.getcwd()
        if self.args["Directory"] is not None:
            os.chdir(self.args["Directory"])
            os.chdir("results")
        else:
            os.chdir('results')
        if  self.args["Optimum"] != []:
            sys.stdout.write("Removing non-optimal files from results folder...\n")
            for file in os.listdir():
                remove = True
                for string,filename in self.saved_files:
                    if string in file and filename in file:
                        remove = False
                if remove:
                    os.remove(file)
        os.chdir(original_location)
    def run(self):
        """
        Handler for creating Parallel TreeQSM processes
        """
        try:
            num_cores = int(self.num_cores)
            if num_cores >mp.cpu_count():
                raise Exception()
        except:
            num_cores = mp.cpu_count()
            print(f"Invalid number of cores specified. Using {num_cores} cores instead.\n")
        clouds = [load_cloud(os.path.join(self.folder, file), self.intensity_threshold) for file in self.files]
        if self.normalize:
            clouds = [centre(cloud, z=False) for cloud in clouds]
        names = [file.replace(".las","").replace(".laz","")+self.runname for file in self.files]
        patch = (self.inputs['PatchDiam1'], self.inputs['PatchDiam2Min'], self.inputs['PatchDiam2Max'])
        settings = dict(names=names, savemat=0, savetxt=1, plot=0, disp=2 if self.verbose else 0)
        if self.generate_values:
            inputs = build_inputs(clouds, n_patchdiam=patch, **settings)
        else:
            inputs = build_inputs(clouds, custom=patch, **settings)
        clouds = [cloud for cloud in clouds if cloud.shape[0] > 0]  # build_inputs skips empty clouds
        run_batch(clouds, inputs, num_cores, self.directory, on_result=self.save_optimal)
        self.file_cleanup()
        print("Processing Complete.\n")

    def save_optimal(self, index, models, cyl_htmls):
        try:
            self.saved_files += process_output((index, models, cyl_htmls), directory=self.directory, args=self.args)
        except Exception as e:
            print(e)
            print(traceback.format_exc())

def process_output(output,directory,args):
    """Takes output of TreeQSM and processes it to save the optimal models and their metrics.
    This will save the relevant files as well as save the data to be shown in the GUI.

    Args:
        output (tuple): direct output of TreeQSM
        directory (str): directory to save files
        args (dict): parsed command line arguments; args["Optimum"] lists the metrics to keep

    Returns:
        list: list of files corresponding to optimal models, to be saved and not deleted.
    """
    original_location = os.getcwd()
    if directory is not None:
        os.chdir(directory)

    batch,models, cyl_htmls = output

    saved_files = []
    for metric in args["Optimum"]:
        optimum,value,metric_data = calculate_optimal(models,metric)
        npd1 = models[optimum]['PatchDiam1']
        max_pd = models[optimum]['PatchDiam2Max']
        min_pd = models[optimum]['PatchDiam2Min']
        file = models[optimum]['rundata']['inputs']['name']
        sys.stdout.write(f"File: {file}, For Metric {metric}, Optimal PatchDiam1: {npd1}, Max PatchDiam: {max_pd}, Min PatchDiam: {min_pd}\n\tValue is {value}\n")

        string = models[optimum]["file_id"]
        filename = f"{models[optimum]['rundata']['inputs']['name']}_t{models[optimum]['rundata']['inputs']['tree']}_m{models[optimum]['rundata']['inputs']['model']}"
        Utils.save_fit(metric_data[3]["CylDist"],os.path.join("results",filename+"_"+string))
        saved_files.append((string,filename))
    os.chdir(original_location)
    return saved_files

if __name__== "__main__":


    try:
        folder = sys.argv[1]
    except:
        print("No arguments found, for instructions on how to run this script, please run with the --help flag.")
        sys.exit(1)
    parsed_args = Utils.parse_args(sys.argv[2:])


    if parsed_args not in ["ERROR","Help"]:
        print(parsed_args)
        files = os.listdir(folder)

        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]

        batch_process = BatchQSM(folder,files,parsed_args)
        batch_process.run()
