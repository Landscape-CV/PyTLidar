try:
    from .treeqsm import treeqsm
    from .treeqsm import calculate_optimal
    from .Utils.define_input import define_input
    from .Utils.Utils import load_point_cloud
    from .Utils import Utils
except ImportError:
    from treeqsm import treeqsm
    from treeqsm import calculate_optimal
    from Utils.define_input import define_input
    from Utils.Utils import load_point_cloud
    import Utils.Utils as Utils
import os
import sys
import numpy as np
import pandas as pd

import warnings
import traceback
import multiprocessing as mp


warnings.filterwarnings('ignore')

class BatchQSM():
    """
    A class to handle multiprocess objects to perform batch processing of TreeQSM
    """
    def __init__(self, folder,files,args):
        self.folder = folder
        self.files = files
        self.intensity_threshold = float(threshold)
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
        if parsed_args["Directory"] is not None:
            os.chdir(parsed_args["Directory"])
            os.chdir("results")
        else:
            os.chdir('results')
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
        clouds = []
        for i, file in enumerate(self.files):
            point_cloud = load_point_cloud(os.path.join(self.folder, file), self.intensity_threshold)
            if point_cloud is not None:
                point_cloud = point_cloud - np.mean(point_cloud,axis = 0) if self.normalize else point_cloud
                clouds.append(point_cloud)
        if self.generate_values:
            inputs = define_input(clouds,self.inputs['PatchDiam1'], self.inputs['PatchDiam2Min'], self.inputs['PatchDiam2Max'])
        else:
            inputs = define_input(clouds,1,1,1)
            for cld in inputs:
                cld['PatchDiam1'] = self.inputs['PatchDiam1']
                cld['PatchDiam2Min'] = self.inputs['PatchDiam2Min']
                cld['PatchDiam2Max'] = self.inputs['PatchDiam2Max']
                cld['BallRad1'] = [i+.01 for i in cld['PatchDiam1']]
                cld['BallRad2'] = [i+.01 for i in cld['PatchDiam2Max']]
        for i, input_params in enumerate(inputs):
            input_params['name'] = self.files[i].replace(".las","").replace(".laz","")+self.runname
            input_params['savemat'] = 0
            input_params['savetxt'] = 1
            input_params["disp"] = 2 if self.verbose else 0
            input_params["plot"] = 0
            
        
    # Process each tree
        try:
            mp.set_start_method('spawn')
        except:
            pass
        Q=[]
        P=[]
        
        for i, input_params in enumerate(inputs):

            
            q = mp.Queue()
            p = mp.Process(target=treeqsm, args=(clouds[i],input_params,i,q,self.directory))
            Q.append(q)
            P.append(p)
        process = 0
    
        while process < len(inputs):
            for i in range(num_cores):
                
                if process+i > len(inputs)-1:
                    break
                print(f"Processing {inputs[process+i]['name']}. This may take several minutes...\n")
                
                P[process+i].start()

            for i in range(num_cores):
                if process+i > len(inputs)-1:
                    break
                q=Q[process+i]
                p = P[process+i]
                try:
                    batch,data,plot = q.get()
                    if data =="ERROR":
                        raise Exception("Error in processing file")
                    p.join()
                    # data,plot = treeqsm(clouds[i],input_params,i)
                    self.saved_files += process_output((batch,data,plot),directory = self.directory) 
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())  
            process+=num_cores
            
        self.file_cleanup()
        print("Processing Complete.\n")

def process_output(output,directory):
    """Takes output of TreeQSM and processes it to save the optimal models and their metrics.
    This will save the relevant files as well as save the data to be shown in the GUI.

    Args:
        output (tuple): direct output of TreeQSM
        directory (str): directory to save files

    Returns:
        list: list of files corresponding to optimal models, to be saved and not deleted. 
    """
    original_location = os.getcwd()
    if directory is not None:
        os.chdir(directory)
    
    batch,models, cyl_htmls = output

    saved_files = []
    for metric in parsed_args["Optimum"]:
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
        threshold = parsed_args["Intensity"]
        files = os.listdir(folder)
        
        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]

        batch_process = BatchQSM(folder,files,parsed_args)
        batch_process.run()
