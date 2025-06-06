from treeqsm import treeqsm
from treeqsm import calculate_optimal
import os
import sys
import numpy as np
import pandas as pd
from tools.define_input import define_input
from Utils.Utils import load_point_cloud
import warnings
import multiprocessing as mp
import Utils.Utils as Utils

warnings.filterwarnings('ignore')

class BatchQSM():
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
    def run(self):
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
            input_params['name'] = self.files[i]+self.runname
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
                    process_output((batch,data,plot)) 
                except:
                    print(f"An error occured on file {input_params['name']}. Please try again. Consider checking the console and reporting the bug to us.")  
            process+=num_cores
            
            
        print("Processing Complete.\n")

def process_output(output):
    batch,models, cyl_htmls = output

    for metric in parsed_args["Optimum"]:
        optimum = calculate_optimal(models,metric)
        npd1 = models[optimum]['PatchDiam1']
        max_pd = models[optimum]['PatchDiam2Max']
        min_pd = models[optimum]['PatchDiam2Min']
        file = models[optimum]['rundata']['inputs']['name']
        sys.stdout.write(f"File: {file}, Optimal PatchDiam1: {npd1}, Max PatchDiam: {max_pd}, Min PatchDiam: {min_pd}\n")


if __name__== "__main__":

    folder = sys.argv[1]
    
    parsed_args = Utils.parse_args(sys.argv[2:])
    
    
    if parsed_args not in ["ERROR","Help"]:
        print(parsed_args)
        threshold = parsed_args["Intensity"]
        files = os.listdir(folder)
        
        files = [f for f in files if f.endswith('.las') or f.endswith('.laz')]

        batch_process = BatchQSM(folder,files,parsed_args)
        batch_process.run()
