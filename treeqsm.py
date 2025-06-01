"""
Python adaptation and extension of TREEQSM:

% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.

Version: 0.0.1
Date: 9 Feb 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""
from numba import jit
import numpy as np
from datetime import datetime
from main_steps.cover_sets import cover_sets
from main_steps.tree_sets import tree_sets
from main_steps.segments import segments
from main_steps.relative_size import relative_size
from main_steps.cylinders import cylinders
from main_steps.branches import branches
from main_steps.tree_data import tree_data
from main_steps.point_model_distance import point_model_distance
from Utils.Utils import display_time,load_point_cloud
from tools.define_input import define_input
import Utils.Utils as Utils
import time
import cProfile
import pstats
import warnings
warnings.filterwarnings('ignore')
from main_steps.correct_segments import correct_segments
from main_steps.cube_volume import cube_volume
from plotting.cylinders_line_plotting import cylinders_line_plotting
from plotting.point_cloud_plotting import point_cloud_plotting
from plotting.qsm_plotting import qsm_plotting
import sys
import json

def test():
    # file_path = r'C:\Users\johnh\Documents\LiDAR\A-04-7007_post.las'
    # file_path = r'/Users/johnhagood/Documents/LiDAR/segmented_trees/A-04-7007_post.las'
    # file_path = r'/Users/johnhagood/Documents/LiDAR/segmented_trees/tree_1.las'
    file_path = r'/Users/johnhagood/Documents/LiDAR/segmented_trees/test_palm.xyz'
    # file_path = r'E:\5-Study\OMSCS\CS8903_Research\TreeQSM\PyTLidar\Dataset\tree_1.las'
    points = load_point_cloud(file_path,0)
    if points is not None:
        sys.stdout.write(f"Loaded point cloud with {points.shape[0]} points.")
    # Step 3: Define inputs for TreeQSM
    points = points - np.mean(points,axis = 0)

    inputs = define_input(points, 1, 1, 1)[0]

    #specific inputs for testing
    inputs['PatchDiam1'] = [0.05]
    inputs['PatchDiam2Min'] = [0.03]
    inputs['PatchDiam2Max'] = [0.12]
    inputs['BallRad1'] = [0.06]
    inputs['BallRad2'] = [0.13]
    inputs['plot'] = 0
    treeqsm(points,inputs)



def treeqsm(P,inputs,batch =0,processing_queue = None):
    try:
        # Save computation times for modeling steps
        
        Time = np.zeros(12)
        Date = np.zeros((2, 6))
        Date[0, :] = datetime.now().timetuple()[:6]  # Starting date

        # Names of the steps to display
        name = np.array([
            'Cover sets      ',
            'Tree sets       ',
            'Initial segments',
            'Final segments  ',
            'Cylinders       ',
            'Branch & data   ',
            'Distances       '
        ])

        if inputs['disp'] > 0:
            sys.stdout.write('---------------\n')
            sys.stdout.write(f'  {inputs["name"]}, Tree = {inputs["tree"]}, Model = {inputs["model"]}\n')

        # Input parameters
        PatchDiam1 = inputs['PatchDiam1']
        PatchDiam2Min = np.array(inputs['PatchDiam2Min'])
        PatchDiam2Max = np.array(inputs['PatchDiam2Max'])
        BallRad1 =inputs['BallRad1']
        BallRad2 = np.array(inputs['BallRad2'])
        nd = len(PatchDiam1)
        ni = len(PatchDiam2Min)
        na = len(PatchDiam2Max)

        if inputs['disp'] == 2:
            # Display parameter values
            sys.stdout.write(f'  PatchDiam1 = {PatchDiam1}\n')
            sys.stdout.write(f'  BallRad1 = {BallRad1}\n')
            sys.stdout.write(f'  PatchDiam2Min = {PatchDiam2Min}\n')
            sys.stdout.write(f'  PatchDiam2Max = {PatchDiam2Max}\n')
            sys.stdout.write(f'  BallRad2 = {BallRad2}\n')
            sys.stdout.write(f'  Tria = {inputs["Tria"]}, OnlyTree = {inputs["OnlyTree"]}\n')
            sys.stdout.write('Progress:\n')

        # Make the point cloud into proper form
        # Only 3-dimensional data
        if P.shape[1] > 3:
            P = P[:, :3]

        # Only double precision data
        if not np.issubdtype(P.dtype, np.floating):
            P = P.astype(np.float64)

        # Initialize the output file
        QSM = {'cylinder': [], 'branch': [], 'treedata': [], 'rundata': [], 'pmdistance': [], 'triangulation': []}

        # Reconstruct QSMs
        models = []
        cyl_htmls = []
        nmodel = 0
        iter = batch*10
        for h in range(nd):
            start_time = datetime.now()
            Inputs = inputs.copy()
            Inputs['PatchDiam1'] = PatchDiam1[h]
            Inputs['BallRad1'] = BallRad1[h]
            
            if nd > 1 and inputs['disp'] >= 1:
                sys.stdout.write('  -----------------\n')
                sys.stdout.write(f'  PatchDiam1 = {PatchDiam1[h]}\n')
                sys.stdout.write('  -----------------\n')

            # Generate cover sets
            cover1 = cover_sets(P, Inputs)
            Time[0] = (datetime.now() - start_time).total_seconds()
            
            if inputs['disp'] == 2:
                display_time(Time[0], Time[0], name[0], 1)

            # Determine tree sets and update neighbors
            cover1, Base, Forb = tree_sets(P, cover1, Inputs)
            Time[1] = (datetime.now() - start_time).total_seconds() - Time[0]
            
            if inputs['disp'] == 2:
                display_time(Time[1], np.sum(Time[:2]), name[1], 1)
            
            start_time = datetime.now()
            segment1 = segments(cover1,Base,Forb)
            Time[2] = (datetime.now() - start_time).total_seconds()
            if inputs['disp'] == 2:
                display_time(Time[2], np.sum(Time[:3]), name[2], 1)
            
            start_time = datetime.now()
            segment1 = correct_segments(P,cover1,segment1,Inputs,0,1,1)
            Time[3] = (datetime.now() - start_time).total_seconds()
            if inputs['disp'] == 2:
                display_time(Time[3], np.sum(Time[:4]), name[3], 1)
            for i in range(na):
                # Modify inputs
                Inputs['PatchDiam2Max'] = PatchDiam2Max[i]
                Inputs['BallRad2'] = BallRad2[i]
                
                if na > 1 and inputs['disp'] >= 1:
                    sys.stdout('    -----------------\n')
                    sys.stdout.write(f'    PatchDiam2Max = {PatchDiam2Max[i]}\n')
                    sys.stdout.write('    -----------------\n')
                
                for j in range(ni):
                    start_time = datetime.now()
                    
                    # Modify inputs
                    Inputs['PatchDiam2Min'] = PatchDiam2Min[j]
                    
                    if ni > 1 and inputs['disp'] >= 1:
                        sys.stdout.write('      -----------------\n')
                        sys.stdout.write(f'      PatchDiam2Min = {PatchDiam2Min[j]}\n')
                        sys.stdout.write('      -----------------\n')
                    
                    # Generate new cover sets
                    RS = relative_size(P, cover1, segment1)
                    
                    # Generate new cover
                    cover2 = cover_sets(P, Inputs, RS)
                    Time[4] = (datetime.now() - start_time).total_seconds() #
                    if inputs['disp'] == 2:
                        display_time(Time[4], sum(Time[:5]), name[0], 1)

                    # Determine tree sets and update neighbors
                    start_time = datetime.now()
                    cover2, Base, Forb = tree_sets(P, cover2, Inputs, segment1)
                    Time[5] = (datetime.now() - start_time).total_seconds()
                    
                    if inputs['disp'] == 2:
                        display_time(Time[5], sum(Time[:6]), name[1], 1)
                    
                    start_time = datetime.now()
                    # Determine segments
                    segment2 = segments(cover2, Base, Forb)
                    Time[6] = (datetime.now() - start_time).total_seconds()# - sum(Time[4:6])
                    
                    if inputs['disp'] == 2:
                        display_time(Time[6], sum(Time[:7]), name[2], 1)
                    
                    start_time = datetime.now()
                    # Correct segments
                    segment2 = correct_segments(P, cover2, segment2, Inputs, 1, 1, 0)
                    Time[7] = (datetime.now() - start_time).total_seconds() #- sum(Time[6:8])
                    
                    if inputs['disp'] == 2:
                        display_time(Time[7], sum(Time[:8]), name[3], 1)

                    start_time = datetime.now()
                    cylinder = cylinders(P,cover2,segment2,Inputs)
                    Time[8] = (datetime.now() - start_time).total_seconds() #- sum(Time[6:8])
                    if inputs['disp'] == 2:
                        display_time(Time[8], sum(Time[:9]), name[4], 1)
                    if np.size(cylinder['radius']) > 0:
                    
                    

                        # calculate cylinder volume in cube partitions
                        # c_volume = cube_volume(P, cylinder, 1)  # 1m cube, revise if needed
                        #sys.stdout.write(c_volume[10, 10, 10])
                        

                        
                        branch= branches(cylinder)
                        
                        
                        # Extract trunk point cloud
                        T = segment2['segments'][0]  # Assuming segment2 is a dictionary with 'segments' key
                        T = np.concatenate(T)  # Vertically concatenate segments
                        T = np.concatenate([cover2['ball'][idx] for idx in T])  # Extract points from cover2
                        trunk = P[T, :]  # Point cloud of the trunk

                        # Compute attributes and distributions from the cylinder model
                        start_time = datetime.now()
                        treedata, triangulation = tree_data(cylinder, branch, trunk, inputs,iter )
                        Time[9] = (datetime.now() - start_time).total_seconds() 

                        # Display time for tree_data computation
                        if inputs['disp'] == 2:
                            display_time(Time[9], sum(Time[:10]), name[5], 1)

                        # Compute point-model distances
                        if inputs['Dist']:
                            pmdis = point_model_distance(P, cylinder)

                            # Display mean point-model distances and surface coverages
                            if inputs['disp'] >= 1:
                                D = [pmdis['TrunkMean'], pmdis['BranchMean'],
                                    pmdis['Branch1Mean'], pmdis['Branch2Mean']]
                                D = np.round(10000 * np.array(D)) / 10

                                T = cylinder['branch'] == 0
                                B1 = cylinder['BranchOrder'] == 1
                                B2 = cylinder['BranchOrder'] == 2
                                SC = 100 * cylinder['SurfCov']
                                S = [np.mean(SC[T]), np.mean(SC[~T]), np.mean(SC[B1]), np.mean(SC[B2])]
                                S = np.round(10 * np.array(S)) / 10

                                sys.stdout.write('  ----------\n')
                                sys.stdout.write(f'  PatchDiam1 = {PatchDiam1[h]}, PatchDiam2Max = {PatchDiam2Max[i]}, PatchDiam2Min = {PatchDiam2Min[j]}\n')
                                sys.stdout.write('  Distances and surface coverages for trunk, branch, 1branch, 2branch:\n')
                                sys.stdout.write(f'  Average cylinder-point distance: {D[0]}  {D[1]}  {D[2]}  {D[3]} mm\n')
                                sys.stdout.write(f'  Average surface coverage: {S[0]}  {S[1]}  {S[2]}  {S[3]} %\n')
                                sys.stdout.write('  ----------\n')

                            Time[10] = (datetime.now() -start_time).total_seconds()
                            if inputs['disp'] == 2:
                                display_time(Time[10], sum(Time[:11]), name[6], 1)

                        # Reconstruct the output "QSM"
                        Date[1] = datetime.now().timetuple()[:6]  # Update date
                        Time[11] = sum(Time[:11])

                        qsm = {
                            'cylinder': cylinder,
                            'branch': branch,
                            'treedata': treedata,
                            'rundata': {
                                'inputs': inputs,
                                'time': np.float32(Time),
                                'date': np.float32(Date),
                                'version': '2.4.1'
                            
                            },
                            'cover': cover2,
                            'segment': segment2,
                            'points':P,
                            'PatchDiam1': PatchDiam1[h],
                            'PatchDiam2Max': PatchDiam2Max[i],
                            'PatchDiam2Min': PatchDiam2Min[j],
                        }

                        if inputs['Dist']:
                            qsm['pmdistance'] = pmdis
                        if inputs['Tria']:
                            qsm['triangulation'] = triangulation

                        nmodel += 1
                        for key in QSM.keys():
                            try:
                                QSM[key].append(qsm[key])
                            except KeyError:
                                pass

                        # Save the output into results folder
                        if inputs['savemat']:
                            str = f"{inputs['name']}_t{inputs['tree']}_m{inputs['model']}"
                            if nd > 1 or na > 1 or ni > 1:
                                if nd > 1:
                                    str += f"_D{PatchDiam1[h]}"
                                if na > 1:
                                    str += f"_DA{PatchDiam2Max[i]}"
                                if ni > 1:
                                    str += f"_DI{PatchDiam2Min[j]}"
                            np.savez(f"results_QSM_{str}.npz", QSM=QSM)

                        if inputs['savetxt']:
                            if nd > 1 or na > 1 or ni > 1:
                                str = f"{inputs['name']}_t{inputs['tree']}_m{inputs['model']}"
                                if nd > 1:
                                    str += f"_D{PatchDiam1[h]}"
                                if na > 1:
                                    str += f"_DA{PatchDiam2Max[i]}"
                                if ni > 1:
                                    str += f"_DI{PatchDiam2Min[j]}"
                            else:
                                str = f"{inputs['name']}_t{inputs['tree']}_m{inputs['model']}"
                            Utils.save_model_text(qsm, str)

                        # Plot models and segmentations

                        if nd > 1 or na > 1 or ni > 1:
                            str = f"{inputs['name']}_t{inputs['tree']}_m{inputs['model']}"
                            if nd > 1:
                                str += f"_D{PatchDiam1[h]}"
                            if na > 1:
                                str += f"_DA{PatchDiam2Max[i]}"
                            if ni > 1:
                                str += f"_DI{PatchDiam2Min[j]}"
                        else:
                            str = f"{inputs['name']}_t{inputs['tree']}_m{inputs['model']}"
                        fidelity = min(100000/ P.shape[0],1)  # Adjust fidelity based on point cloud size
                        base_fig = point_cloud_plotting(P, subset=True,fidelity=fidelity,marker_size=1,return_html=False)
                        qsm_plotting(P,cover2,segment2,qsm,return_html=True,subset = True, fidelity=fidelity,marker_size=1)
                        fig,cyl_html = cylinders_line_plotting(cylinder, 100, 8,str,False,base_fig=base_fig)

                        cyl_htmls.append(cyl_html)

                        models.append(qsm)
                        cyl_htmls.append(cyl_html)
                        iter+=1
        response = Utils.package_outputs(models,cyl_htmls)    
        sys.stdout.write(json.dumps(response))   
        if processing_queue is not None:
            processing_queue.put([batch,models,cyl_htmls])
        return models, cyl_htmls
    except Exception as e:
        sys.stderr.write(f"An error occurred: {e}\n")
        if processing_queue is not None:
            processing_queue.put([batch, "ERROR", "ERROR"])
        return "ERROR", "ERROR"


if __name__ == "__main__":
    # cProfile.run("test()",filename="results.txt",sort=1)
    # stats = pstats.Stats('results.txt')
    # stats.sort_stats('tottime')
    # stats.reverse_order()
    # stats.print_stats()
    try:
        filename = sys.argv[1]
        inputs = sys.argv[2].split()

        points = load_point_cloud(filename,inputs[0])
        if points is not None:
            sys.stdout.write(f"Loaded point cloud with {points.shape[0]} points.\n")
        # Step 3: Define inputs for TreeQSM
        points = points - np.mean(points,axis = 0)
        inputs = define_input(points, inputs[1], inputs[2], inputs[3])[0]
        inputs['plot'] = 0
        treeqsm(points,inputs)
    except:
        test()