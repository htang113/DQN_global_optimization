This is the code repository for using deep Q learning to accelerate global energy minimization in the potential energy surface of a particle system.

In the first part (folder DQN), we implemented deep Q network for the training. 
Running this training requires modules: PyTorch, torch_cluster, torch_scatter, e3nn, ase, and some common packages in anaconda3
This program uses multiprocessing
The training can be launched by running DQN/workspace.py. 
The training generates a series of log files (log0, log1, ...) and model files (100model3.pt, ...)
Training curves (Fig. 2 in the report) are generated by putting the log files into DQN/plot and running DQN/plot/ave_Fig2.py

In the second part (folder DDQN), we implemented double deep Q network for the training. 
Running this training requires modules: PyTorch (cuda enabled version), torch_cluster, torch_scatter, e3nn, ase, and some common packages in anaconda3
This program uses GPU acceleration
The training can be launched by running DDQN/workspace.py. 
The training generates a series of log files (log0, log3) and model files (100model3.pt, ...)
Training curves (Fig. 3a in the report) are generated by putting the log files into DDQN/plot and running DDQN/plot/ave_Fig3a.py

In the second part (folder deploy), we deploy the model trained by the above methods and sample 200 trajectories. 
Running this training requires modules: PyTorch, torch_cluster, torch_scatter, e3nn, ase, and some common packages in anaconda3
The trajectory generation using the trained model can be launched by putting the trained model files into deploy/generate_trajectory and running deploy/generate_trajectory/workspace.py.
Here we already put the trained model files model_DQN.pt and model_DDQN.pt into deploy/generate_trajectory folder.
Running the trajectory generation outputs trajectory files (trajectory_DQN.json, trajectory_DDQN.json).
We also setting temperature T as 1E6 in deploy/generate_trajectory/workspace.py to generate a random policy and output trajectory_random.json.
Fig. 3b in the report is generated by putting all trajectory files into deploy/analysis_plot/ folder and running deploy/analysis_plot/traj_Fig3b.py.
Data for plotting Fig. 4 is generated by running deploy/analysis_plot/animation.py, which outputs Fig4a.txt and XDATCAR_Fig4b. 
Fig. 4a is then generated by plotting Fig4a.txt using OriginLab, and Fig. 4b is generated by visualize XDATCAR_Fig4b using OVITO.
