# Master-Thesis-Scripts
The scripts included are used to retrace the results from my master's thesis project titled: "Transition to multiple attractor phase in the many-species
Lotka-Volterra model"

1. The scripts were written in python ver 3.8, using Spyder ver 5.0.5
2. Packages needed: numpy, scipy, matplotlib, pylab, plotly (used for interactive saved plots)
3. The scripts are supposed to be run in a certain order. sigma_crit->f123(optional)->Solving_CdZ->Plot Results
4. The results of some are required for others. Which script is required, along with script purpose, input, output are detailed at the begining of each script as comments
5. The scripts save all required (and some addendum) data in the specified HOME directory. Make sure you find all instances of HOME in the script and adjust it to the desired path, else python will return an error when trying to run the script.

Script description:

1.
Name: sigma_crit
Purpose: find sigma_crit for given mu and lambda
Previous script results required: none

2.
Name: f123 
Purpose: Calculate f1,2,3(omega, Omega, Zbar)'s average over Zbar
Previous script results required: sigma_crit.npy 
NOTE: This script is optional to run. Its function is also contained in the script Solving_C_dZ

3. 
Name: Solving_CdZ
Purpose: 
    1.Calculating f1,2,3(Optional)
    2.Finding numerical solution to Eq (44) in thesis
Previous script results required: 
    1. sigma_crit.npy
    2. f123.npy(Optional)
    3. Solving C_dZ(Optional - to plug a C_dZ solution with neighboring parameters into solution finder) 
NOTE: The functionality of script f123 is imbeded in this one for convenience. When run, the script will ask if you wish to calculate f123 from scratch, or load previously calculated f123 results. 

4. 
Name: Plot Results
Purpose: Compile all the solutions to eq (44) found for all lambda and sigma, calculate Q, the time scale and plot graphs
Previous script results required: 
    1. Solving C_dZ.npy for all lambda and sigma that is needed
    2. sigma_crit.npy
    
5.
Name:f23 scaling check
Purpose: Checking the scaling of f2, f3 at lambda=0 for Omega > omega
Previous script results required: none
