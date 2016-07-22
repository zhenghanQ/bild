********************************

INFO ON HOW TO RUN THESE SCRIPTS

********************************


***********************************************************************************************
***********************************************************************************************

IMPORTANT:
 
	If you don't know how to call(run) any or some of these scripts, STOP, DON'T RUN ANYTHING.
	First learn some basic python and bash. Otherwise you'll end up making a lot of errors.

***********************************************************************************************
***********************************************************************************************

* get_dicoms.py

	python get_dicoms.py -u user -s subid -dd /path/to/directory/where/dicoms/are/stored
	this will find the dicoms on sigma and copy them over to where you store the dicoms
	on mindhive. Once the script runs you will still have mv all of the .dcm files in the 
	Trio folder one directory up and then delete the Trio folder 

* single_use_brainplot.ipynb

	1. open the file in jupyter-notebook. 
	2. PAY ATTENTION TO THE COMMENTS AT THE TOP OF THE NOTEBOOK
	3. make sure all dependencies are installed, by this I mean python packages
	4. change the paths for "pos" and "neg" towards the end 
	5. run the cells
	
	this will create 3D plots of brains with the two (positive and negative) zstats 
	overlaid. 

* art_outliers_ratios.py

	will create a file that tells you the % of outliers per run for the task specified
        per participant. YOU CAN'T JUST RUN IT AS IS YOU WILL HAVE TO LOOK AT THE SCRIPT
        AND MAKE THE REQUIRED CHANGES.

* motion_averages.py

	will get an average across runs for a task, and plot it. YOU CAN'T JUST RUN IT AS IS
	YOU WILL HAVE TO LOOK AT THE SCRIPT AND MAKE THE REQUIRED CHANGES.

* outliers_per_cond.py

	will try to tell you on what onset an outlier occured during the scanning session. 
	YOU CAN'T JUST RUN IT AS IS YOU WILL HAVE TO LOOK AT THE SCRIPT AND MAKE THE REQUIRED CHANGES.

* dicom.sh

	YOU WILL NEED TO UPDATE THE SUBJECTS INSIDE THE SCRIPT
	this will run the dicom conversion. it calls dicomconvert2.py
	and uses the heuristic.py file

* onset_gen.ipynb

	this notebook creates the onset folders and files
	UPDATE THE SUBJECTS INSIDE THE NOTEBOOK

* recon.sh

	will run the recon-all -all for the subjects
	UPDATE THE SUBJECTS INSIDE THE FILE


