Timeintframe
------------

Timeintframe is a Python package for time integration of frame structure equations.
The user has the option between the following time integration algorithms:
 - Linear generalized alpha (lingalpha)
 - Newmark (newmark)


### INSTALLATION

Download the zip file timeintframe.zip, extract it and place it in your path.


### CONTENT

Driver    : timeintframe.py
Functions : lingalpha.py
	    newmark.py


### USAGE

Run the package from the driver file timeintframe.
Fill out the input data or import from separate file under INPUT DATA.
Establish load array under LOAD and run.
The driver then establish the damping matrix based on Rayleigh damping (should be suppressed if damping matrix is given as input).
Time integration is then performed in order to compute the response, velocity and acceleration history.
The user should uncomment the desired time integration function call depending on the desired time integration algorithm.
Different damping parameters with respect to Rayleigh damping may be chosen under RAYLEIGH DAMPING MATRIX.
Different time integration parameters may be chosen under TIME INTEGRATION.
