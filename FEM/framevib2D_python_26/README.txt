Framevib2D
----------

Framevib2D is a Python package for multi-degree-of-freedom vibration analysis.
Indexing follows classic FEM theory with dofs, node and element numbers, etc. starting at 1 (in contrast to Python indexing from 0).


### INSTALLATION

Download the zip file framevib2D.zip, extract it and place it in your path.


### CONTENT

Driver    : framevib2d.py
Functions : assem.py
			constidx.py
			kbeam.py
			kebeam.py
			mbeam.py
			mebeam.py
			nebeam.py
			ubeam.py
			plotelem.py
			plotelemdisp.py


### USAGE

Run the package from the driver file framevib2d.
Fill out the input data or import from separate file under INPUT DATA and run.
The driver then establishes the system matrices and solves the generalized eigenvalue problem.

For topology visualization and post-processing, the package contains the plot functions plotelem() and plotelemdisp(), which plots the topology and the deformed geometry, respectively.

### REVISIONS for F22

A loop that add extra masses and constructs the geometric stiffness matrix is included. Furthermore, kebeam includes soil stiffness matrix.

