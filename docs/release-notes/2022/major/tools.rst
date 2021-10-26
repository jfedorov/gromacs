Improvements to |Gromacs| tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without the
   a space between the colon and number!

``gmx msd`` has been migrated to the trajectoryanalysis framework
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The tool now uses the |Gromacs| selection syntax. Rather than piping selections via stdin,
selections are now made using the "-sel" option.

This migration comes with about a 20% speedup in execution time.

TODO: Modify/Delete this segment as features are added back in.
Some rarely used features have yet to be migrated, including:

- Mass weighting of MSDs cannot currently be turned on or off. It is set to on when -mol is set, otherwise off.
- The -tensor option is not yet implemented.
- System COM removal with -rmcomm has not yet been implemented.
- B-factor writing using the -pdb option is not yet supported.

:issue:`2368`

``gmx chi`` no longer needs ``residuetypes.dat`` entries for custom residues
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The need to add the names of custom residues to ``residuetypes.dat`` has been
removed, because it served no purpose. This makes ``gmx chi`` easier to use.

``gmx do_dssp`` supports DSSP version 4
"""""""""""""""""""""""""""""""""""""""

The newer DSSP version 4 program can be used by ``do_dssp`` by specifying 
option ``-ver 4`` and setting the DSSP environement variable to the ``mkdssp``
executable path (e.g. ``setenv DSSP /opt/dssp/mkdssp``)

:issue:`4129`