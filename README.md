##############################################################################
#                                TODO LIST                                   #
#                                                                            #
#                     Format: Description [Owner; Deadline]                  #
##############################################################################

1. Get code linter and edit code [Jason; 12pm Sept 25th]
2. Create generic template from code and add to research template [Jason; Sept 28]
3. Add code to limit GPU usage to ^ template
4. Determine how to fix meta data embedding to unit length  
5. 

##############################################################################
#                                                                            #
#                             Directory Structure                            #
#                                                                            #
##############################################################################

./doc/          Documents, e.g., paper sections
./src/          Code
    ./src/checkpoints/       saved model parameters
    ./src/codebase/          classes and utilities
    ./src/log/               logs for visualizing learning
    ./src/out/               exploratory output after training finishes
./dat/          Data (if not too large)
./ref/          Reference materials, e.g., PDFs of papers
./etc/          Other stuff, e.g., pictures of writeboards, notes


Acknowledgement:
The above format was taken from David Blei and Dustin Tran
www.cs.columbia.edu/~blei/seminar/2016_discrete_data/notes/week_01.pdf
www.dustintran.com/blog/a-research-to-engineering-workflow
