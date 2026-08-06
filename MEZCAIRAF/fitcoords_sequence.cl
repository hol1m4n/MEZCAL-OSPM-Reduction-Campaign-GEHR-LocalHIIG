string inimg, outimg
struct *inlist, *outlist

inlist  = "Arcs_crwoFormat"
outlist = "Arcs_idFitCoor"

fitcoords.interactive = yes
fitcoords.combine = no
fitcoords.database = "database"
fitcoords.deletions = "deletions.db"
fitcoords.function = "chebyshev"
fitcoords.xorder = 4
fitcoords.yorder = 3
fitcoords.logfiles = "STDOUT,logfile"
fitcoords.plotfile = "plotfile"
fitcoords.graphics = "stdgraph"

print ("Iniciando Fitcoords...")

while (fscan(inlist,inimg) != EOF) {
    if (fscan(outlist,outimg) == EOF)
        break

    print ("Procesando "//inimg//" -> "//outimg)

    fitcoords images=(inimg) fitname=(outimg)
}

print ("Terminado.")
