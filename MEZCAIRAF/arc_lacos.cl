string inimg, outimg, mskimg
struct *inlist, *outlist, *msklist

inlist  = "Arcs_trim"
outlist = "Arcs_cr"
msklist = "Arcs_msk"

lacos_spec.gain    = 1.
lacos_spec.readn   = 5.
lacos_spec.xorder  = 9
lacos_spec.yorder  = 0
lacos_spec.sigclip = 4.5
lacos_spec.sigfrac = 0.3
lacos_spec.objlim  = 7.0
lacos_spec.niter   = 4

print ("Iniciando L.A.Cosmic...")

while (fscan(inlist,inimg) != EOF) {
    if (fscan(outlist,outimg) == EOF)
        break
    if (fscan(msklist,mskimg) == EOF)
        break

    print ("Procesando "//inimg//" -> "//outimg)

    lacos_spec (inimg, outimg, mskimg)
}

print ("Terminado.")
