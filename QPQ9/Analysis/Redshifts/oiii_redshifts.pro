;; Script to generate the zem files (and Joes QA)
path = getenv('QSO_DIR')+'tex/QPQ9/Analysis/Redshifts/'

zdir =  path+'zem/'
qadir = path+'zQA/'
inputfile = path+'new_input.txt'
oiii_vdiff = 200. ; km/s
pk2c = 0.25
;
readcol, inputfile, filename, zguess, format = 'A,F', comment = '#'
name1 = fileandpath(filename)
name2 = repstr(name1, '.fits.gz', '')
name = repstr(name2, '.fits', '')
nqsos = n_elements(name)
plot = 0
FOR ii = 0L, nqsos-1L DO BEGIN

   fil = file_search(filename[ii], count=nfil)
   if nfil EQ 0 then BEGIN
      print, 'No file found!', filename[ii]
      print, 'Skipping'
      continue
   endif
   ;;
   flux = x_readspec(filename[ii], wav = wave, sig = sig, inflg = 2)
   loglam = alog10(wave)
   ivar = (sig GT 0)/(sig^2 + (sig LE 0.0))
   zsys = zsys_driver(loglam, flux, ivar, zguess[ii] $
                      , zstruct = zstruct, out_struct = out_struct $
                      , /lya_peak $
                      , VEDG = 1000.0d $
                      , OIII_VDIFF = oiii_vdiff $
                      , pk2c = pk2c $
                      , qafile = qadir + name[ii] + '_zQA.ps' $
                      , MIN_SN = MIN_SN $
                      , MAX_X2 = 25.0D)
   ;; high-value of MAX_X2 because H-alpha is never well-fit by a Gaussian
   IF KEYWORD_SET(PLOT) THEN x_specplot, flux, sig, wav = wave, inflg = 2, /qso, zin = zsys, title = name[ii], /block
   ;; Write
    out_struct.spec_fil = filename[ii]
    outfile = zdir + name[ii] + '_zem.fits'
    mwrfits, out_struct, outfile, /create
    mwrfits, zstruct, outfile
ENDFOR

print, 'All done'

END



