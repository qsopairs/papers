pro oiii_redshifts
;; Script to generate the zem files (and Joes QA)
path = '~/Documents/papers/QPQ9/Analysis/Redshifts/'
;path = '~/Documents/papers/QPQ9/Analysis/Redshifts/QPQ_on_HW/'

zdir =  path+'zem/'
qadir = path+'zQA/'
inputfile = path+'new_input.txt'
oiii_vdiff = 200. ; km/s
pk2c = 0.25
min_sn = 8
;
readcol, inputfile, filename, zguess, inflg, format = 'A,F,I', comment = '#'
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
   flux = x_readspec(filename[ii], wav = wave, sig = sig, inflg = inflg)
   loglam = alog10(wave)
   ivar = (sig GT 0)/(sig^2 + (sig LE 0.0))
;   if strmatch(filename[ii],'*023946.43-010640.5*') then min_sn = 50 ; pick MgII
   if strmatch(filename[ii],'*091338.30-010708.7*') then min_sn = 60 ; pick MgII
;   if strmatch(filename[ii],'*023946.43-010640.5*') then exclude_line = [1,0,0,0,1,0,0,0,0,0]
;   min_sn = 8
   zsys = zsys_driver(loglam, flux, ivar, zguess[ii] $
                      , zstruct = zstruct, out_struct = out_struct $
                      , /lya_peak $
                      , VEDG = 1000.0d $
                      , OIII_VDIFF = oiii_vdiff $
                      , pk2c = pk2c $
                      , qafile = qadir + name[ii] + '_zQA.ps' $
                      , MIN_SN = MIN_SN $
                      , MAX_X2 = 60.0D $
                      , exclude_line = exclude_line)
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



