zdir =  '/Users/joe/Projects/QSOClustering/tex/QPQ9/Analysis/Redshifts/zem/'
qadir = '/Users/joe/Projects/QSOClustering/tex/QPQ9/Analysis/Redshifts/zQA/'
inputfile = '/Users/joe/Projects/QSOClustering/tex/QPQ9/Analysis/Redshifts/input_file.txt'
readcol, inputfile, filename, zguess, format = 'A,F', comment = '#'
name1 = fileandpath(filename)
name2 = repstr(name1, '.fits.gz', '')
name = repstr(name2, '.fits', '')
nqsos = n_elements(name)

dropbox = '~/Dropbox/QSOPairs/data/'

file = fileandpath(filename, path = path)
file_from = file
file_to = file
inirspec = WHERE(strmatch(file_to, '*NIRSPEC*'))
nirspec_files = file_to[inirspec]
nirspec_files = repstr(nirspec_files,  '_NIRSPEC', '')
file_to[inirspec] = nirspec_files

lines = strarr(nqsos)
FOR ii = 0L, nqsos-1L DO BEGIN
   IF strmatch(path[ii], '*niri*') THEN inst_path = 'NIRI_redux/' $
   ELSE IF strmatch(path[ii], '*NIRSPEC*') THEN inst_path = 'NIRSPEC_redux/' $
   ELSE IF strmatch(path[ii], '*gnirs*') THEN inst_path = 'GNIRS_redux/' $
   ELSE IF strmatch(path[ii], '*ISAAC*') THEN inst_path = 'ISAAC_redux/' $
   ELSE IF strmatch(path[ii], '*Triplespec*') THEN $
      inst_path = 'TRIPLESPEC_redux/' $
   ELSE IF $
      strmatch(path[ii], '*XSHOOTER*') THEN inst_path = 'XSHOOTER_redux/' $
   ELSE message, 'Unrecognized instrument'
   newfile = dropbox + inst_path + file_to[ii]
   command = 'cp ' + path[ii] + file_from[ii] + '  ' + newfile 
   lines[ii] = newfile
   temp = file_search(newfile, /test_regular, count = nfound)
   IF nfound EQ 0 THEN spawn, command $ 
   ELSE print, 'File already there'
ENDFOR
lines = string(strcompress(lines, /rem), '-A80')
forprint, lines, zguess, format = '(-A80,"  ",F5.3)', textout = './new_input.txt'

END
