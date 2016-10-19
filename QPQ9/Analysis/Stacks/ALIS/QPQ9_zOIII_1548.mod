# This is a comment line

# Change the default settings
run  ncpus     4
run  blind     False
run  nsubpix   100
#run  convergence True
#run  convcriteria 0.1
chisq  atol      0.0001
chisq  xtol      0.0
chisq  ftol      0.0
chisq  gtol      0.0
out  model     True
#out  verbose	2
out fits True
out  covar  QPQ9_zOIII_1548_covar.dat
plot  dims  1x1
plot  pages all
plot ticks True
plot ticklabels True
plot fits True
plot labels True
#plot only True

# Read in the data
data read
  QPQ9_zOIII_1548.dat  resolution=vfwhm(200) fitrange=[1500,1600] columns=[wave:0,flux:1,error:2]  specid=1
data end

# Read in the model
model read
	fix	voigt	temperature	True
	emission
                constant        1.0    specid=1
#                constant         0.96395954CNS  specid=1 
	absorption
                voigt  ion=12C_IV    12.0  -0.00665   300.0    0.0e+04TA  specid=1
                voigt  ion=12C_IV    13.0   0         300.0    0.0e+04TA  specid=1
                voigt  ion=12C_IV    14.0   0.00167   300.0    0.0e+04TA  specid=1
                voigt  ion=12C_IV    12.0   0.00669   300.0    0.0e+04TA  specid=1
#                voigt  ion=12C_IV    14.4131365N   0   574.561620B   0.0e+04TA  specid=1
model end
