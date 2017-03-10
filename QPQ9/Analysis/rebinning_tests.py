import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
import matplotlib.pyplot as plt

## Ryan's idea

# Setup the old and new wavelength arrays
v_old = 3.0   # velocity of old scale
n_old = 100  # number of data points
v_new = 7.6  # velocity of new scale

# Generate fake data (wavelength array to start at 5000A)
f_old = np.random.normal(1.0,0.1,n_old)
w_old = 5000.0*(1.0+v_old/299792.458)**np.arange(n_old)

# Generate new wavelength array
n_new = 1+np.int(np.log10((w_old[-1]/w_old[0]))/np.log10((1.0+v_new/299792.458)))
w_new = 5000.0*(1.0+v_new/299792.458)**np.arange(n_new)

# Create a spline representation of the old wavelength array, k=1 forces linear interpolation
spl = InterpolatedUnivariateSpline(w_old, f_old, k=1)

# linearly interpolate onto new flux array
f_new = spl(w_new)

plt.plot(w_old,f_old,'b-',drawstyle='steps')
plt.plot(w_new,f_new,'r-',drawstyle='steps')
plt.show()

# Check flux conserving
gd_old = np.where( (w_old > 5001.) & (w_old < 5004.) )[0]
gd_new = np.where( (w_new > 5001.) & (w_new < 5004.) )[0]
sum_old = np.sum( f_old[gd_old] )
sum_new = np.sum( f_old[gd_new] * v_new/v_old )
print('The sums are old={:g}, new={:g}'.format(sum_old, sum_new))

## Idea with cumsum

# Endpoints of original pixels
npix = len(w_old)
#wvl = (w_old + np.roll(w_old, 1))/2.
wvh = (w_old + np.roll(w_old, -1))/2.
#wvl[0] = w_old[0] - (w_old[1] - w_old[0])/2.
wvh[npix-1] = w_old[npix-1] + (w_old[npix-1] - w_old[npix-2])/2.

# Cumulative Sum
cumsum = np.cumsum(f_old)

# Interpolate
fcum = interp1d(wvh, cumsum, fill_value=0., bounds_error=False)

# Endpoints of new pixels
nnew = len(w_new)
#wvl = (w_old + np.roll(w_old, 1))/2.
nwvh = (w_new + np.roll(w_new, -1))/2.
nwvh[nnew-1] = w_new[nnew-1] + (w_new[nnew-1] - w_new[nnew-2])/2.
#
allnwv = np.zeros(nnew+1)
allnwv[0] = w_new[0] - (w_new[1] - w_new[0])/2.
allnwv[1:] = nwvh

# Evaluate
newcum = fcum(allnwv)
if (allnwv[-1] > wvh[-1]):
    newcum[-1] = cumsum[-1]

print(wvh[20])
print(cumsum[20])
print(fcum(wvh[20]))

print(fcum(wvh[20]))
print(fcum(wvh[21]))
print(fcum((wvh[20]+wvh[21])/2.))

# Cumulative Plot
plt.clf()
plt.plot(wvh,cumsum,'b-', drawstyle='steps')
plt.plot(allnwv,newcum,'r+')
plt.show()
# This looks goofy but I think is correct

# Rebinned flux
new_fx = (np.roll(newcum,-1)-newcum)[:-1]

# Plot
plt.plot(w_old,f_old,'b-',drawstyle='steps')
plt.plot(w_new,f_new,'r-',drawstyle='steps')
plt.plot(w_new,new_fx*v_old/v_new,'g-',drawstyle='steps')
plt.show()

