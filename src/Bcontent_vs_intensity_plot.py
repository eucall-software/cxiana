import pylab
import time

pylab.ion

# Start "event loop" ;)

while True:
    # Read data.
    data= pylab.loadtxt('Bcontent_vs_intensity.txt')

    #runs = data[:,0]
    #intensities = data[:,1]
    #B_content = data[:,2]
    #delta_B_content = data[:,3]

    intensities = pylab.arange(10)
    B_content = pylab.random(10)
    delta_B_content = pylab.random(10)*0.1

    pylab.errorbar(x=intensities, y=B_content, yerr=delta_B_content, fmt="s")

    pylab.pause(1)
    pylab.clf()
