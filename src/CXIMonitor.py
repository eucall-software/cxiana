#### REMOVE ME in experiment run. ######
DEBUG = True ###########################
########################################

# Import std python modules.
from mpi4py import MPI
from optparse import OptionParser
import h5py
import math
import numpy
# Useful for debugging.

if DEBUG: import pdb


# Plotting library
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot
pyplot.ion()

from mpl_tricks import setupHistogram

# Import psana modules
from psana import DataSource, Detector

# Hit finder and peak finder libs.
from ImgAlgos.PyAlgos import PyAlgos

# C++ histogramming lib.
from pypsalg import Hist1D

# XTCav pulse shape analyzer.
from xtcav.ShotToShotCharacterization import *
### REMOVE ME in exp.
#psana.setOptions({'psana.calib-dir':'calib',
                  #'psana.allow-corrupt-epics':True})

# Angular integration
from pypsalg import AngularIntegrationM

# Setup mpi.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class CXIMonitor(object):
    """ Class for encapsulation of data analysis at CXI. """

    def __init__(self,
                 experiment,
                 run,
                 peak_finder_parameters=None,
                 use_events=None,
                 background_image=None,
                 interactive=False,
                 prefix='',
                 number_of_hot_peak_time_bins=3,
                 ):

        """ Constructor of the CXIMonitor class.
        :param experiment: The number of the experiment.
        :param run: The run number.
        :param peak_finder_parameters: Dictionary of peak finder parameters.
        :example peak_finder_parameters: {'npix_min' : 3,
                                      'npix_max' : 1000,
                                      'amax_thr' : 15,
                                      'atot_thr' : 30,
                                      'son_min'  : 10,
                                      'thr_low'  : 10,
                                      'thr_high' : 15,
                                      'radius'   : 15,
                                      'dr'       : 1.0 }
        :param use_events: List of events to include in the analysis (mainly used for debugging.
        :param background_image: Specify the image to use as background. E.g. a path to a file. None for no background subtraction.
        :param interactive: Flag to control whether interactive plot windows open or if graphics are dumped to disk."
        :param prefix : Prefix for plot filenames.
        """

        # Set intensity threshold
        self.intensity_threshold = 0.0

        # Set pixel threshold (decide whether event is good or bad depending on number of pixels that are above intensity threshold).
        self.number_of_pixel_threshold = 1
        self.number_of_hot_peak_time_bins = number_of_hot_peak_time_bins

        # Get data source.
        self.station = 'cxi'
        self.__experiment = experiment
        self.__run = run
        self.__use_events = use_events
        self.__interactive = interactive
        self.__prefix = prefix

        # Sample-detector distance.
        self.__z = 0.0794 # (m)
        # FEL photon energy.
        self.__photon_energy = 1.0e4 # (eV)

        # Set default peak finder parameters if none given.
        if peak_finder_parameters is None:
            peak_finder_parameters = {'npix_min' : 3,
                                      'npix_max' : 1000,
                                      'amax_thr' : 15,
                                      'atot_thr' : 30,
                                      'son_min'  : 10,
                                      'thr_low'  : 10,
                                      'thr_high' : 15,
                                      'radius'   : 15,
                                      'dr'       : 1.0 }

        self.__peak_finder_parameters = peak_finder_parameters

        # Handle default run = latest in experiment.
        ### TODO: take last run
        if self.__run is None:
             self.__run = 1

        # Get the data source.
        ### TODO at experiment: Use Fast-Feedback disks (ffb)
        data_source = DataSource('exp=%s%s:run=%d:smd' % (self.station, self.experiment, self.run) )

        # Initialize the detector interfaces.
        ### Update for experiment.
        # CSPAD
        self.cspad = Detector('CxiDs1.0:Cspad.0')
        # Pulse energy.
        self.bld   = Detector("FEEGasDetEnergy")
        # Beam parameters.
        self.beam = Detector('EBeam') # mJ.
        ### TODO: Gas attenuation data.
        if DEBUG: self.gas_attenuator_device = None

        # Initialize background.
        self.__background_image = None
        if background_image is not None:
            self.__background_image = loadBackgroundImage(background_image)
        self.__background_roi = (numpy.arange(710, 875), numpy.arange(210, 375) )

        # Pixel size on cspad is 110 microns.
        self.__pixel_size = 110.0e-6

        # Keep only values in region of interest.
        ### TODO: Update according to actual geometry at exp. Could also set to 20-30 deg.
        min_angle = 5.0
        max_angle = 30.0

        self.min_radius = self.__z * math.tan(min_angle / 180.0 * math.pi )
        self.max_radius = self.__z * math.tan(max_angle / 180.0 * math.pi )

        # Binning for radial histograms.
        self.number_of_bins = 16
        self.bin_edges = numpy.linspace( self.min_radius, self.max_radius, self.number_of_bins + 1)

        # Take reference to events.
        self.events = data_source.events()

        # Initialize the algorithms.
        self.hit_finder = PyAlgos()
        self.peak_finder = PyAlgos()
        self.peak_finder.set_peak_selection_pars(npix_min=self.__peak_finder_parameters['npix_min'],
                                                 npix_max=self.__peak_finder_parameters['npix_max'],
                                                 amax_thr=self.__peak_finder_parameters['amax_thr'],
                                                 atot_thr=self.__peak_finder_parameters['atot_thr'],
                                                 son_min=self.__peak_finder_parameters['son_min'])
        self.angular_integrator = AngularIntegrationM.AngularIntegratorM()

        # Initialize XTCAV Retrieval.
        if self.number_of_hot_peak_time_bins > 1:
            self.xtcav_retrieval=ShotToShotCharacterization();
            self.xtcav_retrieval.SetEnv(data_source.env())

    # Queries and setters.
    #experiment
    @property
    def experiment(self):
        return self.__experiment
    @experiment.setter
    def experiment(self, value):
        self.__experiment=value

    #run
    @property
    def run(self):
        return self.__run
    @run.setter
    def run(self, value):
        self.__run=value


    def analyze(self):
        """ Perform the data analysis. """

        # Initialize containers.
        indices = [] # Index of event.
        good_event_indices = [] # Index of "good" events (above threshold)
        intensities = [] # Total intensities of "good" events.
        peaks = [] # Peaks in each event, stored as pixel coordinates, max signal, total signal
        radii_and_weights = []
        pulse_energies = []
        weightsAB = []

        r_histogram = [None for i in range(self.number_of_hot_peak_time_bins)]  # Histrogrammed peak radii.
        image = None # Total image (sum over all events).

        # Set read_once to True to switch on one-time reading of run-constant slow data on master.
        read_once = True
        # Loop over events.
        for nevent, event in enumerate(self.events):
            # Only continue if we actually want this event.
            if self.__use_events is not None:
                if nevent not in self.__use_events:
                    continue
            # Distribute events in round-robin fashion over mpi processes.
            if nevent % size != rank:
                continue
            print "Rank %d now processing event #%d." % (rank, nevent)

            # Get calibrated event.
            calibrated_event = self.cspad.calib(event)
            if calibrated_event is None:
                continue

            # Store good pixel mask on master process.
            if rank ==0:
                if read_once is True:
                    self.good_pixels = self.cspad.image(event,numpy.ones_like(calibrated_event))
                    ### TODO: replace by actual call to event readout on gas attenuator device.
                    if DEBUG: self.gas_attenuation = numpy.random.random(1)

                    self.relative_intensity = 1.-self.gas_attenuation
                    read_once = False

            # Get total intensity via serialized image (all data in one vector.)
            raw_image = self.cspad.raw( event )
            intensities.append( [nevent, numpy.sum( raw_image ) ] )

            # Filter event based on threshold on number of pixels above intensity threshold.
            # Uses hit finding algorithm.
            event_is_good = self.eventIsGood( calibrated_event, self.intensity_threshold )

            # If the event is good, do more fine grained analysis.
            if event_is_good:

                #Store index.
                good_event_indices.append( nevent )

                # Sum image.
                event_image = self.cspad.image(event)

                # Subtract background.
                if self.__background_image is not None:
                    event_image = self.subtractBackground(event_image)

                if image is None:
                    image = event_image
                    # Store image dimensions.
                    self.Nx, self.Ny = event_image.shape
                else:
                    image += event_image

                # Retrieve time bin index of time interval with most juice in the pulse. Only if more than 1 bins.
                if self.number_of_hot_peak_time_bins > 1:
                    self.get_hot_peak_time_bin(nevent, event)
                else:
                    time_bin = 0
                ### FOR DEBUGGING ONLY.
                if DEBUG: time_bin = nevent % self.number_of_hot_peak_time_bins

                # Find peaks.
                pks = self.peak_finder.peak_finder_v1(event_image,
                        thr_low=self.__peak_finder_parameters['thr_low'],
                        thr_high=self.__peak_finder_parameters['thr_high'],
                        radius=self.__peak_finder_parameters['radius'],
                        dr=self.__peak_finder_parameters['dr'])
                peaks.append( [nevent, pks] )

                # Perform radial peak histogramming.
                r, w = self.peakRadiiAndWeights( pks, event )
                radii_and_weights.append([nevent, r, w])

                rh = self.peakRadiusHistogram( r, w)

                # Get ratio of structure A over B for this event.
                ### TODO: find correct indices. Need to know bin edge for relevant angles.
                # b = int(math.floor((self.__z * math.tan( angle /180. * math.pi) - self.min_radius ) / (self.max_radius - self.min_radius) * self.number_of_bins))
                #angle_A =
                #angle_B =
                #bin_index_A = int(math.floor((self.__z * math.tan( angle_A /180. * math.pi) - self.min_radius ) / (self.max_radius - self.min_radius) * self.number_of_bins))
                #bin_index_B = int(math.floor((self.__z * math.tan( angle_B /180. * math.pi) - self.min_radius ) / (self.max_radius - self.min_radius) * self.number_of_bins))

                if DEBUG:
                    bin_index_A = 5
                    bin_index_B = 8

                weightsAB.append( [nevent, rh[bin_index_A], rh[bin_index_B]]  )

                # Initialize if first round, otherwise sum up the histogram.
                if r_histogram[time_bin] is None:
                    r_histogram[time_bin] = rh
                else:
                    r_histogram[time_bin] += rh

                ### Store pulse energy.
                pulse_energy = numpy.sum( [
                            self.bld.get(event).f_11_ENRC(), self.bld.get(event).f_12_ENRC(),
                            self.bld.get(event).f_21_ENRC(), self.bld.get(event).f_22_ENRC(),
                            ]) * 0.25
                pulse_energies.append( [nevent, pulse_energy] )

            # Save event index.
            indices.append(nevent)

        # End loop over events.

        # Gather all results.
        self.good_event_indices = mpiGather( good_event_indices, comm, rank )
        self.intensities = comm.gather( intensities )
        self.peak_radii_and_weights= comm.gather(radii_and_weights)
        self.weightsAB = comm.gather(weightsAB)

        # Gather peaks.
        self.peaks = comm.gather( peaks )
        # Reduce (sum up) all histograms.
        self.radial_histogram = [numpy.empty_like( r_histogram[i] ) for i in range(self.number_of_hot_peak_time_bins)]
        [comm.Reduce( r_histogram[i], self.radial_histogram[i] ) for i in range(self.number_of_hot_peak_time_bins)]

        # Reduce (sum up) all images.
        self.image_sum = numpy.empty_like(image)
        comm.Reduce( image, self.image_sum )

        ## Gather pulse energies.
        pulse_energies = comm.gather( pulse_energies )

        # Only on root. Repack the data structures.
        if rank == 0:
            # Concatenate into one array.
            self.pulse_energies = numpy.array( [ pe for pes in pulse_energies for pe in pes ] )

            # Average pulse energies.
            self.avg_pulse_energy = numpy.mean(self.pulse_energies[:,1])
            self.rms_pulse_energy = numpy.std(self.pulse_energies[:,1])
            print "pulse energy = %e +/- %e. " % (self.avg_pulse_energy, self.rms_pulse_energy)

            # Rearrange gathered data (strip artificial dimension stemming from mpi ranks.)
            self.peaks = [sp for peaks in self.peaks for sp in peaks]
            self.intensities = numpy.array([it for intensity in self.intensities for it in intensity])
            self.peak_radii_and_weights = [rw for radii_and_weights in self.peak_radii_and_weights for rw in radii_and_weights]

            self.weightsAB = [w for weightsAB in self.weightsAB for w in weightsAB]

            ### BACKUP MODE: Angular integration over summed image.
            #ai = AngularIntegrationM.AngularIntegratorM()
            #ai.setParameters(self.Nx, self.Ny,
                             #mask=self.good_pixels)
            #self.radial_bins, self.radial_intensity = ai.getRadialHistogramArrays(self.image_sum)


    def get_hot_peak_time_bin(self, nevent, event):
        """ Returns the time bin where the most pulse energy is contained."""

        # Get the XTCav data.
        if not self.xtcav_retrieval.SetCurrentEvent(event):
            print "No XTCav for event #%d." % (nevent)
            return 0

        # Get time and power.
        time,power,ok=self.xtcav_retrieval.XRayPower()
        if not ok:
            print "XTCav not ok for event #%d." % (nevent)
            return 0

        # If ok.
        else:
            # Strip unused dimension. ATTENTION: This assumes that only one run is analyzed.
            time = time[0]
            power = power[0]
            # Split up the data into number of intervals according to user input.
            length = len(power)
            fraction_length = length / self.number_of_hot_peak_time_bins

            # Determine interval boundaries.
            index_intervals = [(i*fraction_length,(i+1)*fraction_length) for i in range(self.number_of_hot_peak_time_bins) ]

            # Integrate over intervals
            weights = [numpy.sum(power[index_intervals[i][0]:index_intervals[i][1]]) for i in range(self.number_of_hot_peak_time_bins)]

            # Get max weight.
            max_weight = numpy.max(weights)

            # Get the interval with maximum weight.
            time_bin = numpy.where(weights == max_weight)[0]

            # Retrieve agreement flag.
            agreement,ok=self.xtcav_retrieval.ReconstructionAgreement()
            if not ok:
                print "XTCav agreement not ok for event #d." % (nevent)
            if abs(agreement) < 0.5:
                print "XTCav agreement = %4.3f < 0.5 for event #d." % ( agreemen, nevent)

        return time_bin


    def peakRadiusHistogram( self, radii, weights):
        """ Return a histogram over the peak radii.
        @param radii : The peak radii to histogram.
        @param weights : The weights to apply.
        """

        # Filter according to min and max radius.
        high_pass = numpy.where( radii >= self.min_radius )
        radii = radii[high_pass]
        weights = weights[high_pass]

        low_pass = numpy.where( radii <= self.max_radius )
        radii = radii[low_pass]
        weights = weights[low_pass]

        # Convert to floats
        radii = numpy.array([float(r) for r in radii])
        weights = numpy.array([float(r) for r in weights])

        histogram = Hist1D(self.number_of_bins, self.min_radius, self.max_radius)
        if weights is not None:
            histogram.fill(radii, weights)
        else:
            histogram.fill(radii)

        # Return.
        return histogram.get()


    def peakRadiiAndWeights(self, peaks, event=None ):
        """ Get the distance from the center pixel for each peak.

        @param peaks : The list of peak coordinates and amplitudes

        @param event : The event from which the peaks were taken.
        @default : None (-> center pixel coordinates will be taken as 1/2 * dimensions in x, y)
        """

        # Get center pixel coordinates.
        if event is not None:
            cpx, cpy = self.cspad.point_indexes(event, pxy_um=(0,0) )
        else:
            cpx, cpy = self.Nx / 2, self.Ny / 2

        radii = []
        weights = []
        # Loop over all peaks in this event.
        for peak in peaks:
            seg, nx, ny, npx, amax, amp = peak[0:6]
            distance_to_center_pixel = self.__pixel_size * math.sqrt( (nx-cpx)**2 + (ny-cpy)**2 )
            radii.append( distance_to_center_pixel )
            weights.append( amp )

        return numpy.array( radii ), numpy.array( weights )

    def subtractBackground( self, image ):
        """ Subtracts the background image (self.__background_image) from the given image. Applies normalization using average over background roi (self.__background_roi).
        """
        # Get background data in ROI.
        bg_roi =  self.__background_image[self.__background_roi]
        # Average.
        bg_avg = numpy.average( bg_roi)

        # Get image data in roi.
        img_roi = self.__background_image[self.__background_roi]
        # Average.
        img_avg = numpy.average( img_roi)

        # Get normalization factor.
        bg_normalization = bg_avg / img_avg

        # Normalize background to image signal level in ROI.
        bg_correction = self.__background_image / bg_normalization

        # Subtract normalized background.
        bg_corrected_image = image - bg_correction

        # Return.
        return bg_corrected_image

    def makePlots( self, save = False) :
        """ Generate plots for various parameters. Should be called only from master. """

        # Plots and printouts only on root process.
        if rank != 0:
            return

        #### Plot intensity histogram.
        self._makeIntensityHistogram()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_intensity_histogram.pdf" % (self.__prefix, self.station, self.experiment, self.run))

        #### Intensity vs. event index.
        self._makeIntensityVsEventPlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_intensity_vs_event.pdf" % (self.__prefix, self.station, self.experiment, self.run))

        ### Angular integrated sum image.
        #self._makeRadialIntensityHistogram()
        #pyplot.savefig( "%s_%s%s_run%d_radial_intensity.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        ### Summed image.
        #self._makeSummedImagePlot()
        #pyplot.savefig( "%s_%s%s_run%d_image_sum.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        ### Histogram over peak radii.
        self._makeBraggTracePlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_bragg_histogram.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        ### Pulse energy.
        self._makePulseEnergyPlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_pulse_energy.pdf" % ( self.__prefix,  self.station,  self.experiment,  self.run))

        ### Weight in ROI vs. event.
        self._makePeaksInROIPlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_peaks_in_roi.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        ### Intensity vs. pulse energy correlation plot.
        self._makeScatterPulseEnergyCorrelationPlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_intensity_vs_pulse_energy.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        ### A/B ratio vs. event no.
        self._makeABRatioPlot()
        pyplot.draw()
        pyplot.savefig( "%s_%s%s_run%d_ratioAB_vs_event.pdf" % ( self.__prefix, self.station,  self.experiment,  self.run))

        if self.__interactive:
            pyplot.show()
            raw_input("Press key to end this analysis run. (Closes all plot windows)")

    def peakReport(self):
        """ Find peaks and print out a log. Should be called only from master. """

        if rank != 0:
            return

        # Find peaks. Need a fresh PyAlgos instance.
        peak_finder = PyAlgos()
        peak_finder.set_peak_selection_pars(npix_min=2, npix_max=50, amax_thr=10, atot_thr=20, son_min=5)
        peaks = peak_finder.peak_finder_v1(self.image_sum, thr_low=5, thr_high=30, radius=5, dr=0.05)

        # Peak finding report.
        hdr = '\nSeg  Row  Col  Npix    Amptot'
        fmt = '%3d %4d %4d  %4d  %8.1f'
        print 'Found %s peaks.' % (len(peaks))
        print hdr
        for peak in peaks :
            seg,row,col,npix,amax,atot = peak[0:6]
            print fmt % (seg, row, col, npix, atot)


        # Peak report per event.
        # Get peak radii.
        hdr = '\nEvent Seg  Row  Col  Npix    Amptot'
        fmt = '%4d %3d %4d %4d  %4d  %8.1f'
        print 'Found %s peaks.' % (len(peaks))
        print hdr
        for nevent, peaks in self.peaks :
            for peak in peaks:
                seg,row,col,npix,amax,atot = peak[0:6]
                print fmt % (nevent, seg, row, col, npix, atot)


    def peakRadiiReport(self):
        """ Print out the radii and weights of peaks. """
        # Perform only on rank 0.
        if rank != 0:
            return
        filename ="%s_%s%s_run%d_peak_vs_radii.log" % ( self.__prefix, self.station,  self.experiment,  self.run)
        with open(filename, 'w') as file_handle:
            file_handle.write( "\n\n# Peak radii and weights.\n")
            for nevent, radii, weights in self.peak_radii_and_weights:
                file_handle.write( "\n#Event #%d\n" % (nevent)  )

                angles = numpy.arctan(radii / self.__z ) * 180. / numpy.pi
                for i,r in enumerate(radii):
                    file_handle.write( "%4.3f \t %4.3f\t %4.3e \n" % (r, angles[i], weights[i]) )
            file_handle.close()

    def eventIsGood( self,  event, intensity_threshold ):
        # Get number of pixels above threshold.
        number_of_pixels_above_threshold = self.hit_finder.number_of_pix_above_thr(     event, self.intensity_threshold )
        total_intensity_above_threshold  = self.hit_finder.intensity_of_pix_above_thr( event, self.intensity_threshold )
        # Save event number if enough pixels are above threshold intensity.
        good_event = number_of_pixels_above_threshold >= self.number_of_pixel_threshold

        return good_event

    def sumImage( self, image, event ):
                return image

    def _makeIntensityHistogram(self):
        # Make a histogram over intensities.
        intensity_histogram, intensity_bins = numpy.histogram( self.intensities[:,1], bins=20 )

        pyplot.figure(0)
        setupHistogram(intensity_histogram, intensity_bins)
        pyplot.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        pyplot.title( "Intensity histogram\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
        pyplot.xlabel( "Intensity (ADU) ")
        pyplot.ylabel( "Occurence")

    def _makeIntensityVsEventPlot(self):
        pyplot.figure(1)
        pyplot.plot( self.intensities[:,0], self.intensities[:,1], 's' )
        pyplot.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        pyplot.title( "Total intensity vs. event no.\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
        pyplot.ylabel( "Intensity (ADU) ")
        pyplot.xlabel( "Event no.")

    def _makePeaksInROIPlot(self):
        pyplot.figure(6)

        data = []
        for nevent, radii, weights in self.peak_radii_and_weights:
            total_weight = 0.0
            angles = numpy.arctan(radii / self.__z ) * 180. / numpy.pi
            for i,r in enumerate(radii):
                if angles[i] >= 20.0 and angles[i] <= 28.0:
                    total_weight += weights[i]
            data.append([nevent, total_weight])
        data = numpy.array(data)

        pyplot.plot( data[:,0], data[:,1], 's')
        pyplot.xlabel( "Event no." )
        pyplot.ylabel( "Total intensity in ROI" )
        pyplot.title( "Weight in ROI vs. event no.\n run # %d I=%3.2f" % (self.run, self.relative_intensity) )

        number_of_hit_events = len( [ d for d in data if d[1] > 0.0] )

        print "Found %d events of %d total events (%f rate) with >0 peak intensity in ROI." % (number_of_hit_events, len(data), number_of_hit_events/(1.0*len(data)) )

    def _makeRadialIntensityHistogram(self):
        pyplot.figure(2)
        width = 0.7*(self.radial_bins[1] - self.radial_bins[0])
        pyplot.bar(self.radial_bins, self.radial_intensity, align='center', width=width)
        pyplot.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        pyplot.title( "Radial Intensity histogram\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
        pyplot.xlabel( "Intensity (ADU) ")
        pyplot.ylabel( "Occurence")

    def _makeSummedImagePlot(self):
        pyplot.figure(3)
        pyplot.imshow( self.image_sum, cmap='YlGnBu_r')
        pyplot.title("Total image\n run # %d I=%3.2f" % (self.run, self.relative_intensity))

        pyplot.colorbar()

    def _makeBraggTracePlot(self):

        bins = numpy.arctan(self.bin_edges / self.__z ) * 180. / numpy.pi
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        if self.number_of_hot_peak_time_bins > 1:
            pyplot.figure(3)
            pyplot.title("Bragg histogram (hot peak time resolved)\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
            for h, hist in enumerate(self.radial_histogram):
                pyplot.subplot((self.number_of_hot_peak_time_bins)*100+10+h+1)
                pyplot.bar(center, hist, align='center', width=width, label='time bin=%d' % (h) )
                pyplot.legend()
                pyplot.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

            pyplot.xlabel( r"scattering angle $2\vartheta$ (deg) ")
            pyplot.ylabel( "Occurence")


        pyplot.figure(4)
        # Total histogram.
        histogram = numpy.sum(numpy.array(self.radial_histogram), axis=0)
        pyplot.bar(center, histogram, align='center', width=width )
        pyplot.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        pyplot.title("Bragg histogram\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
        pyplot.xlabel( r"scattering angle $2\vartheta$ (deg) ")
        pyplot.ylabel( "Occurence")


    def _makePulseEnergyPlot(self):
        pyplot.figure(5)
        pyplot.plot( self.pulse_energies[:,0], self.pulse_energies[:,1], "s" )
        pyplot.plot( self.pulse_energies[:,0], numpy.ones_like(self.pulse_energies[:,0]) * self.avg_pulse_energy,'--', label="Average pulse energy" )
        pyplot.title("Pulse energy vs. event no.\n run # %d I=%3.2f" % (self.run, self.relative_intensity))
        pyplot.xlabel( r"Event no.")
        pyplot.legend()
        pyplot.ylabel( "Pulse energy (mJ)")


    def _makeScatterPulseEnergyCorrelationPlot(self):
        pyplot.figure(7)

        x = []
        y = []
        for nevent, pulse_energy in self.pulse_energies:
            x.append(pulse_energy)
            index = numpy.where(self.intensities[:,0] == nevent)
            intensity = self.intensities[:,1][index]
            y.append(intensity)

        pyplot.plot( x, y, 's')
        pyplot.xlabel( " Pulse energy (mJ)")
        pyplot.ylabel( " Total intensity (ADU)" )
        pyplot.title(" Total intensity vs. pulse energy\n run # %d I=%3.2f" % (self.run, self.relative_intensity) )

    def _makeABRatioPlot(self):

        pyplot.figure(8)

        # Get indices
        indices = numpy.array([w[0] for w in self.weightsAB])

        # Get total weight.
        total_weights = numpy.array([w[1] + w[2] for w in self.weightsAB])

        # Get weight in B.
        weightsB = numpy.array([w[2]  for w in self.weightsAB])

        # Get slice where total weight is not 0.
        good_indices = numpy.where(total_weights > 0.0)

        # Get ratios B/(A+B)
        ratios = weightsB[good_indices] / total_weights[good_indices]

        # Get indices where total weight is not 0.
        event_indices = indices[good_indices]

        # Average and RMS
        avg_ratio = numpy.mean(ratios)
        rms_ratio = numpy.std(ratios)

        # Plot.
        pyplot.plot(event_indices, ratios, "s")
        pyplot.title("B content = %4.3f +/- %4.3f\n run # %d I=%3.2f" % (avg_ratio, rms_ratio, self.run, self.relative_intensity))
        pyplot.xlabel("Event no.")
        pyplot.ylabel("B content B/(A+B)")

        # Write to file.
        with open('Bcontent_vs_intensity.txt','a') as file_handle:
            file_handle.write("%d \t %6.5f \t %6.5f \t %6.5f \n" % (self.run, self.gas_attenuation, avg_ratio, rms_ratio) )
            file_handle.close()

#### Helper functions.
def mpiGather( data, comm, rank ):
    """ Gather from all processes to root."""
    data_lengths = numpy.array( comm.gather( len( data ) ) )

    # Convert to numpy array.
    data = numpy.array( data )

    all_data = None
    if rank == 0:
        all_data = numpy.empty( ( numpy.sum(data_lengths) ), data.dtype )

    # Gather data.
    comm.Gatherv( sendbuf = data, recvbuf = [all_data, data_lengths] )

    # Return.
    return all_data

def loadBackgroundImage(h5_filename):
    """ Loads background data from the specified hdf5 file.
    :param h5_filename : Relative or absolute path to the hdf5 file containing the background data.
    :return: Background image as numpy.array or None if error during read.
    """

    try:
        h5 = h5py.File(h5_filename)
        background = numpy.array(h5['data/background'])
        h5.close()
        return background
    except:
        raise
        print( "Error while reading background image from %s. Continuing w/o background subtraction." % (h5_filename) )
        return None


### The main script to run.
def main():

    parser = OptionParser()

    parser.add_option("-x", "--experiment", dest="experiment",
                      help="number of the experiment to analyze", default='k8816')
    ### TODO AT EXP: Change defaul experiment number.

    parser.add_option("-r", "--run",
                      dest="run", default=None,
                      help="The run number to analyze (defaults to last run in experiment.)")

    parser.add_option("-e", "--events",
                      dest="events", default=None,
                      help="Which events to analyze (default to all events in run.")

    parser.add_option("-i", "--interactive", action="store_true",
                      dest="interactive", default=False,
                      help="Whether to run in interactive plotting mode (generates interactive matplotlib plots).")

    parser.add_option("-b", "--background",
                      dest="background", default=None,
                      help="Background data (file) or calibration directory.")

    parser.add_option("-p", "--prefix", dest='prefix', default='', help='prefix for plot files' )


    (options, args) = parser.parse_args()

    experiment = options.experiment
    run = options.run
    if run is not None:
        run = int(run)

    events = eval(options.events)
    interactive=options.interactive

    # Switch mpl backend if necessary.
    if not interactive:
        pyplot.ioff()
        pyplot.switch_backend('pdf')

    background=options.background

    # Support for single event.
    if isinstance(events, int):
        events = [events]

    prefix=options.prefix

    # Setup peak finder parameters.
    peak_finder_parameters = {'npix_min' : 3,
                              'npix_max' : 1000,
                              'amax_thr' : 15,
                              'atot_thr' : 30,
                              'son_min'  : 10,
                              'thr_low'  : 10,
                              'thr_high' : 15,
                              'radius'   : 15,
                              'dr'       : 1.0 }

    # Construct the Analyzer.
    cxi = CXIMonitor(experiment=experiment,
                     run=run,
                     peak_finder_parameters=peak_finder_parameters,
                     use_events=events,
                     interactive=interactive,
                     background_image=background,
                     prefix=prefix,
                     number_of_hot_peak_time_bins=1,
                     )

    # Do analyze.
    cxi.analyze()

    # Plots and logs only on master.
    if rank == 0:
        cxi.peakRadiiReport()
        #cxi.peakReport()
        cxi.makePlots()

# Execute the main() function if called as a script.
if __name__ == '__main__':

    main()

    MPI.Finalize()
