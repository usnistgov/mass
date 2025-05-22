import resource
from mass.off import util, ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks, NoCutInds, OffFile
import os

def test_open_many_OFF_files():
    """Open more OFF ChannelGroup objects than the system allows. Test that close method closes them.
    This is in it's own file so it is easier to skip the test on windows. resource doesnt exist on windows."""
    try:
        dirname = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        dirname = os.getcwd()
    # LOWER the system's limit on number of open files, to make the test smaller
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    request_maxfiles = min(60, soft_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (request_maxfiles, hard_limit))
    try:
        maxfiles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        NFilesToOpen = maxfiles // 2 + 10

        filename = os.path.join(dirname, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
        filelist = getOffFileListFromOneFile(filename, maxChans=2)
        for _ in range(NFilesToOpen):
            _ = ChannelGroup(filelist, verbose=True, channelClass=Channel,
                                excludeStates=["START", "END"])

        # Now open one ChannelGroup with too many files. If the resources aren't freed, we can
        # only open it once, not twice.
        NFilePairsToOpen = (maxfiles - 12) // 6
        filelist *= NFilePairsToOpen
        for _ in range(3):
            _ = ChannelGroup(filelist, verbose=True, channelClass=Channel,
                                excludeStates=["START", "END"])

    # Use the try...finally to undo our reduction in the limit on number of open files.
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

def test_mmap_many_files():
    """Open more OFF file objects than the system allows. Test that close method closes them."""
    files = []  # hold on to the OffFile objects so the garbage collector doesn't close them.

    # LOWER the system's limit on number of open files, to make the test smaller
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    request_maxfiles = min(60, soft_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (request_maxfiles, hard_limit))
    try:
        maxfiles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        NFilesToOpen = maxfiles // 3 + 10

        filename = os.path.join(d, "data_for_test/off_with_binary_projectors_and_basis.off")
        for _ in range(NFilesToOpen):
            f = OffFile(filename)
            assert f.nRecords > 0
            files.append(f)
            f.close()

    # Use the try...finally to ensure that the gc can close files at the end of this test,
    # preventing a cascade of meaningless test failures if this one fails.
    # Also undo our reduction in the limit on number of open files.
    finally:
        del files
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))