from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, OffFile, labelPeak, labelPeaks
import os
import unittest as ut

d = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(d,"data_for_test","20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")

class TestOff(ut.TestCase):
    def test_open_file_with_mmap_projectors_and_basis(self):
        filename = os.path.join(d,"data_for_test/off_with_binary_projectors_and_basis.off")
        f = OffFile(filename)
        self.assertAlmostEquals(f.projectors[0,0],1.124)
        self.assertAlmostEquals(f.projectors[0,3],0)
        self.assertAlmostEquals(f.basis[0,0],1)
        self.assertAlmostEquals(f.basis[3,0],0)
        self.assertAlmostEquals(f[0]["framecount"],123456)
        self.assertAlmostEquals(f[0]["residualStdDev"],0.123456)
        self.assertAlmostEquals(len(f),1)

    def test_open_file_with_base64_projectors_and_basis(self):
        filename = os.path.join(d,"data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
        f = OffFile(filename)



if __name__ == '__main__':
    ut.main()
