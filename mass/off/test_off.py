import os
import mass.off
from mass.off import OffFile
import unittest as ut
import resource

d = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")


class TestOff(ut.TestCase):
    def test_off_imports(self):
        self.assertIsNotNone(mass.off)

    def test_open_file_with_mmap_projectors_and_basis(self):
        filename = os.path.join(d, "data_for_test/off_with_binary_projectors_and_basis.off")
        f = OffFile(filename)
        self.assertAlmostEqual(f.projectors[0, 0], 1.124)
        self.assertAlmostEqual(f.projectors[0, 3], 0)
        self.assertAlmostEqual(f.basis[0, 0], 1)
        self.assertAlmostEqual(f.basis[3, 0], 0)
        self.assertAlmostEqual(f[0]["framecount"], 123456)
        self.assertAlmostEqual(f[0]["residualStdDev"], 0.123456)
        self.assertAlmostEqual(len(f), 1)

    def test_open_file_with_base64_projectors_and_basis(self):
        filename = os.path.join(d, "data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
        self.assertIsNotNone(OffFile(filename))

    def test_mmap_many_files(self):
        """Open 1 more OFF file objects than the system allows. Test that close method closes them."""
        maxfiles, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        N = maxfiles//2 + 1

        filename = os.path.join(d, "data_for_test/off_with_binary_projectors_and_basis.off")
        files = []  # hold on to the OffFile objects so the garbage collector doesn't close them.
        for _ in range(N):
            f = OffFile(filename)
            self.assertGreater(f.nRecords, 0)
            files.append(f)
            f.close()
            print(len(files), " open files so far.")

if __name__ == '__main__':
    ut.main()
