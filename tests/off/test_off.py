import pytest
import os
import mass.off
from mass.off import OffFile

d = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(d, "data_for_test", "20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")


def test_off_imports():
    assert mass.off is not None


def test_open_file_with_mmap_projectors_and_basis():
    filename = os.path.join(d, "data_for_test/off_with_binary_projectors_and_basis.off")
    f = OffFile(filename)
    assert f.projectors[0, 0] == pytest.approx(1.124)
    assert f.projectors[0, 3] == pytest.approx(0)
    assert f.basis[0, 0] == pytest.approx(1)
    assert f.basis[3, 0] == pytest.approx(0)
    assert f[0]["framecount"] == pytest.approx(123456)
    assert f[0]["residualStdDev"] == pytest.approx(0.123456)
    assert len(f) == pytest.approx(1)


def test_open_file_with_base64_projectors_and_basis():
    filename = os.path.join(d, "data_for_test/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off")
    assert OffFile(filename) is not None


