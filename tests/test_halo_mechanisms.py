import numpy as np
from tools.eval.halo_mechanism_ablation import independent_paper_rgb


def test_ideal_paper_has_calibrated_endpoints_and_no_chroma_on_gray():
    k=.01;black=.002
    np.testing.assert_allclose(independent_paper_rgb(np.zeros((1,3)),k,black),1)
    np.testing.assert_allclose(independent_paper_rgb(np.full((1,3),-np.log10(k)),k,black),black,atol=1e-15)
    gray=np.repeat(np.linspace(0,2,20)[:,None],3,axis=1)
    out=independent_paper_rgb(gray,k,black)
    np.testing.assert_allclose(out[:,0],out[:,1],atol=1e-15)
    assert np.all(np.diff(out[:,0])<0)


def test_single_paper_channel_cannot_change_other_channels_in_control():
    dp=np.array([[.1,.5,1.]])
    before=independent_paper_rgb(dp,.01,.002)
    after=independent_paper_rgb(dp+np.array([[.2,0,0]]),.01,.002)
    np.testing.assert_array_equal(before[:,1:],after[:,1:])
    assert after[0,0]<before[0,0]


def test_separable_control_preserves_arbitrary_neutral_curve_exactly():
    from tools.eval.halo_mechanism_ablation import separable_paper_rgb
    class CoupledPaper:
        def reference_print(self,dp,black):
            d=np.asarray(dp).reshape(-1,3)
            return np.exp(-(d+.2*d[:,::-1]))+black
    paper=CoupledPaper();gray=np.repeat(np.linspace(0,2,21)[:,None],3,axis=1)
    np.testing.assert_array_equal(separable_paper_rgb(paper,gray,.002),paper.reference_print(gray,.002))
    dp=np.array([[.2,.6,.9]]);before=separable_paper_rgb(paper,dp,.002)
    after=separable_paper_rgb(paper,dp+[.1,0,0],.002)
    np.testing.assert_array_equal(before[:,1:],after[:,1:])
