import numpy as np

from tools.eval.ceiling_line_registration import normal_design,theta_matrix,line_shift,fit_normal_affine


def test_normal_constraints_equal_projected_affine_displacements():
    rng=np.random.default_rng(10)
    points=rng.uniform(30,480,(40,2))
    angle=rng.uniform(-np.pi,np.pi,40)
    normals=np.c_[np.cos(angle),np.sin(angle)]
    theta=np.array([1.,-.5,5.,.75,-1.2,-3.])
    matrix=theta_matrix(theta)
    transformed=np.c_[points,np.ones(len(points))]@matrix.T
    independent=np.sum((transformed[:,:2]-points)*normals,axis=1)
    np.testing.assert_allclose(normal_design(points,normals)@theta,independent,atol=1e-13)


def test_line_match_measures_normal_displacement_without_tangent_information():
    yy,xx=np.indices((128,128))
    raw=np.exp(-.5*((xx-64)/2.)**2)
    real=np.exp(-.5*((xx-67)/2.)**2)
    matched=line_shift(raw,real,[64,64],[1,0])
    assert matched is not None
    np.testing.assert_allclose(matched[0],3.,atol=1e-12)
    assert matched[1]>.99


def test_two_orientation_families_recover_known_six_parameter_affine():
    xx,yy=np.meshgrid([60.,160.,300.,440.],[50.,180.,320.,450.])
    points=np.repeat(np.c_[xx.ravel(),yy.ravel()],2,axis=0)
    normals=np.tile(np.eye(2),(16,1))
    truth=np.array([1.2,-.8,4.,.6,-1.,3.])
    displacement=normal_design(points,normals)@truth
    recovered=np.linalg.lstsq(normal_design(points,normals),displacement,rcond=None)[0]
    np.testing.assert_allclose(recovered,truth,atol=1e-13)


def test_robust_solver_does_not_mutate_the_fixed_design_in_its_residual():
    rng=np.random.default_rng(22)
    points=rng.uniform(40,470,(60,2));angle=rng.uniform(-np.pi,np.pi,60)
    normals=np.c_[np.cos(angle),np.sin(angle)]
    design=normal_design(points,normals);before=design.copy()
    truth=np.array([1.,-.3,4.,.2,-1.,3.])
    result=fit_normal_affine(design,design@truth)
    assert result.success
    np.testing.assert_array_equal(design,before)
    np.testing.assert_allclose(result.x,truth,atol=1e-7)
