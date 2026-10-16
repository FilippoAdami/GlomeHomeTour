"""Distance-fit and full-scene level-collapse regressions."""
import sys
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06_mesh_elaboration'))
from level_footprints import fit_distance_footprint, collapse_level


def test_distance_fit_removes_unsupported_recess_but_keeps_evidenced_opening():
    initial=Polygon([(0,0),(3,0),(3,3),(0,3),(0,2),(.17,2),(.17,1),(0,1)])
    observed=Polygon([(0,0),(3,0),(3,3),(0,3),(0,2),(.03,2),(.03,1),(0,1)])
    fitted,scores=fit_distance_footprint(initial,observed)
    assert len(fitted.exterior.coords)-1 == 4
    assert fitted.covers(Point(.05,1.5))
    assert scores[0]['fitted_mean_squared_distance_m2'] < scores[0]['initial_mean_squared_distance_m2']
    edges=np.diff(np.array(fitted.exterior.coords),axis=0)
    assert np.min(np.linalg.norm(edges,axis=1)) >= .1-1e-7
    assert np.max(np.min(abs(edges),axis=1)) < 1e-8
    supported=Polygon([(0,0),(3,0),(3,3),(0,3),(0,2),(.4,2),(.4,1),(0,1)])
    fitted,_=fit_distance_footprint(supported,supported)
    assert not fitted.covers(Point(.2,1.5))
    assert fitted.symmetric_difference(supported).area < .01


def test_distance_fit_is_translation_invariant():
    poly=Polygon([(0,0),(3,0),(3,3),(0,3)])
    observed=Polygon([(.02,0),(3.01,0),(3.01,3),(.02,3)])
    fitted,_=fit_distance_footprint(poly,observed)
    shifted,_=fit_distance_footprint(translate(poly,xoff=123,yoff=-57),translate(observed,xoff=123,yoff=-57))
    assert shifted.symmetric_difference(translate(fitted,xoff=123,yoff=-57)).area < 1e-5


def test_level_collapse_uses_all_scene_faces_and_rejects_other_storeys():
    vertices=np.array([[0,y,0] for y in [0.,3.,6.]])
    faces=np.array([[0,0,0],[1,1,1],[2,2,2]])
    centers=vertices.copy();normals=np.array([[0.,1.,0.]]*3)
    floor={'name':'upper','equation':np.array([0.,1.,0.,-3.])}
    ceiling={'equation':np.array([0.,-1.,0.,5.9]),'footprint':Polygon([(-1,-1),(1,-1),(1,1),(-1,1)])}
    selected=collapse_level(vertices,faces,normals,centers,floor,[ceiling])
    np.testing.assert_array_equal(selected,faces[1:2])
