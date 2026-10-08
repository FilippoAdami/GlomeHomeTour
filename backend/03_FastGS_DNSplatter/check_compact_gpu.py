#!/usr/bin/env python3
"""Small compact-box test with analytically known tile bounds."""
import time
import torch
from gsplat import rasterization
from check_gsplat_gpu import reference


def main():
    values = dict(means=torch.tensor([[0.,0.,4.]],device="cuda",requires_grad=True),
                  quats=torch.tensor([[1.,0.,0.,0.]],device="cuda"),
                  scales=torch.full((1,3),.5,device="cuda"),
                  opacities=torch.tensor([.1],device="cuda",requires_grad=True),
                  colors=torch.tensor([[.3,.5,.7]],device="cuda",requires_grad=True),
                  viewmats=torch.eye(4,device="cuda")[None],
                  Ks=torch.tensor([[[64.,0,32.],[0,64.,32.],[0,0,1.]]],device="cuda"),
                  width=64,height=64,tile_size=16,packed=False)
    for multiplier in (1., .5):
        r,a,info=rasterization(**values,footprint_multiplier=multiplier)
        # Sigma_xx = sigma_yy = (64*.5/4)^2 + .3 = 64.3.
        extent=(2*__import__("math").log(.1*255)*64.3*multiplier)**.5
        radius=__import__("math").ceil(extent)
        lo=max(0,(32-radius)//16);hi=min(4,__import__("math").ceil((32+radius)/16))
        expected_tiles=[y*4+x for y in range(lo,hi) for x in range(lo,hi)]
        got_tiles=(info["isect_ids"]>>32).cpu().tolist()
        assert got_tiles==expected_tiles,(got_tiles,expected_tiles)
        m=info["means2d"][0].detach().cpu(); q=info["conics"][0].detach().cpu()
        c=values["colors"].detach().cpu();o=values["opacities"].detach().cpu()
        want,wa=reference(m,q,c,o,torch.zeros(3),64,64)
        keep=torch.zeros(64,64,dtype=torch.bool)
        for tile in expected_tiles:
            y,x=divmod(tile,4);keep[y*16:(y+1)*16,x*16:(x+1)*16]=True
        want=torch.where(keep[...,None],want,0);wa=torch.where(keep[...,None],wa,0)
        torch.testing.assert_close(r[0].cpu(),want,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a[0].cpu(),wa,atol=2e-6,rtol=2e-5)
        grads=torch.autograd.grad(r.sum()+a.sum(),(values["means"],values["colors"],values["opacities"]))
        assert all(torch.isfinite(g).all() for g in grads)
        torch.cuda.synchronize();time.sleep(.2)
        print(f"PASS compact multiplier={multiplier}: {len(expected_tiles)} tiles",flush=True)
    for bad in (0.,-1.,2.,float('nan')):
        try:rasterization(**values,footprint_multiplier=bad)
        except ValueError:pass
        else:raise AssertionError(bad)


if __name__=="__main__":main()
