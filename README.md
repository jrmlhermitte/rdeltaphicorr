## Angular Correlations
This code is intended for the calculation of angular correlations on X-ray
diffraction data.
See examples for usage.

## Sample Usage
Usage is as follows:
```python
    from rdeltaphicorr.rdpc import RDeltaPhiCorrelator
    norbins = 800
    nophibins = 400
    # load a mask
    # load an image
    rdpc = RDeltaPhiCorrelator(img.shape,origin=origin,mask=mask,rbins=nobins, phibins=nophibins)
    # imgs is a 3D array (series of 2D images)
    rdpc.run(imgs)
    # result is stored in rdpc.rdeltaphiavg_n
    plt.imshow(rdpc.rdeltaphiavg_n);
    plt.clim(0,1);
```
