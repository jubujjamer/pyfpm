# pyfpm
Interface for the FPM microscope

## Samples

1. 2017-04-05_113601.npy

**Description:** NA = 0.12 with illumination parameters optimization. Pupil centering is made with 
`<crop_image(image_dict[(theta, phi)], [150, 150], 170, 245).>`

**Adjusted parameters**
* pc.heigth = 88
* pc.platform_tilt = [0, 2.1]
* pc.source_center = [0, 1.9]
* theta_sim_offset = 24
* objective_na: 0.12
* wavelength: 450E-9
* pixel_size: 2.15E-6

## Scripts

* acquire_complete_set.py   Acquires acomplete set of images with the parameters set in config.yaml.
* view_and_analyze.py       Visualize and compare with teh simulated images.
* generate_sampling.py      Simulated images useful for analysis and reconstruction tests.
* optimize.py               Optimizes parameter of the illuminator (shift, rotation, etc.).
