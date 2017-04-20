# pyfpm
Interface for the FPM microscope

== Muestra 2017-04-05_113601.npy
La de NA = 0.12 con la que hice la optimización de los parámetros del iluminador.
Para centrarme en la pupila lo recorto con crop_image(image_dict[(theta, phi)], [150, 150], 170, 245).
Parametros ajustados:
pc.heigth = 88
pc.platform_tilt = [0, 2.1]
pc.source_center = [0, 1.9]
theta_sim_offset = 24
objective_na: 0.12
wavelength: 450E-9
pixel_size: 2.15E-6

== Scripts

acquire_complete_set.py    Adquiere un set completo de imagenes de fpm con los parametros dados por config.yaml
view_and_analyze.py        Visualiza las imágenes adquiridas y las compara con las simuladas
generate_sampling.py       Genera imágenes simuladas para reconstruir luego
optimize.py                Primera prueba de aproximación e parámetros de desperfectos del iluminador.
