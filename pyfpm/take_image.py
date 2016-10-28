


def take_image(client, overlap):
    for theta, phi, power in iter_positions(client.get_pupil_size(), overlap):
        img[(theta, phi)] = (power, client.acquire(theta, phi, power))

    return img
