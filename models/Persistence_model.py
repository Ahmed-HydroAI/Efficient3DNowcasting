


import numpy as np


class Persistence:

    """
    The basic class of the Eulerian Persistence model (Persistence)
    of the rainymotion library.

    To run your nowcasting model you first have to set up a class instance
    as follows:

    `model = Persistence()`

    and then use class attributes to set up model parameters, e.g.:

    `model.lead_steps = 12`

    For getting started with nowcasting you must specify only `input_data`
    attribute which holds the latest radar data observations.
    After specifying the input data, you can run nowcasting model and
    produce the corresponding results of nowcasting using `.run()` method:

    `nowcasts = model.run()`

    Attributes
    ----------

    input_data: 3D numpy array (frames, dim_x, dim_y) of radar data for
                previous hours. "frames" dimension must be > 2.

    lead_steps: int, default=12
        Number of lead times for which we want to produce nowcasts. Must be > 0

    Methods
    -------
    run(): perform calculation of nowcasts.
        Return 3D numpy array of shape (lead_steps, dim_x, dim_y).

    """

    def __init__(self):

        self.input_data = None

        self.lead_steps = 12

    def run(self):
        """
        Run nowcasting calculations.

        Returns
        -------
        nowcasts : 3D numpy array of shape (lead_steps, dim_x, dim_y).

        """

        last_frame = self.input_data[-1, :, :]

        forecast = np.dstack([last_frame for i in range(self.lead_steps)])

        return np.moveaxis(forecast, -1, 0).copy()
