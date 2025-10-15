Step 1:


Satellites send signals

GPS satellites (about 20–30 active at any time) constantly broadcast radio signals.

Each signal contains two key things:

The exact time the signal was sent (from the satellite’s atomic clock).

The satellite’s position in space at that moment.



Step 2:

Your receiver listens

Your phone (or car GPS, etc.) listens to several satellites at once.

The signals all arrive at slightly different times because each satellite is at a different distance from you.



Step 3:

Measuring distance

Since radio waves travel at the speed of light, the receiver can calculate:

distance=time delay×speed of light
distance=time delay×speed of light

So if the signal took 0.07 seconds to reach you, that satellite is about 21,000 km away (since satellites orbit ~20,000 km up).

This gives you the distance to each satellite.




Step 4:

Triangulation (really “trilateration”)

If you know you are:

21,000 km from satellite A,

20,500 km from satellite B,

22,300 km from satellite C,
then your position must be at the point in space where these 3 spheres overlap.

In practice, you need at least 4 satellites:

3 to pinpoint your 3D position (latitude, longitude, altitude).

1 extra to correct your phone’s clock (because your phone’s clock isn’t as accurate as the satellites’ atomic clocks).



Step 5:


Your location

Once the receiver solves those equations, it knows your position on Earth to within a few meters (or even centimeters if you use corrections like RTK).
