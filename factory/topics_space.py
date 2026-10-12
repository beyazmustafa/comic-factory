"""Composed space topics for simulation Shorts: real events and objects,
each told in three stages with the visual that explains it best — a NASA
photograph, a true-scale comparison, a timeline, a light-travel clock, a
cutaway, or an orbit simulation. The physics and the stage facts are fixed
here; the writer only phrases them.

Colours are the house palette; stats are (value, colour, unit) rows."""

from __future__ import annotations

SUN = (255, 220, 110)
ORANGE = (255, 160, 90)
BLUE = (120, 220, 255)
PINK = (255, 110, 170)
VIOLET = (170, 140, 255)
GREEN = (120, 255, 190)
INK = (235, 236, 250)
RED = (255, 90, 90)
GREY = (160, 165, 200)

TOPICS = [
    {
        "key": "betelgeuse", "backdrop": "Orion nebula Hubble", "title": "When Betelgeuse dies.",
        "brief": "Betelgeuse, the red supergiant in Orion, about 640 light-years away, is near the end of its life and will explode as a supernova within roughly 100,000 years; it would shine as bright as the half Moon for weeks.",
        "caption": "Red supergiant · Orion",
        "stats": [{"id": "time", "label": "DISTANCE · LIGHT-YEARS"}, {"id": "right", "label": "RADIUS · SUNS"}],
        "phases": [
            {"fact": "A red supergiant in Orion, seven hundred times wider than the Sun.", "scene": {"type": "image", "query": "Betelgeuse star", "focus": [0.5, 0.5]},
             "stats": {"time": [("640", INK, "")], "right": [("764", ORANGE, "")]}},
            {"fact": "Placed where the Sun is, it would reach past Jupiter's orbit.", "scene": {"type": "scale", "items": [("Sun", 1, SUN), ("Betelgeuse", 764, ORANGE)], "unit": "solar radii"},
             "stats": {"time": [("640", INK, "")], "right": [("764", ORANGE, "")]}},
            {"fact": "Its core is running out of fuel; it will explode within a hundred thousand years.", "scene": {"type": "timeline", "events": [(-10000000, "Born", BLUE), (0, "Today", INK), (100000, "Supernova", RED)], "unit": "years"},
             "stats": {"right": [("100,000", RED, "yrs")]}},
        ],
    },
    {
        "key": "sun-light-delay", "backdrop": "solar flare SDO", "title": "The Sun you see is old.",
        "brief": "Light takes 8 minutes 20 seconds to reach Earth from the Sun; what we see is the Sun as it was. If it vanished, we would not know for eight minutes; Neptune would not know for four hours.",
        "caption": "Speed of light · 299,792 km/s",
        "stats": [{"id": "time", "label": "LIGHT TRAVEL TIME"}, {"id": "right", "label": "DISTANCE · MILLION KM"}],
        "phases": [
            {"fact": "Light leaves the Sun's surface at three hundred thousand kilometres per second.", "scene": {"type": "image", "query": "sun surface SDO", "focus": [0.5, 0.5]},
             "stats": {"right": [("150", SUN, "")]}},
            {"fact": "It reaches Earth eight minutes and twenty seconds later; we always see the Sun of the past.", "scene": {"type": "light", "a": ("Sun", SUN, 46), "b": ("Earth", BLUE, 14), "seconds": 500, "span": 20},
             "stats": {"right": [("150", SUN, "")]}},
            {"fact": "At Neptune the same light arrives four hours later.", "scene": {"type": "light", "a": ("Sun", SUN, 46), "b": ("Neptune", VIOLET, 18), "seconds": 15000, "span": 14},
             "stats": {"right": [("4,500", VIOLET, "")]}},
        ],
    },
    {
        "key": "jupiter-inside", "backdrop": "Jupiter clouds JunoCam", "title": "Inside Jupiter.",
        "brief": "Jupiter has no surface: clouds of ammonia, then hydrogen gas thickening into liquid, then metallic hydrogen under crushing pressure, around a hot, partly dissolved core.",
        "caption": "Gas giant · no solid surface",
        "stats": [{"id": "time", "label": "RADIUS · KM"}, {"id": "right", "label": "CORE TEMPERATURE · °C"}],
        "phases": [
            {"fact": "Jupiter's clouds, seen by the Juno spacecraft, are the top of a world with no ground.", "scene": {"type": "image", "query": "Jupiter Juno clouds", "focus": [0.5, 0.5]},
             "stats": {"time": [("69,911", INK, "")]}},
            {"fact": "Below the clouds the gas thickens into liquid hydrogen, then into metal under pressure.", "scene": {"type": "cutaway", "layers": [("Ammonia clouds", 1.0, (230, 200, 170)), ("Liquid hydrogen", 0.9, ORANGE), ("Metallic hydrogen", 0.78, VIOLET), ("Core", 0.2, RED)]},
             "stats": {"time": [("69,911", INK, "")], "right": [("24,000", RED, "")]}},
            {"fact": "The core is hotter than the surface of the Sun, and it is slowly dissolving into the metal around it.", "scene": {"type": "image", "query": "Jupiter Great Red Spot Juno", "focus": [0.5, 0.5]},
             "stats": {"time": [("69,911", INK, "")], "right": [("24,000", RED, "")]}},
        ],
    },
    {
        "key": "voyager", "backdrop": "Pale Blue Dot Voyager", "title": "Voyager 1.",
        "brief": "Launched in 1977, Voyager 1 is the farthest human-made object, about 24 billion km away in interstellar space; its radio signal takes over 22 hours to reach Earth; its power will run out around 2030.",
        "caption": "Interstellar space · launched 1977",
        "stats": [{"id": "time", "label": "SIGNAL DELAY"}, {"id": "right", "label": "DISTANCE · BILLION KM"}],
        "phases": [
            {"fact": "Launched in 1977, it passed Jupiter, Saturn, and left the planets behind.", "scene": {"type": "timeline", "events": [(1977, "Launch", INK), (1979, "Jupiter", ORANGE), (1980, "Saturn", SUN), (2012, "Interstellar", VIOLET), (2030, "Silent", GREY)], "unit": "years", "span": 16},
             "stats": {"right": [("24", VIOLET, "")]}},
            {"fact": "It carries a golden record of Earth's sounds; today it is twenty-four billion kilometres away.", "scene": {"type": "image", "query": "Voyager golden record", "focus": [0.5, 0.5]},
             "stats": {"right": [("24", VIOLET, "")]}},
            {"fact": "Its signal takes twenty-two hours to reach us; around 2030 its power runs out and it goes quiet.", "scene": {"type": "light", "a": ("Voyager 1", VIOLET, 10), "b": ("Earth", BLUE, 14), "seconds": 80000, "span": 16},
             "stats": {"right": [("24", VIOLET, "")]}},
        ],
    },
    {
        "key": "planet-sizes", "backdrop": "Jupiter Great Red Spot", "title": "The planets to scale.",
        "brief": "The eight planets at true relative size: Jupiter is eleven times wider than Earth, and the Sun is ten times wider than Jupiter; all the planets together would fit inside the Sun with room to spare.",
        "caption": "True relative sizes",
        "stats": [{"id": "time", "label": "EARTH RADIUS · KM"}, {"id": "right", "label": "JUPITER · EARTHS"}],
        "phases": [
            {"fact": "The four rocky worlds: Mercury, Venus, Earth, Mars.", "scene": {"type": "scale", "items": [("Mercury", 0.38, GREY), ("Venus", 0.95, ORANGE), ("Earth", 1.0, BLUE), ("Mars", 0.53, RED)], "unit": "× Earth"},
             "stats": {"time": [("6,371", INK, "")]}},
            {"fact": "The giants: Jupiter is eleven Earths across; Saturn nine.", "scene": {"type": "scale", "items": [("Earth", 1.0, BLUE), ("Neptune", 3.9, VIOLET), ("Saturn", 9.1, SUN), ("Jupiter", 11.2, ORANGE)], "unit": "× Earth"},
             "stats": {"time": [("6,371", INK, "")], "right": [("11.2", ORANGE, "")]}},
            {"fact": "And the Sun is ten Jupiters across: a hundred and nine Earths side by side.", "scene": {"type": "scale", "items": [("Earth", 1.0, BLUE), ("Jupiter", 11.2, ORANGE), ("Sun", 109, SUN)], "unit": "× Earth"},
             "stats": {"time": [("6,371", INK, "")], "right": [("109", SUN, "suns")]}},
        ],
    },
    {
        "key": "sun-death", "backdrop": "Helix Nebula", "title": "The Sun's last days.",
        "brief": "In about five billion years the Sun swells into a red giant that swallows Mercury and Venus and boils Earth; then it sheds its outer layers into a nebula and shrinks to a white dwarf the size of Earth.",
        "caption": "Stellar evolution · G-type star",
        "stats": [{"id": "time", "label": "TIME FROM NOW · YEARS"}, {"id": "right", "label": "RADIUS · TODAY'S SUN"}],
        "phases": [
            {"fact": "The Sun is halfway through its life; it has burned hydrogen for four and a half billion years.", "scene": {"type": "timeline", "events": [(-4600, "Born", SUN), (0, "Today", INK), (5000, "Red giant", RED), (7000, "White dwarf", BLUE)], "unit": "million years", "span": 18},
             "stats": {"right": [("1", SUN, "")]}},
            {"fact": "When the hydrogen runs low, it swells two hundred times wider and swallows Mercury and Venus.", "scene": {"type": "scale", "items": [("Sun today", 1, SUN), ("Red giant", 200, RED)], "unit": "× today"},
             "stats": {"right": [("200", RED, "")]}},
            {"fact": "It sheds its outer layers into a glowing nebula, and what remains is a white dwarf the size of Earth.", "scene": {"type": "image", "query": "Helix Nebula", "focus": [0.5, 0.5]},
             "stats": {"right": [("0.01", BLUE, "")]}},
        ],
    },
    {
        "key": "iss-orbit", "backdrop": "International Space Station Earth", "title": "The station that never falls.",
        "brief": "The ISS orbits at about 400 km, moving at 7.66 km/s, circling Earth every 92 minutes; it is falling all the time but moving sideways fast enough to keep missing the ground.",
        "caption": "Low Earth orbit · 400 km",
        "stats": [{"id": "time", "label": "ALTITUDE · KM"}, {"id": "right", "label": "SPEED · KM/S"}],
        "phases": [
            {"fact": "Four hundred kilometres up, the station is still inside Earth's gravity; it weighs almost as much as on the ground.", "scene": {"type": "image", "query": "International Space Station Earth", "focus": [0.5, 0.5]},
             "stats": {"time": [("400", INK, "")], "right": [("7.66", BLUE, "")]}},
            {"fact": "It is falling, but moving sideways at seven point seven kilometres per second, so the ground curves away beneath it.", "scene": {"type": "image", "query": "ISS orbit Earth limb", "focus": [0.5, 0.5]},
             "stats": {"time": [("400", INK, "")]}},
            {"fact": "One lap takes ninety-two minutes; the crew sees sixteen sunrises a day.", "scene": {"type": "image", "query": "ISS sunrise orbit", "focus": [0.5, 0.5]},
             "stats": {"time": [("400", INK, "")], "right": [("92", BLUE, "min")]}},
        ],
    },
    {
        "key": "mars-distance", "backdrop": "Mars Perseverance landscape", "title": "How far is Mars.",
        "brief": "Mars is between 55 and 400 million km away depending on the orbits; radio signals take 3 to 22 minutes one way, so rovers must drive themselves.",
        "caption": "Earth–Mars · changing distance",
        "stats": [{"id": "time", "label": "SIGNAL DELAY"}, {"id": "right", "label": "DISTANCE · MILLION KM"}],
        "phases": [
            {"fact": "Mars, photographed by the Perseverance rover, is a world we drive on by remote control.", "scene": {"type": "image", "query": "Mars surface Perseverance", "focus": [0.5, 0.5]},
             "stats": {"right": [("225", RED, "")]}},
            {"fact": "At its closest, a radio command takes three minutes to arrive.", "scene": {"type": "light", "a": ("Earth", BLUE, 16), "b": ("Mars", RED, 10), "seconds": 183, "span": 14},
             "stats": {"right": [("55", RED, "")]}},
            {"fact": "At its farthest, twenty-two minutes; a rover cannot wait for an answer, so it drives itself.", "scene": {"type": "light", "a": ("Earth", BLUE, 16), "b": ("Mars", RED, 10), "seconds": 1340, "span": 16},
             "stats": {"right": [("400", RED, "")]}},
        ],
    },
    {
        "key": "earth-inside", "backdrop": "Earth Blue Marble", "title": "Inside the Earth.",
        "brief": "Earth's crust is thinner than an apple's skin in proportion; beneath it the mantle, the liquid iron outer core that makes our magnetic field, and a solid inner core as hot as the Sun's surface.",
        "caption": "Cutaway · 6,371 km to the centre",
        "stats": [{"id": "time", "label": "DEPTH TO CENTRE · KM"}, {"id": "right", "label": "INNER CORE · °C"}],
        "phases": [
            {"fact": "From orbit the crust looks like everything; it is less than one percent of the way down.", "scene": {"type": "image", "query": "Earth from space blue marble", "focus": [0.5, 0.5]},
             "stats": {"time": [("6,371", INK, "")]}},
            {"fact": "Beneath it, hot rock flows slowly; then a sea of liquid iron, spinning, making the magnetic field.", "scene": {"type": "cutaway", "layers": [("Crust", 1.0, (190, 170, 140)), ("Mantle", 0.99, ORANGE), ("Outer core (liquid iron)", 0.55, RED), ("Inner core (solid)", 0.19, SUN)]},
             "stats": {"time": [("6,371", INK, "")], "right": [("5,400", SUN, "")]}},
            {"fact": "At the centre, a solid iron ball as hot as the Sun's surface, kept solid by the weight of the planet above it.", "scene": {"type": "image", "query": "aurora from space station", "focus": [0.5, 0.5]},
             "stats": {"time": [("6,371", INK, "")], "right": [("5,400", SUN, "")]}},
        ],
    },
    {
        "key": "deep-field", "backdrop": "Hubble ultra deep field", "title": "A grain of sand of sky.",
        "brief": "The Hubble and Webb deep fields pointed at an apparently empty patch of sky the size of a grain of sand held at arm's length and found thousands of galaxies; the oldest light is over 13 billion years old.",
        "caption": "Deep field · thousands of galaxies",
        "stats": [{"id": "time", "label": "EXPOSURE · HOURS"}, {"id": "right", "label": "GALAXIES IN FRAME"}],
        "phases": [
            {"fact": "Astronomers pointed the telescope at a dark, empty patch of sky and kept the shutter open for days.", "scene": {"type": "image", "query": "Hubble ultra deep field", "focus": [0.5, 0.5]},
             "stats": {"time": [("270", INK, "")], "right": [("10,000", VIOLET, "")]}},
            {"fact": "Every smudge is a galaxy with billions of stars; the patch is the size of a grain of sand at arm's length.", "scene": {"type": "scale", "items": [("Full Moon", 30, GREY), ("Deep field", 2.4, VIOLET)], "unit": "arcmin"},
             "stats": {"time": [("270", INK, "")], "right": [("10,000", VIOLET, "")]}},
            {"fact": "The faintest light left its galaxy more than thirteen billion years ago, when the universe was young.", "scene": {"type": "timeline", "events": [(13800, "Big Bang", RED), (13200, "First galaxies", VIOLET), (4600, "Sun forms", SUN), (0, "Today", INK)], "unit": "million years ago", "span": 16},
             "stats": {"right": [("13,800", RED, "Myr")]}},
        ],
    },
    {
        "key": "moon-drift", "backdrop": "Moon surface LRO", "title": "The Moon is leaving.",
        "brief": "The Moon moves away from Earth by about 3.8 cm per year because tides transfer Earth's spin to the Moon's orbit; days are getting longer, and total solar eclipses will end in about 600 million years.",
        "caption": "Tidal drift · 3.8 cm per year",
        "stats": [{"id": "time", "label": "DISTANCE · KM"}, {"id": "right", "label": "DRIFT · CM PER YEAR"}],
        "phases": [
            {"fact": "Laser mirrors left by Apollo astronauts measure the Moon's distance to the millimetre.", "scene": {"type": "image", "query": "Apollo lunar laser retroreflector", "focus": [0.5, 0.5]},
             "stats": {"time": [("384,400", INK, "")], "right": [("3.8", BLUE, "")]}},
            {"fact": "The Moon raises tides; the tides drag on Earth's spin and push the Moon outward, a little every year.", "scene": {"type": "image", "query": "Earth and Moon from space", "focus": [0.5, 0.5]},
             "stats": {"time": [("384,400", INK, "")], "right": [("3.8", BLUE, "")]}},
            {"fact": "Days grow longer; in six hundred million years the Moon will look too small to fully cover the Sun.", "scene": {"type": "timeline", "events": [(-4500, "Moon forms", GREY), (0, "Today", INK), (600, "Last total eclipse", SUN)], "unit": "million years", "span": 16},
             "stats": {"right": [("3.8", BLUE, "")]}},
        ],
    },
    {
        "key": "saturn-rings", "backdrop": "Saturn rings Cassini photograph", "title": "Saturn's rings are young.",
        "brief": "Saturn's rings are mostly water ice, up to 280,000 km wide but often only about 10 metres thick; Cassini found they are probably only 100 to 400 million years old and are slowly raining onto the planet.",
        "caption": "Water ice · 10 metres thick",
        "stats": [{"id": "time", "label": "RING WIDTH · KM"}, {"id": "right", "label": "THICKNESS · METRES"}],
        "phases": [
            {"fact": "Cassini photographed the rings from inside the gap between them and the planet.", "scene": {"type": "image", "query": "Saturn rings Cassini", "focus": [0.5, 0.5]},
             "stats": {"time": [("280,000", INK, "")], "right": [("10", BLUE, "")]}},
            {"fact": "They stretch wider than the distance from Earth to the Moon, yet most are thinner than a house is tall.", "scene": {"type": "scale", "items": [("Thickness", 0.01, BLUE), ("Earth–Moon", 384, GREY), ("Ring width", 280, SUN)], "unit": "thousand km", "mode": "bars"},
             "stats": {"time": [("280,000", INK, "")], "right": [("10", BLUE, "")]}},
            {"fact": "They may be only a few hundred million years old, and they are slowly raining down onto Saturn.", "scene": {"type": "timeline", "events": [(-4500, "Saturn forms", SUN), (-300, "Rings form", BLUE), (0, "Today", INK), (300, "Rings gone", GREY)], "unit": "million years", "span": 16},
             "stats": {"right": [("10", BLUE, "")]}},
        ],
    },
]
